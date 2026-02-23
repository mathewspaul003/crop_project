from django.shortcuts import render, redirect
from django.utils import timezone
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
import numpy as np
import joblib
import os
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from .models import CropDetails, CropCycle, UserCropSession, UserCycleTask, ChatbotKnowledge, Feedback
import google.generativeai as genai
from django.conf import settings
genai.configure(api_key=settings.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# ✅ Load ML Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)


# ============================
# ✅ USER PAGES (Protected)
# ============================

def home(request):
    # ✅ Block Admin from user pages
    if request.user.is_authenticated and request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    result = None
    values = {}
    session_id = None

    if request.method == "POST":
        values = {
            "N": request.POST.get("N"),
            "P": request.POST.get("P"),
            "K": request.POST.get("K"),
            "temperature": request.POST.get("temperature"),
            "humidity": request.POST.get("humidity"),
            "ph": request.POST.get("ph"),
            "rainfall": request.POST.get("rainfall"),
        }

        user_input = np.array([[
            float(values["N"]),
            float(values["P"]),
            float(values["K"]),
            float(values["temperature"]),
            float(values["humidity"]),
            float(values["ph"]),
            float(values["rainfall"]),
        ]])

        prediction = model.predict(user_input)
        result = le.inverse_transform(prediction)[0]

        # ✅ Only Create session if user is LOGGED IN
        if request.user.is_authenticated:
            session = UserCropSession.objects.create(
                user=request.user,
                crop_name=result,
                N=float(values["N"]),
                P=float(values["P"]),
                K=float(values["K"]),
                temperature=float(values["temperature"]),
                humidity=float(values["humidity"]),
                ph=float(values["ph"]),
                rainfall=float(values["rainfall"]),
            )
            session_id = session.id

    template_name = "crop_app/home.html"
    if not request.user.is_authenticated:
        template_name = "crop_app/predict_guest.html"

    return render(
        request,
        template_name,
        {
            "result": result, 
            "values": values, 
            "session_id": session_id,
            "chatbot_crop": result
        }
    )


@login_required
def crop_cycle(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    crop = request.GET.get("crop")
    session_id = request.GET.get("session_id")
    stages = []

    if crop:
        stages = CropCycle.objects.filter(crop_name__iexact=crop).order_by("start_day")

    return render(
        request,
        "crop_app/crop_cycle.html",
        {"crop": crop, "stages": stages, "session_id": session_id}
    )


@login_required
def start_cycle(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    session_id = request.GET.get("session_id")
    tasks = []
    session = None
    current_stage = None

    if session_id:
        session = UserCropSession.objects.get(
            id=session_id,
            user=request.user
        )

        crop = session.crop_name
        stages = CropCycle.objects.filter(
            crop_name__iexact=crop
        ).order_by("start_day")

        for stage in stages:
            task, _ = UserCycleTask.objects.get_or_create(
                session=session,
                crop_cycle=stage
            )
            tasks.append(task)

        from django.utils import timezone
        days_passed = (timezone.now() - session.created_at).days + 1

        # ✅ FIND FIRST INCOMPLETE STAGE (This is the most accurate for progress tracking)
        current_stage_obj = None
        for task in tasks:
            if not task.is_completed:
                current_stage_obj = task.crop_cycle
                break

        # ✅ FALLBACK: If all stages are completed, show the last one
        if not current_stage_obj and tasks:
            current_stage_obj = tasks[-1].crop_cycle
        
        if current_stage_obj:
            current_stage = current_stage_obj.stage_name

        # ✅ IDENTIFY NEXT ALLOWED TASK (First incomplete)
        next_task_id = None
        for task in tasks:
            if not task.is_completed:
                next_task_id = task.id
                break

        # ✅ If all completed
        if not current_stage and tasks:
            current_stage_obj = tasks[-1].crop_cycle
            current_stage = current_stage_obj.stage_name

        current_stage_id = current_stage_obj.id if current_stage_obj else None

    return render(
        request,
        "crop_app/start_cycle.html",
        {
            "session": session,
            "tasks": tasks,
            "current_stage": current_stage,
            "current_stage_id": current_stage_id,
            "chatbot_crop": session.crop_name if session else None,
            "chatbot_stage": current_stage,
            "chatbot_stage_id": current_stage_id,
            "days_passed": days_passed if session else 0,
            "next_task_id": next_task_id
        }
    )


@login_required
def mark_done(request, task_id):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    # ✅ IMPORTANT: only allow user to mark their own tasks
    task = UserCycleTask.objects.get(id=task_id, session__user=request.user)

    # 🛑 CHECK: Previous stages must be completed first
    previous_incomplete = UserCycleTask.objects.filter(
        session=task.session,
        crop_cycle__start_day__lt=task.crop_cycle.start_day,
        is_completed=False
    ).exists()

    if previous_incomplete:
        from django.contrib import messages
        messages.warning(request, "Please complete previous stages first!")
        return redirect(f"/start-cycle/?session_id={task.session.id}")

    task.is_completed = True
    task.completed_at = timezone.now()
    task.save()

    return redirect(f"/start-cycle/?session_id={task.session.id}")


@login_required
def report_view(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    session_id = request.GET.get("session_id")
    analyze = request.GET.get("analyze") == "true"
    session = None
    tasks = []
    ai_analysis = None
    is_manual = False

    if session_id:
        # ✅ IMPORTANT: only allow report for their own session
        session = UserCropSession.objects.get(id=session_id, user=request.user)
        tasks = UserCycleTask.objects.filter(session=session).order_by("crop_cycle__start_day")

        # Check if manual (all metrics at 0)
        if session.N == 0 and session.P == 0 and session.K == 0:
            is_manual = True

        if analyze:
            # 🧠 AI ANALYSIS PROMPT
            progress_str = "\n".join([
                f"- {t.crop_cycle.stage_name}: {'Completed' if t.is_completed else 'Pending'}" 
                for t in tasks
            ])

            if is_manual:
                analysis_prompt = f"""
                Analyze this Agriculture Report for {session.crop_name}.
                This is a MANUAL search session (no soil data provided).
                
                GROWTH PROGRESS:
                {progress_str}
                
                INSTRUCTIONS:
                - Since soil data is missing, provide the IDEAL N-P-K and pH values for growing {session.crop_name}.
                - Provide general care suggestions for the current pending stage.
                - Keep it professional and bulleted.
                """
            else:
                analysis_prompt = f"""
                Analyze this Agriculture Crop Report and provide 3-4 professional suggestions.
                
                REPORT DATA:
                Crop: {session.crop_name}
                Soil Nutrients: N={session.N}, P={session.P}, K={session.K}
                Environment: Temp={session.temperature}°C, Humidity={session.humidity}%, pH={session.ph}, Rainfall={session.rainfall}mm
                
                GROWTH PROGRESS:
                {progress_str}
                
                INSTRUCTIONS:
                - Provide specific advice on fertilizer adjustments if NPK is imbalanced for this crop.
                - Provide a 'Current Focus' suggestion based on growth progress.
                """

            try:
                response = gemini_model.generate_content(analysis_prompt)
                if response and response.text:
                    ai_analysis = response.text.strip()
            except Exception as e:
                print(f"Report Analysis Error: {e}")
                ai_analysis = "Sorry, I couldn't generate the analysis right now."

    return render(
        request, 
        "crop_app/report.html", 
        {
            "session": session, 
            "tasks": tasks, 
            "ai_analysis": ai_analysis,
            "is_manual": is_manual
        }
    )


# ============================
# ✅ CUSTOM ADMIN PANEL
# ============================

def admin_login(request):
    # Redirect to the unified secure portal
    return redirect("/login/")


@login_required
def admin_dashboard(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        return redirect("/login/")

    from django.contrib.auth.models import User
    
    context = {
        "crop_count": CropDetails.objects.count(),
        "user_count": User.objects.filter(is_staff=False).count(),
        "feedback_count": Feedback.objects.count(),
    }

    return render(request, "admin_panel/admin_dashboard.html", context)

@login_required
def admin_users_view(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        return redirect("/login/")

    from django.contrib.auth.models import User
    users = User.objects.filter(is_staff=False).order_by('-date_joined')
    return render(request, "admin_panel/view_users.html", {"users": users})


def admin_logout(request):
    logout(request)
    return redirect("/")


# ============================
# ✅ USER AUTH PAGES
# ============================

def user_register(request):
    msg = None
    error = None

    if request.method == "POST":
        full_name = request.POST.get("full_name")
        email = request.POST.get("email")
        password = request.POST.get("password")

        if User.objects.filter(username=email).exists():
            error = "An account with this email already exists!"
        else:
            # Map email to username for unique identification
            new_user = User.objects.create_user(username=email, email=email, password=password)
            new_user.first_name = full_name
            new_user.save()
            msg = "✅ Registration successful! Please login."

    return render(request, "user/register.html", {"msg": msg, "error": error})


def user_login(request):
    error = None

    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        # Authenticate using email as the username
        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)

            # Unified Redirection Logic
            if user.is_staff:
                return redirect("/admin-panel/dashboard/")
            return redirect("/dashboard/")
        else:
            error = "Invalid email or password!"

    return render(request, "user/login.html", {"error": error})


def user_logout(request):
    logout(request)
    return redirect("/login/")


@login_required
def user_dashboard(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    return render(request, "user/dashboard.html")
@login_required
def add_crop_cycle(request):
    # ✅ Only admin can access
    if not request.user.is_staff:
        return redirect("/login/")

    msg = None

    if request.method == "POST":
        crop_name = request.POST.get("crop_name")
        stage_name = request.POST.get("stage_name")
        start_day = request.POST.get("start_day")
        end_day = request.POST.get("end_day")
        advisory = request.POST.get("advisory")

        CropCycle.objects.create(
            crop_name=crop_name.lower(),
            stage_name=stage_name,
            start_day=int(start_day),
            end_day=int(end_day),
            advisory=advisory
        )

        msg = "✅ Crop Cycle Stage Added Successfully!"

    return render(request, "admin_panel/add_crop_cycle.html", {"msg": msg})

@login_required
def admin_manage_cycles(request):
    if not request.user.is_staff:
        return redirect("/login/")

    from itertools import groupby
    from operator import attrgetter

    # Fetch all cycles ordered by crop name and start day
    all_cycles = CropCycle.objects.all().order_by('crop_name', 'start_day')
    
    # Group by crop name
    grouped_cycles = {}
    for key, group in groupby(all_cycles, key=attrgetter('crop_name')):
        grouped_cycles[key] = list(group)

    return render(request, "admin_panel/admin_manage_cycles.html", {"grouped_cycles": grouped_cycles})

@login_required
def admin_edit_cycle(request, cycle_id):
    if not request.user.is_staff:
        return redirect("/login/")

    cycle = CropCycle.objects.get(id=cycle_id)
    msg = None

    if request.method == "POST":
        cycle.crop_name = request.POST.get("crop_name").lower()
        cycle.stage_name = request.POST.get("stage_name")
        cycle.start_day = int(request.POST.get("start_day"))
        cycle.end_day = int(request.POST.get("end_day"))
        cycle.advisory = request.POST.get("advisory")
        cycle.save()
        msg = "✅ Growth Stage Updated Successfully!"

    return render(request, "admin_panel/admin_edit_cycle.html", {"cycle": cycle, "msg": msg})

@login_required
def admin_delete_cycle(request, cycle_id):
    if not request.user.is_staff:
        return redirect("/login/")

    cycle = CropCycle.objects.get(id=cycle_id)
    cycle.delete()
    return redirect("/admin-panel/cycles/")

@login_required
def admin_delete_user(request, user_id):
    if not request.user.is_staff:
        return redirect("/login/")
    
    from django.contrib.auth.models import User
    user_to_delete = User.objects.get(id=user_id)
    
    # Prevent deleting staff/admin users for safety
    if not user_to_delete.is_staff:
        user_to_delete.delete()
        
    return redirect("/admin-panel/users/")

@login_required
def my_sessions(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    sessions = UserCropSession.objects.filter(user=request.user).order_by("-created_at")

    session_data = []
    for s in sessions:
        total = UserCycleTask.objects.filter(session=s).count()
        done = UserCycleTask.objects.filter(session=s, is_completed=True).count()

        is_completed = (total > 0 and total == done)

        session_data.append({
            "session": s,
            "is_completed": is_completed
        })

    return render(request, "user/my_sessions.html", {"session_data": session_data})

def index(request):
    # If already authenticated, skip landing and go to respective hub
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect("/admin-panel/dashboard/")
        return redirect("/dashboard/")

    # Guest user → Render Pre-Login Dashboard
    return render(request, "crop_app/landing.html")
@login_required
def all_reports(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    sessions = UserCropSession.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "user/reports.html", {"sessions": sessions})

@login_required
def download_report_pdf(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    session_id = request.GET.get("session_id")

    # ✅ Security: only allow user's own session
    session = UserCropSession.objects.get(id=session_id, user=request.user)
    tasks = UserCycleTask.objects.filter(session=session).order_by("crop_cycle__start_day")

    # Check if manual
    is_manual = (session.N == 0 and session.P == 0 and session.K == 0)

    template_path = "crop_app/report_pdf.html"
    context = {
        "session": session, 
        "tasks": tasks, 
        "is_manual": is_manual
    }

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{session.crop_name}_report_{session.id}.pdf"'

    template = get_template(template_path)
    html = template.render(context)

    pisa_status = pisa.CreatePDF(html, dest=response)

    if pisa_status.err:
        return HttpResponse("❌ Error generating PDF")

    return response
@login_required
def all_tasks(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    sessions = UserCropSession.objects.filter(user=request.user).order_by("-created_at")

    task_data = []
    for s in sessions:
        total_tasks = UserCycleTask.objects.filter(session=s).count()
        completed_tasks = UserCycleTask.objects.filter(session=s, is_completed=True).count()

        if total_tasks == 0:
            status = "Not Started"
        elif completed_tasks == total_tasks:
            status = "Completed"
        else:
            status = "In Progress"

        task_data.append({
            "session": s,
            "total": total_tasks,
            "completed": completed_tasks,
            "status": status,
        })

    return render(request, "user/tasks.html", {"task_data": task_data})
@login_required
def admin_crops(request):
    if not request.user.is_staff:
        return redirect("/login/")

    crops = CropDetails.objects.all().order_by("crop_name")
    return render(request, "admin_panel/admin_crops.html", {"crops": crops})


@login_required
def admin_add_crop(request):
    if not request.user.is_staff:
        return redirect("/login/")

    msg = None
    error = None

    if request.method == "POST":
        crop_name = request.POST.get("crop_name").lower()
        season = request.POST.get("season")
        soil_type = request.POST.get("soil_type")
        duration_days = request.POST.get("duration_days")
        watering = request.POST.get("watering")
        fertilizer = request.POST.get("fertilizer")
        common_diseases = request.POST.get("common_diseases")
        image = request.FILES.get("image")


        if CropDetails.objects.filter(crop_name=crop_name).exists():
            error = "Crop already exists!"
        else:
            CropDetails.objects.create(
                crop_name=crop_name,
                season=season,
                soil_type=soil_type,
                duration_days=int(duration_days),
                watering=watering,
                fertilizer=fertilizer,
                common_diseases=common_diseases,
                image=image
            )
            msg = "✅ Crop Details Added Successfully!"

    return render(request, "admin_panel/admin_add_crop.html", {"msg": msg, "error": error})

@login_required
def admin_delete_crop(request, crop_id):
    if not request.user.is_staff:
        return redirect("/login/")

    crop = CropDetails.objects.get(id=crop_id)
    crop.delete()
    return redirect("/admin-panel/crops/")
@login_required
def admin_edit_crop(request, crop_id):
    if not request.user.is_staff:
        return redirect("/login/")

    crop = CropDetails.objects.get(id=crop_id)
    msg = None

    if request.method == "POST":
        crop.crop_name = request.POST.get("crop_name").lower()
        crop.season = request.POST.get("season")
        crop.soil_type = request.POST.get("soil_type")
        crop.duration_days = int(request.POST.get("duration_days"))
        crop.watering = request.POST.get("watering")
        crop.fertilizer = request.POST.get("fertilizer")
        crop.common_diseases = request.POST.get("common_diseases")

        # ✅ if admin uploads new image, update it
        new_image = request.FILES.get("image")
        if new_image:
            crop.image = new_image

        crop.save()
        msg = "✅ Crop Details Updated Successfully!"

    return render(request, "admin_panel/admin_edit_crop.html", {"crop": crop, "msg": msg})


@login_required
def crop_search(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    crops = CropDetails.objects.all().order_by("crop_name")
    return render(request, "user/crop_search.html", {"crops": crops})


@login_required
def crop_details(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    crop_name = request.GET.get("crop")
    crop = CropDetails.objects.get(crop_name=crop_name)

    return render(request, "user/crop_details.html", {
        "crop": crop,
        "chatbot_crop": crop.crop_name
    })
@login_required
def start_manual_cycle(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    crop_name = request.GET.get("crop")

    if not crop_name:
        return redirect("/crop-search/")

    session = UserCropSession.objects.create(
        user=request.user,
        crop_name=crop_name,
        N=0, P=0, K=0,
        temperature=0,
        humidity=0,
        ph=0,
        rainfall=0
    )

    stages = CropCycle.objects.filter(
        crop_name__iexact=crop_name
    ).order_by("start_day")

    for stage in stages:
        UserCycleTask.objects.create(
            session=session,
            crop_cycle=stage
        )
    
    return redirect(f"/start-cycle/?session_id={session.id}")
from django.http import JsonResponse
from .models import CropDetails, CropCycle

from django.http import JsonResponse

@login_required
def chatbot_api(request):
    user_message = request.POST.get("message", "").lower()
    crop = request.POST.get("crop")
    stage = request.POST.get("stage")
    stage_id = request.POST.get("stage_id")


    # ============================
    # 🤖 DATABASE KNOWLEDGE LAYER
    # ============================

    knowledge_qs = ChatbotKnowledge.objects.filter(is_active=True)
    import re

    for kb in knowledge_qs:
        keywords = [k.strip().lower() for k in kb.keywords.split(",")]

        if any(re.search(rf'\b{re.escape(k)}\b', user_message) for k in keywords):

            if kb.crop_name and crop and kb.crop_name.lower() != crop.lower():
                continue

            if kb.stage_name and stage_id:
                try:
                    stage_obj = CropCycle.objects.get(id=stage_id)
                    if kb.stage_name.lower() != stage_obj.stage_name.lower():
                        continue
                except CropCycle.DoesNotExist:
                    continue

            return JsonResponse({"reply": kb.response})

    # ============================
    # Default help message
    # ============================

    response = (
        "👋 Hello! I can help you with:\n"
        "- Crop prediction\n"
        "- Manual crop search\n"
        "- Crop cycle tasks\n"
        "- Reports & progress\n\n"
        "Select a crop to get detailed guidance 🌱"
    )

    # ============================
    # 🔵 STAGE-AWARE LOGIC
    # ============================

    if crop and stage:
        stage_obj = CropCycle.objects.filter(
            crop_name__iexact=crop,
            stage_name__icontains=stage.strip()
        ).first()

        if stage_obj:

            if "now" in user_message or "current" in user_message:
                response = stage_obj.advisory

            elif "next" in user_message:
                next_stage_obj = CropCycle.objects.filter(
                    crop_name__iexact=crop,
                    start_day__gt=stage_obj.end_day
                ).order_by("start_day").first()

                if next_stage_obj:
                    response = (
                        f"The next stage is **{next_stage_obj.stage_name}** (Day {next_stage_obj.start_day}–{next_stage_obj.end_day}).\n\n"
                        f"**Advisory:** {next_stage_obj.advisory}"
                    )
                else:
                    response = "You have reached the final stage of the crop cycle! 🎉"

        else:
            response = "Stage information is not available."

    # ============================
    # 🟢 CROP-AWARE LOGIC
    # ============================

    elif crop:
        try:
            crop_obj = CropDetails.objects.get(crop_name__iexact=crop)

            if re.search(r'\b(water|irrigation|watering)\b', user_message):
                response = crop_obj.watering

            elif re.search(r'\b(fertilizer|nitrogen|npk|manure|urea)\b', user_message):
                response = crop_obj.fertilizer

            elif "disease" in user_message:
                response = crop_obj.common_diseases or "No major diseases recorded."

            elif "season" in user_message:
                response = f"{crop.title()} is usually grown in {crop_obj.season} season."

            elif "duration" in user_message or "days" in user_message:
                response = f"The crop duration is approximately {crop_obj.duration_days} days."

        except CropDetails.DoesNotExist:
            response = (
                "Crop cycle information is available, "
                "but detailed crop advisory is not yet added by admin."
            )

    # ============================
    # 🟡 DASHBOARD MODE
    # ============================

    else:
        if "predict" in user_message:
            response = "🌾 Crop Prediction uses a Machine Learning model."

        elif "cycle" in user_message:
            response = "🔄 Crop cycle represents stages of cultivation."

        elif "task" in user_message:
            response = "✅ Crop tasks help track cultivation progress."

    # ============================
    # 🧠 HYBRID RAG + GEMINI LAYER
    # ============================

    context_text = build_crop_context(crop, stage_id, stage)

    ai_reply = ask_gemini(user_message, context_text)

    if ai_reply:
        return JsonResponse({"reply": ai_reply})

    # 🛡️ SMART FALLBACK
    # If the local rules generated a specific answer (not the greeting menu), return it.
    if response and ("👋 Hello" not in response):
        return JsonResponse({"reply": response})

    # If all rules + AI failed, and it's not a generic greeting, show a helpful limit message
    is_generic = any(word in user_message for word in ["hi", "hello", "hey", "help", "who are you"])
    
    if is_generic or not user_message.strip():
        return JsonResponse({"reply": response}) # Show menu for greetings
    
    return JsonResponse({
        "reply": "I'm sorry, I'm having trouble providing a specific answer right now (AI Quota Reached or No Match Found). 🌱\n\n"
                 "Try asking about:\n"
                 "- **Watering** or **Irrigation**\n"
                 "- **Fertilizer** or **Nitrogen**\n"
                 "- **Next stages** of your crop cycle"
    })

# ---------- RAG Context Builder ----------

def build_crop_context(crop_name=None, stage_id=None, stage_name=None):

    context_parts = []

    # 🌾 Crop details
    if crop_name:
        crop = CropDetails.objects.filter(
            crop_name__iexact=crop_name
        ).first()

        if crop:
            context_parts.append(f"""
            Crop: {crop.crop_name}
            Season: {crop.season}
            Soil: {crop.soil_type}
            Duration: {crop.duration_days} days
            Watering: {crop.watering}
            Fertilizer: {crop.fertilizer}
            Diseases: {crop.common_diseases}
            """)

    # 🔄 Stage details
    if stage_id:
        stage = CropCycle.objects.filter(id=stage_id).first()
    elif crop_name and stage_name:
        stage = CropCycle.objects.filter(
            crop_name__iexact=crop_name,
            stage_name__icontains=stage_name.strip()
        ).first()
    else:
        stage = None

    if stage:
            context_parts.append(f"""
            Current Stage: {stage.stage_name}
            Stage Days: {stage.start_day}–{stage.end_day}
            Stage Advisory: {stage.advisory}
            """)

    return "\n".join(context_parts)
# ---------- Gemini Safe Ask ----------

def ask_gemini(user_msg, context_text):

    prompt = f"""
    You are an expert agriculture advisor.
    Answer only about crops and farming practices.
    Be practical and concise.

    Context:
    {context_text}

    User question:
    {user_msg}
    """

    try:
        resp = gemini_model.generate_content(prompt)
        if resp and resp.text:
            return resp.text.strip()
        return None

    except Exception as e:
        print("Gemini error:", e)
        return None

@login_required
def feedback_submit(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")
    
    if request.method == "POST":
        subject = request.POST.get("subject")
        message = request.POST.get("message")
        
        Feedback.objects.create(
            user=request.user,
            subject=subject,
            message=message
        )
        from django.contrib import messages
        messages.success(request, "Thank you for your feedback! We'll look into it.")
        return redirect("/dashboard/")
    
    return render(request, "user/feedback_form.html")

@login_required
def admin_feedback_view(request):
    if not request.user.is_staff:
        return redirect("/dashboard/")
        
    feedbacks = Feedback.objects.all().order_by("-created_at")
    return render(request, "admin_panel/view_feedback.html", {"feedbacks": feedbacks})

@login_required
def delete_session(request, session_id):
    from django.contrib import messages
    try:
        session = UserCropSession.objects.get(id=session_id)
        # Security check: Ensure user owns the session OR is staff
        if session.user == request.user or request.user.is_staff:
            session.delete()
            messages.success(request, "✅ Growth session and all related tasks deleted successfully.")
        else:
            messages.error(request, "❌ Access Denied: You cannot delete this session.")
    except UserCropSession.DoesNotExist:
        messages.error(request, "❓ Session not found.")
    except Exception as e:
        messages.error(request, f"❌ Error: {str(e)}")
        
    return redirect("/my-sessions/")

