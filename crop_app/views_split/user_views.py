from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.http import HttpResponse
from io import BytesIO
from django.db.models import Count, Q
import numpy as np
import joblib
import os
from django.conf import settings
from ..models import CropDetails, CropCycle, UserCropSession, UserCycleTask, Feedback

# ✅ Load ML Model (Relative to codebase)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

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

        try:
            # --- Validation Bounds ---
            val_errors = []
            
            # Simple helper to check ranges
            def check_range(val, min_v, max_v, name):
                if float(val) < min_v or float(val) > max_v:
                    val_errors.append(f"{name} must be between {min_v} and {max_v}")

            check_range(values["N"], 0, 140, "Nitrogen")
            check_range(values["P"], 5, 145, "Phosphorus")
            check_range(values["K"], 5, 205, "Potassium")
            check_range(values["temperature"], 8, 48, "Temperature")
            check_range(values["humidity"], 10, 100, "Humidity")
            check_range(values["ph"], 3.5, 9.5, "Soil pH")
            check_range(values["rainfall"], 20, 350, "Rainfall")

            if val_errors:
                return render(request, "crop_app/home.html", {
                    "error": "❌ Unrealistic values detected: " + ", ".join(val_errors[:2]) + (f" (+{len(val_errors)-2} more)" if len(val_errors)>2 else ""),
                    "values": values,
                    "chatbot_crop": None
                })

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
        except Exception as e:
            return render(request, "crop_app/home.html", {"error": f"Invalid input: {str(e)}", "values": values})

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
def user_dashboard(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    overdue_tasks_count = UserCycleTask.objects.filter(
        session__user=request.user,
        is_completed=False
    ).count()

    context = {
        "overdue_tasks_count": overdue_tasks_count,
    }
    return render(request, "user/dashboard.html", context)


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
    days_passed = 0
    current_stage_id = None
    next_task_id = None

    if session_id:
        session = get_object_or_404(UserCropSession, id=session_id, user=request.user)
        crop = session.crop_name
        stages = CropCycle.objects.filter(crop_name__iexact=crop).order_by("start_day")

        # Optimization: Fetch existing tasks efficiently
        existing_tasks = UserCycleTask.objects.filter(session=session).select_related('crop_cycle')
        existing_stage_ids = set(existing_tasks.values_list('crop_cycle_id', flat=True))
        
        missing_stages = [s for s in stages if s.id not in existing_stage_ids]
        if missing_stages:
            UserCycleTask.objects.bulk_create([
                UserCycleTask(session=session, crop_cycle=s) for s in missing_stages
            ])
            # Re-fetch after bulk create
            tasks = list(UserCycleTask.objects.filter(session=session).select_related('crop_cycle').order_by("crop_cycle__start_day"))
        else:
            tasks = list(existing_tasks.order_by("crop_cycle__start_day"))

        days_passed = (timezone.now() - session.created_at).days + 1

        current_stage_obj = None
        for task in tasks:
            if not task.is_completed:
                current_stage_obj = task.crop_cycle
                next_task_id = task.id
                break

        if not current_stage_obj and tasks:
            current_stage_obj = tasks[-1].crop_cycle
        
        if current_stage_obj:
            current_stage = current_stage_obj.stage_name
            current_stage_id = current_stage_obj.id

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
            "days_passed": days_passed,
            "next_task_id": next_task_id
        }
    )


@require_POST
@login_required
def mark_done(request, task_id):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    task = get_object_or_404(UserCycleTask, id=task_id, session__user=request.user)

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
        session = get_object_or_404(UserCropSession, id=session_id, user=request.user)
        tasks = UserCycleTask.objects.filter(session=session).order_by("crop_cycle__start_day")

        if session.N == 0 and session.P == 0 and session.K == 0:
            is_manual = True

        if analyze:
            from .chatbot import gemini_model
            progress_str = "\n".join([f"- {t.crop_cycle.stage_name}: {'Completed' if t.is_completed else 'Pending'}" for t in tasks])

            if is_manual:
                analysis_prompt = f"Analyze this Agriculture Report for {session.crop_name}. This is a MANUAL session. Progress: {progress_str}. Suggest ideal NPK/pH."
            else:
                analysis_prompt = f"Analyze this Agriculture Report for {session.crop_name}. Info: N={session.N},P={session.P},K={session.K}, Stage: {progress_str}."

            try:
                response = gemini_model.generate_content(analysis_prompt)
                ai_analysis = response.text.strip()
            except:
                ai_analysis = "Sorry, I couldn't generate the analysis right now."

    return render(request, "crop_app/report.html", {"session": session, "tasks": tasks, "ai_analysis": ai_analysis, "is_manual": is_manual})


@login_required
def my_sessions(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    sessions = UserCropSession.objects.filter(user=request.user).annotate(
        total_tasks=Count('usercycletask'),
        completed_tasks=Count('usercycletask', filter=Q(usercycletask__is_completed=True))
    ).order_by("-created_at")
    
    session_data = []
    for s in sessions:
        session_data.append({
            "session": s, 
            "is_completed": (s.total_tasks > 0 and s.total_tasks == s.completed_tasks)
        })

    return render(request, "user/my_sessions.html", {"session_data": session_data})


@login_required
def all_reports(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")
    sessions = UserCropSession.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "user/reports.html", {"sessions": sessions})


@login_required
def all_tasks(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")

    sessions = UserCropSession.objects.filter(user=request.user).annotate(
        total_tasks=Count('usercycletask'),
        completed_tasks=Count('usercycletask', filter=Q(usercycletask__is_completed=True))
    ).order_by("-created_at")

    task_data = []
    for s in sessions:
        total = s.total_tasks
        done = s.completed_tasks
        status = "Completed" if (total > 0 and total == done) else "In Progress" if total > 0 else "Not Started"
        task_data.append({"session": s, "total": total, "completed": done, "status": status})

    return render(request, "user/tasks.html", {"task_data": task_data})


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
    crop = get_object_or_404(CropDetails, crop_name=crop_name)
    return render(request, "user/crop_details.html", {"crop": crop, "chatbot_crop": crop.crop_name})


@login_required
def start_manual_cycle(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")
    crop_name = request.GET.get("crop")
    if not crop_name:
        return redirect("/crop-search/")
    session = UserCropSession.objects.create(user=request.user, crop_name=crop_name, N=0, P=0, K=0, temperature=0, humidity=0, ph=0, rainfall=0)
    stages = CropCycle.objects.filter(crop_name__iexact=crop_name).order_by("start_day")
    for stage in stages:
        UserCycleTask.objects.create(session=session, crop_cycle=stage)
    return redirect(f"/start-cycle/?session_id={session.id}")


@login_required
def download_report_pdf(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")
    session_id = request.GET.get("session_id")
    session = get_object_or_404(UserCropSession, id=session_id, user=request.user)
    tasks = UserCycleTask.objects.filter(session=session).order_by("crop_cycle__start_day")
    is_manual = (session.N == 0 and session.P == 0 and session.K == 0)

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=20, textColor=colors.HexColor('#1a6b3c'),
                                  spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     fontSize=11, textColor=colors.HexColor('#555555'),
                                     spaceAfter=16, alignment=TA_CENTER)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'],
                                    fontSize=13, textColor=colors.HexColor('#2d5a27'),
                                    spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, leading=14, textColor=colors.HexColor('#333333'))

    story = []

    # Title
    story.append(Paragraph("🌾 Crop Cycle Report", title_style))
    story.append(Paragraph(f"Generated for: {request.user.first_name or request.user.username}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a6b3c')))
    story.append(Spacer(1, 12))

    # Crop Info
    story.append(Paragraph("Crop Information", section_style))
    crop_data = [
        ["Crop Name", session.crop_name.title()],
        ["Session Type", "Manual Entry" if is_manual else "AI Predicted"],
        ["Started On", session.created_at.strftime("%d %b %Y")],
    ]
    if not is_manual:
        crop_data += [
            ["Nitrogen (N)", f"{session.N}"],
            ["Phosphorus (P)", f"{session.P}"],
            ["Potassium (K)", f"{session.K}"],
            ["Temperature", f"{session.temperature} °C"],
            ["Humidity", f"{session.humidity} %"],
            ["Soil pH", f"{session.ph}"],
            ["Rainfall", f"{session.rainfall} mm"],
        ]

    info_table = Table(crop_data, colWidths=[5*cm, 11*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e9')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a6b3c')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f1f8f1')]),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#dddddd')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 16))

    # Task Table
    story.append(Paragraph("Growth Stage Progress", section_style))
    task_header = [["Stage", "Days", "Status"]]
    task_rows = []
    for t in tasks:
        status = "✔ Completed" if t.is_completed else "○ Pending"
        task_rows.append([
            t.crop_cycle.stage_name,
            f"Day {t.crop_cycle.start_day} – {t.crop_cycle.end_day}",
            status
        ])

    if task_rows:
        task_data = task_header + task_rows
        task_table = Table(task_data, colWidths=[7*cm, 4*cm, 5*cm])
        task_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a6b3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f8f1')]),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#dddddd')),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ]))
        story.append(task_table)
    else:
        story.append(Paragraph("No growth stages found for this session.", body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Generated by Agri Intel Hub · Smart Agriculture Platform", subtitle_style))

    doc.build(story)
    buffer.seek(0)
    response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{session.crop_name}_report.pdf"'
    response["Content-Transfer-Encoding"] = "binary"
    response["Accept-Ranges"] = "bytes"
    return response


@login_required
def feedback_submit(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")
    if request.method == "POST":
        Feedback.objects.create(user=request.user, subject=request.POST.get("subject"), message=request.POST.get("message"))
        from django.contrib import messages
        messages.success(request, "Thank you!")
        return redirect("/dashboard/")
    return render(request, "user/feedback_form.html")


@login_required
def delete_session(request, session_id):
    session = get_object_or_404(UserCropSession, id=session_id, user=request.user)
    session.delete()
    from django.contrib import messages
    messages.success(request, "Deleted successfully.")
    return redirect("/my-sessions/")


@login_required
def profile_view(request):
    if request.user.is_staff:
        return redirect("/admin-panel/dashboard/")
    
    if request.method == "POST":
        email = request.POST.get("email")
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        
        user = request.user
        user.email = email
        user.first_name = first_name
        user.last_name = last_name
        user.save()
        
        from django.contrib import messages
        messages.success(request, "✅ Profile updated successfully!")
        return redirect("/profile/")
        
    return render(request, "user/profile.html")
