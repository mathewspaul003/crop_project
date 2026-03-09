from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.utils import timezone
import datetime, json
from ..models import CropDetails, CropCycle, UserCropSession, UserCycleTask, Feedback

@login_required
def admin_dashboard(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        return redirect("/login/")

    # --- Basic counts ---
    crop_count = CropDetails.objects.count()
    user_count = User.objects.filter(is_staff=False).count()
    feedback_count = Feedback.objects.count()
    session_count = UserCropSession.objects.count()

    # --- Top 5 predicted crops ---
    top_crops_qs = (
        UserCropSession.objects
        .values("crop_name")
        .annotate(total=Count("id"))
        .order_by("-total")[:5]
    )
    top_crop_labels = json.dumps([r["crop_name"].title() for r in top_crops_qs])
    top_crop_data   = json.dumps([r["total"] for r in top_crops_qs])

    # --- Sessions per day for last 7 days ---
    today = timezone.now().date()
    day_labels, day_data = [], []
    for i in range(6, -1, -1):
        day = today - datetime.timedelta(days=i)
        count = UserCropSession.objects.filter(created_at__date=day).count()
        day_labels.append(day.strftime("%b %d"))
        day_data.append(count)
    weekly_labels = json.dumps(day_labels)
    weekly_data   = json.dumps(day_data)

    context = {
        "crop_count":    crop_count,
        "user_count":    user_count,
        "feedback_count": feedback_count,
        "session_count": session_count,
        "top_crop_labels": top_crop_labels,
        "top_crop_data":   top_crop_data,
        "weekly_labels":   weekly_labels,
        "weekly_data":     weekly_data,
    }

    return render(request, "admin_panel/admin_dashboard.html", context)


@login_required
def admin_users_view(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        return redirect("/login/")

    users = User.objects.filter(is_staff=False).annotate(
        session_count=Count('usercropsession')
    ).order_by('-date_joined')

    # Annotate each user with their session count for the table
    user_data = []
    for u in users:
        user_data.append({"user": u, "session_count": u.session_count})

    return render(request, "admin_panel/view_users.html", {"user_data": user_data})


@login_required
def admin_user_crops(request, user_id):
    """Admin view to see all crop sessions for a specific user."""
    if not request.user.is_authenticated or not request.user.is_staff:
        return redirect("/login/")

    target_user = User.objects.get(id=user_id)
    sessions = UserCropSession.objects.filter(user=target_user).annotate(
        total_tasks=Count('usercycletask'),
        completed_tasks=Count('usercycletask', filter=Q(usercycletask__is_completed=True))
    ).order_by("-created_at")

    session_data = []
    for s in sessions:
        total = s.total_tasks
        done = s.completed_tasks
        is_manual = (s.N == 0 and s.P == 0 and s.K == 0)
        session_data.append({
            "session": s,
            "total_tasks": total,
            "completed_tasks": done,
            "progress_pct": int((done / total) * 100) if total > 0 else 0,
            "is_completed": total > 0 and total == done,
            "is_manual": is_manual,
        })

    return render(request, "admin_panel/user_crops.html", {
        "target_user": target_user,
        "session_data": session_data,
    })


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

    all_cycles = CropCycle.objects.all().order_by('crop_name', 'start_day')
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
    
    user_to_delete = User.objects.get(id=user_id)
    if not user_to_delete.is_staff:
        user_to_delete.delete()
        
    return redirect("/admin-panel/users/")


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
        new_image = request.FILES.get("image")
        if new_image:
            crop.image = new_image
        crop.save()
        msg = "✅ Crop Details Updated Successfully!"

    return render(request, "admin_panel/admin_edit_crop.html", {"crop": crop, "msg": msg})


@login_required
def admin_feedback_view(request):
    if not request.user.is_staff:
        return redirect("/dashboard/")
        
    feedbacks = Feedback.objects.all().order_by("-created_at")
    return render(request, "admin_panel/view_feedback.html", {"feedbacks": feedbacks})
