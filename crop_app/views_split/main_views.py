from django.shortcuts import render, redirect

def index(request):
    # If already authenticated, skip landing and go to respective hub
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect("/admin-panel/dashboard/")
        return redirect("/dashboard/")

    # Guest user → Render Pre-Login Dashboard
    return render(request, "crop_app/landing.html")
