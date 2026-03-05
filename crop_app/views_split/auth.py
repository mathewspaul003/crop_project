from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User

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


def admin_login(request):
    # Redirect to the unified secure portal
    return redirect("/login/")


def admin_logout(request):
    logout(request)
    return redirect("/")
