from django.http import JsonResponse
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def validate_password_ajax(request):
    """
    AJAX endpoint to validate a password in real-time.
    Expects a POST request with a JSON body containing {"password": "..."}.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            password = data.get("password", "")
        except json.JSONDecodeError:
            return JsonResponse({"valid": False, "errors": ["Invalid JSON request."]}, status=400)

        if not password:
            return JsonResponse({"valid": True, "errors": []})

        try:
            validate_password(password)
            return JsonResponse({"valid": True, "errors": []})
        except ValidationError as e:
            return JsonResponse({"valid": False, "errors": list(e.messages)})

    return JsonResponse({"error": "Invalid request method."}, status=405)
