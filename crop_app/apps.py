from django.apps import AppConfig
import os


class CropAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'crop_app'

    def ready(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            import warnings
            warnings.warn(
                "⚠️  GEMINI_API_KEY is not set! "
                "Chatbot and AI report analysis will not work. "
                "Set this variable in your .env file or Vercel environment settings.",
                RuntimeWarning,
                stacklevel=2
            )
