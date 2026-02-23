from django.contrib import admin
from .models import CropCycle

admin.site.register(CropCycle)
from .models import ChatbotKnowledge

@admin.register(ChatbotKnowledge)
class ChatbotKnowledgeAdmin(admin.ModelAdmin):
    list_display = ("intent", "crop_name", "stage_name", "is_active")
    list_filter = ("intent", "is_active")
    search_fields = ("keywords", "response", "crop_name", "stage_name")
