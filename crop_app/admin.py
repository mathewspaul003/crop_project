from django.contrib import admin
from .models import CropCycle, CropDetails, UserCropSession, UserCycleTask, ChatbotKnowledge, Feedback


@admin.register(CropCycle)
class CropCycleAdmin(admin.ModelAdmin):
    list_display = ("crop_name", "stage_name", "start_day", "end_day")
    list_filter = ("crop_name",)
    search_fields = ("crop_name", "stage_name", "advisory")
    ordering = ("crop_name", "start_day")


@admin.register(CropDetails)
class CropDetailsAdmin(admin.ModelAdmin):
    list_display = ("crop_name", "season", "soil_type", "duration_days", "created_at")
    list_filter = ("season", "soil_type")
    search_fields = ("crop_name", "season", "soil_type", "common_diseases")
    ordering = ("crop_name",)


@admin.register(UserCropSession)
class UserCropSessionAdmin(admin.ModelAdmin):
    list_display = ("user", "crop_name", "N", "P", "K", "temperature", "ph", "created_at")
    list_filter = ("crop_name",)
    search_fields = ("user__username", "crop_name")
    ordering = ("-created_at",)


@admin.register(UserCycleTask)
class UserCycleTaskAdmin(admin.ModelAdmin):
    list_display = ("session", "crop_cycle", "is_completed", "completed_at")
    list_filter = ("is_completed",)
    search_fields = ("session__user__username", "crop_cycle__stage_name")


@admin.register(ChatbotKnowledge)
class ChatbotKnowledgeAdmin(admin.ModelAdmin):
    list_display = ("intent", "crop_name", "stage_name", "is_active")
    list_filter = ("intent", "is_active")
    search_fields = ("keywords", "response", "crop_name", "stage_name")


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("user", "subject", "created_at")
    search_fields = ("user__username", "subject", "message")
    ordering = ("-created_at",)

