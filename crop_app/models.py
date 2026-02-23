from django.db import models
from django.contrib.auth.models import User
class CropCycle(models.Model):
    crop_name = models.CharField(max_length=50)
    stage_name = models.CharField(max_length=100)
    start_day = models.IntegerField()
    end_day = models.IntegerField()
    advisory = models.TextField()

    def __str__(self):
        return f"{self.crop_name} - {self.stage_name} (Day {self.start_day}-{self.end_day})"
class UserCropSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    crop_name = models.CharField(max_length=50)
    N = models.FloatField()
    P = models.FloatField()
    K = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.crop_name} ({self.created_at})"

from django.utils import timezone

class UserCycleTask(models.Model):
    session = models.ForeignKey(UserCropSession, on_delete=models.CASCADE)
    crop_cycle = models.ForeignKey(CropCycle, on_delete=models.CASCADE)
    is_completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
class CropDetails(models.Model):
    crop_name = models.CharField(max_length=50, unique=True)

    season = models.CharField(max_length=100)
    soil_type = models.CharField(max_length=100)
    duration_days = models.IntegerField()

    watering = models.TextField()
    fertilizer = models.TextField()
    common_diseases = models.TextField(blank=True, null=True)

    image = models.ImageField(upload_to="crop_images/", blank=True, null=True)


    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.crop_name
class ChatbotKnowledge(models.Model):

    INTENT_CHOICES = [
        ("greeting", "Greeting"),
        ("dashboard", "Dashboard Help"),
        ("crop", "Crop Related"),
        ("stage", "Stage Related"),
        ("general", "General Agriculture"),
    ]

    intent = models.CharField(
        max_length=50,
        choices=INTENT_CHOICES
    )

    # Optional – for crop-specific replies
    crop_name = models.CharField(
        max_length=50,
        blank=True,
        null=True
    )

    # Optional – for stage-specific replies
    stage_name = models.CharField(
        max_length=50,
        blank=True,
        null=True
    )

    # Keywords to match user input
    keywords = models.TextField(
        help_text="Comma-separated keywords (e.g. water, irrigation)"
    )

    # Chatbot reply
    response = models.TextField()

    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.intent} | {self.crop_name or 'Any'} | {self.stage_name or 'Any'}"

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.subject}"
