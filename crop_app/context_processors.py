from .models import Feedback

def admin_notifications(request):
    """Context processor to provide unread feedback count to admin templates."""
    if request.user.is_authenticated and request.user.is_superuser and request.path.startswith('/admin-panel/'):
        unread_feedback_count = Feedback.objects.filter(is_read=False).count()
        return {'unread_feedback_count': unread_feedback_count}
    return {}
