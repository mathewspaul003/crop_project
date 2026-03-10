import os
import django
import sys
from io import BytesIO

# Setup Django environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_project.settings')
django.setup()

from crop_app.models import UserCropSession, UserCycleTask
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def test_all_pdf_generation():
    sessions = UserCropSession.objects.all()
    if not sessions.exists():
        print("No sessions found in database.")
        return

    print(f"Testing PDF generation for {sessions.count()} sessions...")
    
    failed_sessions = []
    
    for session in sessions:
        print(f"Testing Session ID: {session.id}, Crop: {session.crop_name}", end="... ")
        tasks = UserCycleTask.objects.filter(session=session).order_by("crop_cycle__start_day")
        
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = [Paragraph(f"Test Report: {session.crop_name}", styles['Title'])]
            
            # Simple table for test
            data = [["Stage", "Status"]]
            for t in tasks:
                data.append([t.crop_cycle.stage_name, "Done" if t.is_completed else "Pending"])
            
            if len(data) > 1:
                story.append(Table(data, style=TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)])))
            
            doc.build(story)
            print("SUCCESS")
        except Exception as e:
            print(f"FAILED (Exception: {str(e)})")
            failed_sessions.append((session.id, str(e)))

    if failed_sessions:
        print("\nSummary of Failures:")
        for sid, err in failed_sessions:
            print(f"Session {sid}: {err}")
    else:
        print("\nAll sessions generated PDF successfully!")

if __name__ == "__main__":
    test_all_pdf_generation()
