import os
import django
import sys
from io import BytesIO

# Setup Django environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_project.settings')
django.setup()

from django.template.loader import render_to_string
from xhtml2pdf import pisa
from crop_app.models import UserCropSession, UserCycleTask

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
        is_manual = (session.N == 0 and session.P == 0 and session.K == 0)

        context = {
            "session": session,
            "tasks": tasks,
            "is_manual": is_manual,
        }
        
        try:
            html = render_to_string("crop_app/report_pdf.html", context)
            buffer = BytesIO()
            pisa_status = pisa.CreatePDF(html, dest=buffer)

            if pisa_status.err:
                print(f"FAILED (pisa error: {pisa_status.err})")
                failed_sessions.append((session.id, "pisa error"))
            else:
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
