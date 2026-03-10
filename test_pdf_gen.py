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

def test_pdf_generation():
    session = UserCropSession.objects.first()
    if not session:
        print("No sessions found in database.")
        return

    print(f"Testing PDF generation for Session ID: {session.id}, Crop: {session.crop_name}")
    tasks = UserCycleTask.objects.filter(session=session).order_by("crop_cycle__start_day")
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph(f"Crop Report: {session.crop_name}", styles['Title'])]
        story.append(Paragraph(f"Session Start: {session.created_at.strftime('%Y-%m-%d')}", styles['Normal']))
        
        # Build task table
        data = [["Stage", "Days", "Status"]]
        for t in tasks:
            data.append([
                t.crop_cycle.stage_name, 
                f"{t.crop_cycle.start_day}-{t.crop_cycle.end_day}", 
                "Yes" if t.is_completed else "No"
            ])
            
        if len(data) > 1:
            story.append(Table(data, style=TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.green),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
            ])))
            
        doc.build(story)
        with open("test_report.pdf", "wb") as f:
            f.write(buffer.getvalue())
        print(f"PDF successfully saved to {os.path.abspath('test_report.pdf')}")
    except Exception as e:
        print(f"Exception during PDF generation: {e}")

if __name__ == "__main__":
    test_pdf_generation()
