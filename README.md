# Agriculture Intelligence Hub 🌾🚜✨

Empowering modern farmers and administrators with high-precision AI predictions, real-time crop tracking, and intelligent agricultural insights.

## 🚀 Key Features

- **AI-Powered Crop Prediction**: High-precision analysis of soil and environmental data.
- **Dynamic Growth Tracker**: Stage-wise monitoring and intelligent advisories for cultivation cycles.
- **Unified Secure Portal**: Streamlined email-based authentication for farmers and admins.
- **Admin Command Center**: Complete management of crop databases, user directories, and growth cycles.
- **Intelligent Reports**: Comprehensive PDF generation for farming history and analysis.
- **Built-in AI Assistant**: Real-time agricultural support and stage-aware guidance.

## 🛠️ Technology Stack

- **Backend**: Django (Python)
- **AI/ML**: Scikit-learn, joblib, Google Gemini (Flash-latest)
- **Frontend**: Glassmorphic UI with Vanilla CSS & Bootstrap 5
- **Reporting**: xhtml2pdf

## 📦 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd crop_project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file or export your Gemini API key:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Run Migrations**:
   ```bash
   python manage.py migrate
   ```

6. **Start the server**:
   ```bash
   python manage.py runserver
   ```

---
*Built with passion for modern agriculture.* 🦾🏗️🏙️🛰️🌾
