# Agriculture Intelligence Hub 🌾🚜✨

Empowering modern farmers and administrators with high-precision AI predictions, real-time crop tracking, and intelligent agricultural insights.

## 🚀 Key Features

- **AI-Powered Crop Prediction**: High-precision analysis of soil and environmental data using Machine Learning (Random Forest).
- **📍 Weather API Integration**: Auto-fill temperature, humidity, and rainfall using real-time local weather data (powered by Open-Meteo).
- **📊 Admin Analytics Center**: Interactive **Chart.js** dashboards tracking weekly activity, popular crop trends, and user growth.
- **🛡️ Smart Validation**: Enhanced input validation to ensure realistic farming conditions (pH, nutrients, etc.), preventing inaccurate predictions.
- **🔄 Dynamic Growth Tracker**: Stage-wise monitoring with intelligent "Typing" assistant support and automated progress banners.
- **⏳ Proactive Notifications**: Farmers receive immediate dashboard alerts for overdue tasks or pending stages in their active cycles.
- **🏗️ Modular Architecture**: Decomposed monolithic views into a clean, scalable structure (`views_split/`) for professional maintainability.
- **Intelligent Reports**: Comprehensive PDF generation for farming history, nutrient analysis, and AI-driven cultivation advice.

## 🛠️ Technology Stack

- **Backend**: Django 5.1 (Python)
- **AI/ML**: Scikit-learn, joblib, Google Gemini AI (Flash-latest)
- **Frontend**: Glassmorphic UI with Vanilla CSS, Bootstrap 5, and Chart.js
- **Weather Services**: Open-Meteo (Geolocation-based)
- **Reporting**: FPDF2

## 📂 Project Structure Improvements

The project has been reorganized for clarity and professional standards:
- `crop_app/views_split/`: Modularized view logic (Auth, Admin, User, Chatbot, Main).
- `data/`: Centralized raw datasets and training resources.
- `scripts/`: Local utilities for AI model training and evaluation.
- `admin_panel/`: Custom-built administrative dashboard separate from standard Django admin.

## 📦 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mathewspaul003/crop_project.git
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
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   # Add database credentials if using external PostgreSQL
   ```

5. **Start the server**:
   ```bash
   python manage.py runserver
   ```

---
*Built with passion for modern agriculture and high-performance AI integration.* 🦾🏗️🏙️🛰️🌾
