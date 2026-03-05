import google.generativeai as genai
import os
import re
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from ..models import ChatbotKnowledge, CropCycle, CropDetails

# ✅ Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None


@login_required
def chatbot_api(request):
    user_message = request.POST.get("message", "").lower()
    crop = request.POST.get("crop")
    stage = request.POST.get("stage")
    stage_id = request.POST.get("stage_id")

    # --- 1. Database Knowledge (RAG Lite) ---
    knowledge_qs = ChatbotKnowledge.objects.filter(is_active=True)
    for kb in knowledge_qs:
        keywords = [k.strip().lower() for k in kb.keywords.split(",")]
        if any(re.search(rf'\b{re.escape(k)}\b', user_message) for k in keywords):
            if kb.crop_name and crop and kb.crop_name.lower() != crop.lower():
                continue
            if kb.stage_name and stage_id:
                try:
                    stage_obj = CropCycle.objects.get(id=stage_id)
                    if kb.stage_name.lower() != stage_obj.stage_name.lower():
                        continue
                except CropCycle.DoesNotExist:
                    continue
            return JsonResponse({"reply": kb.response})

    # --- 2. Hybrid Gemini AI Layer ---
    if gemini_model:
        # 🚀 FAST-PATH: Instant responses for basic greetings to avoid API delay
        greetings = ["hi", "hello", "hey", "hii", "helloo"]
        if user_message.strip() in greetings:
            return JsonResponse({"reply": "Hello! How can I help you with your crops today? 🌱"})

        context_text = build_crop_context(crop, stage_id, stage)
        ai_reply = ask_gemini(user_message, context_text)
        if ai_reply:
            return JsonResponse({"reply": ai_reply})

    # --- 3. Smart Fallbacks ---
    response = "👋 Hello! I can help you with crop prediction, tasks, and reports."
    return JsonResponse({"reply": response})


def build_crop_context(crop_name=None, stage_id=None, stage_name=None):
    context_parts = []
    if crop_name:
        crop = CropDetails.objects.filter(crop_name__iexact=crop_name).first()
        if crop:
            context_parts.append(f"Crop: {crop.crop_name}\nWatering: {crop.watering}\nFertilizer: {crop.fertilizer}")
    
    stage = None
    if stage_id:
        stage = CropCycle.objects.filter(id=stage_id).first()
    elif crop_name and stage_name:
        stage = CropCycle.objects.filter(crop_name__iexact=crop_name, stage_name__icontains=stage_name.strip()).first()
    
    if stage:
        context_parts.append(f"Stage: {stage.stage_name}\nAdvisory: {stage.advisory}")
        
    return "\n".join(context_parts)


def ask_gemini(user_msg, context_text):
    if not gemini_model:
        return None
    prompt = f"Expert Agriculture Advisor context:\n{context_text}\n\nQuestion: {user_msg}"
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text.strip() if resp and resp.text else None
    except:
        return None
