import google.generativeai as genai

api_key = "AIzaSyDqdEywhexl41_ZfYbzx-LfYH3Qq4y2vmE"
genai.configure(api_key=api_key)

try:
    print("Available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("Error listing models:", e)
