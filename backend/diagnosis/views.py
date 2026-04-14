import json
import time
import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Определяем режим работы по переменной окружения
DEBUG = os.environ.get("DEBUG", "True") == "True"

if DEBUG:
    # Локальная разработка: используем БД
    from .models import Disease, Symptom
    
    def get_ml_symptoms():
        """Возвращает список симптомов из БД."""
        return list(Symptom.objects.all().order_by("id").values_list("name", flat=True))
    
    def get_disease_info_from_db(disease_name):
        """Возвращает информацию о заболевании из БД."""
        try:
            disease_obj = Disease.objects.get(name__iexact=disease_name)
            return {
                "description": disease_obj.description,
                "treatment": disease_obj.treatment,
                "symptoms": disease_obj.symptoms,
                "severity": disease_obj.severity,
                "specialist": disease_obj.specialist,
                "category": disease_obj.category,
            }
        except Disease.DoesNotExist:
            return {
                "description": f"Информация о заболевании '{disease_name}' готовится нашими специалистами. Обратитесь к врачу для точной диагностики и лечения.",
                "treatment": "Для назначения лечения обратитесь к квалифицированному медицинскому специалисту. Не занимайтесь самолечением.",
                "symptoms": ["Информация уточняется"],
                "severity": "unknown",
                "specialist": "Терапевт",
                "category": "Уточняется",
            }
else:
    # Продакшен (Render): используем JSON
    from .symptoms_data import SYMPTOMS_LIST
    from .disease_data import DISEASE_DATABASE
    
    def get_ml_symptoms():
        """Возвращает список симптомов из JSON-хранилища."""
        return SYMPTOMS_LIST
    
    def get_disease_info_from_db(disease_name):
        """Возвращает информацию о заболевании из JSON-хранилища."""
        disease_name_lower = disease_name.lower()
        for name, info in DISEASE_DATABASE.items():
            if name.lower() == disease_name_lower:
                return {
                    "description": info["description"],
                    "treatment": info["treatment"],
                    "symptoms": info["symptoms"],
                    "severity": info["severity"],
                    "specialist": info["specialist"],
                    "category": info["category"],
                }
        
        return {
            "description": f"Информация о заболевании '{disease_name}' готовится нашими специалистами. Обратитесь к врачу для точной диагностики и лечения.",
            "treatment": "Для назначения лечения обратитесь к квалифицированному медицинскому специалисту. Не занимайтесь самолечением.",
            "symptoms": ["Информация уточняется"],
            "severity": "unknown",
            "specialist": "Терапевт",
            "category": "Уточняется",
        }


# Глобальные переменные для отслеживания состояния системы
model = None
diseases_list = []
model_loaded_successfully = False
model_error_message = ""

# Инициализация ML-модели и данных при запуске приложения
try:
    model = joblib.load(settings.ML_MODEL_PATH)
    diseases_list = model.classes_.tolist()
    model_loaded_successfully = True
    model_error_message = ""

except Exception as e:
    model = None
    diseases_list = []
    model_loaded_successfully = False
    model_error_message = "система диагностики временно недоступна"
    print(f"Ошибка загрузки ML-модели: {e}")


def home(request):
    """
    Главная страница приложения.
    """
    current_symptoms = get_ml_symptoms()

    symptoms_by_letter = {}

    for symptom in current_symptoms:
        if symptom:
            first_letter = symptom[0].upper() if symptom else ""
            if not ("А" <= first_letter <= "Я"):
                first_letter = "#"

            if first_letter not in symptoms_by_letter:
                symptoms_by_letter[first_letter] = []
            symptoms_by_letter[first_letter].append(symptom)

    sorted_letters = sorted(symptoms_by_letter.keys())
    sorted_symptoms_by_letter = {}

    for letter in sorted_letters:
        sorted_symptoms_by_letter[letter] = sorted(symptoms_by_letter[letter])

    context = {
        "symptoms": current_symptoms,
        "symptoms_by_letter": sorted_symptoms_by_letter,
        "symptoms_count": len(current_symptoms),
        "diseases_count": len(diseases_list) if diseases_list else 0,
        "model_loaded": model_loaded_successfully,
        "model_error": model_error_message,
    }

    return render(request, "diagnosis/home.html", context)


def predict(request):
    """
    Обработчик предсказания заболевания на основе выбранных симптомов.
    """
    if request.method == "POST" and model is not None:
        try:
            current_symptoms = get_ml_symptoms()
            selected_symptoms = request.POST.getlist("symptoms")

            input_vector = np.zeros(len(current_symptoms))

            for symptom in selected_symptoms:
                if symptom in current_symptoms:
                    idx = current_symptoms.index(symptom)
                    input_vector[idx] = 1

            symptom_count = np.sum(input_vector)
            if symptom_count == 0:
                return render(
                    request,
                    "diagnosis/error.html",
                    {"error": "Пожалуйста, выберите хотя бы один симптом"},
                )

            probabilities = model.predict_proba([input_vector])[0]
            top5_indices = np.argsort(probabilities)[-5:][::-1]

            results = []
            for idx in top5_indices:
                disease_name = diseases_list[idx]
                probability = float(probabilities[idx])

                results.append(
                    {
                        "disease": disease_name,
                        "probability": probability,
                        "percentage": f"{probability * 100:.2f}%",
                        "description": get_disease_description(disease_name),
                        "treatment": get_disease_treatment(disease_name),
                    }
                )

            return render(
                request,
                "diagnosis/results.html",
                {
                    "results": results,
                    "symptoms_count": len(selected_symptoms),
                    "selected_symptoms": selected_symptoms,
                },
            )

        except Exception as e:
            return render(
                request,
                "diagnosis/error.html",
                {"error": f"Произошла ошибка при анализе симптомов: {str(e)}"},
            )

    return home(request)


def about(request):
    """Отображает страницу 'О проекте' с информацией о системе."""
    return render(request, "diagnosis/about.html")


def how_to_use(request):
    """Отображает страницу с инструкцией по использованию системы."""
    return render(request, "diagnosis/how_to_use.html")


def get_disease_description(disease_name):
    """
    Возвращает описание заболевания из базы знаний.
    """
    disease_info = get_disease_info_from_db(disease_name)
    return disease_info["description"]


def get_disease_treatment(disease_name):
    """
    Возвращает рекомендации по лечению заболевания из базы знаний.
    """
    disease_info = get_disease_info_from_db(disease_name)
    return disease_info["treatment"]


def get_disease_suggestions(searched_name):
    """
    Возвращает список похожих названий заболеваний для подсказок при поиске.
    """
    from difflib import get_close_matches

    all_diseases = model.classes_.tolist() if model else []

    if not all_diseases:
        return []

    suggestions = get_close_matches(searched_name, all_diseases, n=5, cutoff=0.3)
    return suggestions


def disease_detail(request, disease_name):
    """
    Отображает детальную страницу информации о заболевании.
    """
    disease_info = get_disease_info_from_db(disease_name)
    is_from_knowledge_base = "knowledge-base/disease" in request.path
    is_unknown_disease = disease_info["description"].startswith("Информация о заболевании")

    if is_unknown_disease:
        return render(
            request,
            "diagnosis/disease_not_found.html",
            {
                "searched_disease": disease_name,
                "suggestions": get_disease_suggestions(disease_name) if model else [],
            },
        )

    return render(
        request,
        "diagnosis/disease_detail.html",
        {
            "disease_name": disease_name,
            "description": disease_info["description"],
            "treatment": disease_info["treatment"],
            "symptoms": disease_info["symptoms"],
            "severity": disease_info["severity"],
            "specialist": disease_info["specialist"],
            "category": disease_info["category"],
            "emergency_contacts": [
                "112 - Единая служба спасения",
                "103 - Скорая помощь",
                "03 - Скорая помощь (старый номер)",
            ],
            "is_from_knowledge_base": is_from_knowledge_base,
        },
    )


def knowledge_base(request):
    """
    Отображает страницу базы знаний со всеми заболеваниями.
    """
    if DEBUG:
        # Локальная разработка: из БД
        all_diseases_objects = Disease.objects.order_by("name").all()
        diseases_with_info = []
        for disease_obj in all_diseases_objects:
            severity_text = get_severity_display(disease_obj.severity)
            diseases_with_info.append(
                {
                    "name": disease_obj.name,
                    "info": {
                        "description": disease_obj.description,
                        "treatment": disease_obj.treatment,
                        "symptoms": disease_obj.symptoms,
                        "severity": disease_obj.severity,
                        "specialist": disease_obj.specialist,
                        "category": disease_obj.category,
                    },
                    "severity_display": severity_text,
                }
            )
    else:
        # Продакшен: из JSON
        diseases_with_info = []
        for name, info in DISEASE_DATABASE.items():
            severity_text = get_severity_display(info["severity"])
            diseases_with_info.append(
                {
                    "name": name,
                    "info": {
                        "description": info["description"],
                        "treatment": info["treatment"],
                        "symptoms": info["symptoms"],
                        "severity": info["severity"],
                        "specialist": info["specialist"],
                        "category": info["category"],
                    },
                    "severity_display": severity_text,
                }
            )
        diseases_with_info.sort(key=lambda x: x["name"])

    diseases_by_letter = {}
    for disease in diseases_with_info:
        first_letter = disease["name"][0].upper() if disease["name"] else ""
        if first_letter not in diseases_by_letter:
            diseases_by_letter[first_letter] = []
        diseases_by_letter[first_letter].append(disease)

    return render(
        request,
        "diagnosis/knowledge_base.html",
        {"diseases_by_letter": diseases_by_letter, "total_diseases": len(diseases_with_info)},
    )


def get_severity_display(severity):
    """
    Преобразует код серьезности заболевания в читаемое представление.
    """
    severity_map = {
        "high": "ВЫСОКАЯ",
        "medium": "СРЕДНЯЯ",
        "low": "НИЗКАЯ",
        "unknown": "НЕОПРЕДЕЛЕНА",
        "variable": "ЗАВИСИТ ОТ СТАДИИ",
    }
    return severity_map.get(severity, "НЕОПРЕДЕЛЕНА")


# ============================================================================
# NLP МОДУЛЬ
# ============================================================================

hybrid_extractor = None
nlp_loaded_successfully = False

# Загружаем NLP всегда (на Render - rule-based, локально - все компоненты)
try:
    from ml.nlp.extractors.hybrid import HybridExtractor

    hybrid_extractor = HybridExtractor(
        semantic_threshold_common=0.85, semantic_threshold_rare=0.75, ner_confidence_threshold=0.75
    )

    # Прогрев только локально (на Render не нужно, чтобы не тратить время)
    if DEBUG:
        warmup_phrases = ["болит голова", "кашель и температура", "тошнота, слабость, головокружение"]
        for phrase in warmup_phrases:
            hybrid_extractor.extract(phrase)
        print("NLP модуль загружен (локальный режим, все компоненты)")
    else:
        print("NLP модуль загружен (продакшен режим, только rule-based)")

    nlp_loaded_successfully = True

except Exception as e:
    print(f"Ошибка загрузки NLP модуля: {e}")
    nlp_loaded_successfully = False
    hybrid_extractor = None


@require_http_methods(["POST"])
@csrf_exempt
def extract_from_text_api(request):
    """
    API для извлечения симптомов из текста.
    Возвращает найденные симптомы и рекомендуемые.
    """
    try:
        data = json.loads(request.body)
        user_text = data.get("text", "").strip()

        if not user_text:
            return JsonResponse({"error": "Введите текст с симптомами"}, status=400)

        if not nlp_loaded_successfully or hybrid_extractor is None:
            return JsonResponse({"error": "NLP-модуль временно недоступен"}, status=503)

        extracted = hybrid_extractor.extract(user_text)

        suggested_symptoms = []
        present_symptoms = [s for s in extracted if s["status"] == "present"]

        # Предполагаемые симптомы только в локальном режиме (есть семантика)
        if present_symptoms and DEBUG and hybrid_extractor.semantic:
            all_symptom_names = get_ml_symptoms()
            found_names = set(s["canonical_name"] for s in present_symptoms)

            for symptom in present_symptoms[:3]:
                try:
                    similar = hybrid_extractor.semantic.find_similar(
                        symptom["canonical_name"], exclude=found_names, top_k=2
                    )
                    suggested_symptoms.extend(similar)
                except Exception:
                    pass

        unique_suggested = {}
        for s in suggested_symptoms:
            name = s["canonical_name"]
            if name not in unique_suggested or s["confidence"] > unique_suggested[name]["confidence"]:
                unique_suggested[name] = s

        return JsonResponse(
            {
                "success": True,
                "extracted_symptoms": [
                    {"canonical_name": s["canonical_name"], "status": s["status"], "confidence": s["confidence"]}
                    for s in extracted
                ],
                "suggested_symptoms": [
                    {"name": s["canonical_name"], "similarity": s["confidence"]}
                    for s in list(unique_suggested.values())[:5]
                ],
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Неверный формат запроса"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)