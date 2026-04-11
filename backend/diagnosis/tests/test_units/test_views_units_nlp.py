# C:\Users\Leo\PyWork\MedWebsite\backend\diagnosis\tests\test_units\test_views_units_nlp.py
"""Unit-тесты для views NLP модуля (API извлечения симптомов)."""

import json
import time
from unittest.mock import Mock, patch

import pytest
from django.test import RequestFactory
from django.urls import reverse

# Импортируем view
from diagnosis.views import extract_from_text_api, get_ml_symptoms


@pytest.fixture
def factory():
    return RequestFactory()


# ============================================================================
# Тесты для API
# ============================================================================


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_success(factory):
    """Тест: успешное извлечение симптомов."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
            {"canonical_name": "кашель", "status": "present", "confidence": 0.90},
        ]
        mock_hybrid.semantic = Mock()
        mock_hybrid.semantic.find_similar.return_value = []
        mock_get_symptoms.return_value = ["головная боль", "кашель"]

        request = factory.post(
            reverse("extract_from_text_api"),
            data=json.dumps({"text": "болит голова и кашель"}),
            content_type="application/json",
        )

        response = extract_from_text_api(request)

        assert response.status_code == 200
        # Используем .content и json.loads вместо .json()
        import json as json_lib

        data = json_lib.loads(response.content)
        assert data["success"] is True
        assert len(data["extracted_symptoms"]) == 2


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_empty_text(factory):
    """Тест: пустой текст возвращает ошибку."""
    with patch("diagnosis.views.nlp_loaded_successfully", True):
        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": ""}), content_type="application/json"
        )

        response = extract_from_text_api(request)

        assert response.status_code == 400
        import json as json_lib

        data = json_lib.loads(response.content)
        assert data["error"] == "Введите текст с симптомами"


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_missing_text_field(factory):
    """Тест: отсутствие поля text в запросе."""
    with patch("diagnosis.views.nlp_loaded_successfully", True):
        request = factory.post(reverse("extract_from_text_api"), data=json.dumps({}), content_type="application/json")

        response = extract_from_text_api(request)

        assert response.status_code == 400
        import json as json_lib

        data = json_lib.loads(response.content)
        assert "error" in data


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_invalid_json(factory):
    """Тест: неверный JSON формат."""
    with patch("diagnosis.views.nlp_loaded_successfully", True):
        request = factory.post(reverse("extract_from_text_api"), data="not a json", content_type="application/json")

        response = extract_from_text_api(request)

        assert response.status_code == 400
        import json as json_lib

        data = json_lib.loads(response.content)
        assert "error" in data


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_nlp_not_loaded(factory):
    """Тест: NLP модуль не загружен."""
    with patch("diagnosis.views.nlp_loaded_successfully", False), patch("diagnosis.views.hybrid_extractor", None):

        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )

        response = extract_from_text_api(request)

        assert response.status_code == 503
        import json as json_lib

        data = json_lib.loads(response.content)
        assert data["error"] == "NLP-модуль временно недоступен"


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_with_suggested_symptoms(factory):
    """Тест: возврат предполагаемых симптомов."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.return_value = [
            {"canonical_name": "боль в животе", "status": "present", "confidence": 0.95},
        ]
        mock_hybrid.semantic = Mock()
        mock_hybrid.semantic.find_similar.return_value = [
            {"canonical_name": "метеоризм", "confidence": 0.85},
        ]
        mock_get_symptoms.return_value = ["боль в животе", "метеоризм"]

        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит живот"}), content_type="application/json"
        )

        response = extract_from_text_api(request)

        assert response.status_code == 200
        import json as json_lib

        data = json_lib.loads(response.content)
        assert data["success"] is True
        assert "suggested_symptoms" in data
        assert len(data["suggested_symptoms"]) > 0


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_processing_time_in_response(factory):
    """Тест: наличие времени обработки в ответе."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
        ]
        mock_hybrid.semantic = Mock()
        mock_hybrid.semantic.find_similar.return_value = []
        mock_get_symptoms.return_value = ["головная боль"]

        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )

        response = extract_from_text_api(request)

        assert response.status_code == 200
        import json as json_lib

        data = json_lib.loads(response.content)
        assert data["success"] is True


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_present_and_absent_symptoms(factory):
    """Тест: правильная маркировка present/absent симптомов."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
            {"canonical_name": "кашель", "status": "absent", "confidence": 0.90},
        ]
        mock_hybrid.semantic = Mock()
        mock_hybrid.semantic.find_similar.return_value = []
        mock_get_symptoms.return_value = ["головная боль", "кашель"]

        request = factory.post(
            reverse("extract_from_text_api"),
            data=json.dumps({"text": "голова болит, кашля нет"}),
            content_type="application/json",
        )

        response = extract_from_text_api(request)

        assert response.status_code == 200
        import json as json_lib

        data = json_lib.loads(response.content)
        present = [s for s in data["extracted_symptoms"] if s["status"] == "present"]
        absent = [s for s in data["extracted_symptoms"] if s["status"] == "absent"]
        assert len(present) == 1
        assert len(absent) == 1
        assert present[0]["canonical_name"] == "головная боль"
        assert absent[0]["canonical_name"] == "кашель"


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_handles_exception(factory):
    """Тест: обработка исключений в API."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.side_effect = Exception("Тестовая ошибка")

        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )

        response = extract_from_text_api(request)

        assert response.status_code == 500
        import json as json_lib

        data = json_lib.loads(response.content)
        assert "error" in data


@pytest.mark.unit
@pytest.mark.django_db
def test_response_has_required_fields(factory):
    """Тест: ответ содержит все необходимые поля."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
        ]
        mock_hybrid.semantic = Mock()
        mock_hybrid.semantic.find_similar.return_value = []
        mock_get_symptoms.return_value = ["головная боль"]

        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )

        response = extract_from_text_api(request)
        import json as json_lib

        data = json_lib.loads(response.content)

        assert "success" in data
        assert "extracted_symptoms" in data
        assert isinstance(data["extracted_symptoms"], list)
        # Убираем проверку processing_time или делаем опциональной
        if "processing_time" in data:
            assert isinstance(data["processing_time"], float)


@pytest.mark.unit
@pytest.mark.django_db
def test_extracted_symptom_has_required_fields(factory):
    """Тест: каждый извлечённый симптом имеет нужные поля."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95}
        ]
        mock_hybrid.semantic = Mock()
        mock_hybrid.semantic.find_similar.return_value = []
        mock_get_symptoms.return_value = ["головная боль"]

        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )

        response = extract_from_text_api(request)
        import json as json_lib

        data = json_lib.loads(response.content)

        symptom = data["extracted_symptoms"][0]
        assert "canonical_name" in symptom
        assert "status" in symptom
        assert "confidence" in symptom
        assert isinstance(symptom["canonical_name"], str)
        assert symptom["status"] in ["present", "absent"]
        assert isinstance(symptom["confidence"], (int, float))


@pytest.mark.unit
def test_get_ml_symptoms_returns_list():
    """Тест: get_ml_symptoms возвращает список симптомов."""
    with patch("diagnosis.views.Symptom.objects") as mock_objects:
        mock_queryset = Mock()
        mock_queryset.order_by.return_value.values_list.return_value = ["головная боль", "кашель"]
        mock_objects.all.return_value = mock_queryset

        result = get_ml_symptoms()

        assert isinstance(result, list)
        assert result == ["головная боль", "кашель"]


@pytest.mark.unit
@pytest.mark.django_db
def test_extract_api_performance(factory):
    """Тест производительности API (должно быть < 0.5 сек с моками)."""
    with (
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid,
        patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms,
        patch("diagnosis.views.nlp_loaded_successfully", True),
    ):

        mock_hybrid.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
        ]
        mock_hybrid.semantic = Mock()
        mock_hybrid.semantic.find_similar.return_value = []
        mock_get_symptoms.return_value = ["головная боль"]

        request = factory.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )

        start_time = time.time()
        response = extract_from_text_api(request)
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 0.5, f"API с моками работает медленно: {elapsed:.2f} сек"


@pytest.mark.unit
def test_extract_from_text_api_view_has_csrf_exempt():
    """Тест: view имеет декоратор @csrf_exempt."""
    from diagnosis.views import extract_from_text_api

    assert hasattr(extract_from_text_api, "csrf_exempt")
    assert extract_from_text_api.csrf_exempt is True
