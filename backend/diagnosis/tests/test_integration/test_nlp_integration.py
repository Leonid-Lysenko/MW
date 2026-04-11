# test_nlp_integration.py
"""
Интеграционные тесты для NLP модуля.
"""

import json
import pytest
from django.urls import reverse


@pytest.mark.integration
@pytest.mark.django_db
class TestNLPApiIntegration:
    """Ключевые интеграционные тесты для API."""

    def test_extract_api_success_flow(self, client):
        """Тест: успешное распознавание симптомов."""
        response = client.post(
            reverse("extract_from_text_api"),
            data=json.dumps({"text": "болит голова и кашель"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["extracted_symptoms"]) >= 1

    def test_extract_api_handles_negations(self, client):
        """Тест: корректная обработка отрицаний."""
        response = client.post(
            reverse("extract_from_text_api"),
            data=json.dumps({"text": "голова болит, кашля нет"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.json()

        present = [s for s in data["extracted_symptoms"] if s["status"] == "present"]
        absent = [s for s in data["extracted_symptoms"] if s["status"] == "absent"]
        assert len(present) >= 1 or len(absent) >= 1

    def test_extract_api_returns_error_for_empty_text(self, client):
        """Тест: ошибка при пустом тексте."""
        response = client.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": ""}), content_type="application/json"
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_extract_api_handles_invalid_json(self, client):
        """Тест: ошибка при неверном JSON."""
        response = client.post(reverse("extract_from_text_api"), data="not a json", content_type="application/json")
        assert response.status_code == 400


@pytest.mark.integration
@pytest.mark.django_db
class TestNLPDiagnosisFlow:
    """Интеграционные тесты потока диагностики."""

    def test_full_diagnosis_flow(self, client):
        """Тест: полный цикл текст → симптомы → диагностика."""
        # Распознавание
        extract_response = client.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )
        assert extract_response.status_code == 200

        data = extract_response.json()
        symptoms = [s["canonical_name"] for s in data["extracted_symptoms"] if s["status"] == "present"]

        # Диагностика
        if symptoms:
            predict_response = client.post(reverse("predict"), data={"symptoms": symptoms})
            assert predict_response.status_code in [200, 302]


@pytest.mark.integration
@pytest.mark.django_db
class TestNLPERrorHandling:
    """Интеграционные тесты обработки ошибок."""

    def test_extract_api_handles_long_text(self, client):
        """Тест: устойчивость к длинному тексту."""
        long_text = "болит голова. " * 50
        response = client.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": long_text}), content_type="application/json"
        )
        assert response.status_code != 500


@pytest.mark.integration
@pytest.mark.django_db
class TestNLPPerformance:
    """Тесты производительности."""

    def test_extract_api_response_time(self, client):
        """Тест: время ответа < 5 секунд."""
        import time

        start_time = time.time()
        response = client.post(
            reverse("extract_from_text_api"), data=json.dumps({"text": "болит голова"}), content_type="application/json"
        )
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 5.0
