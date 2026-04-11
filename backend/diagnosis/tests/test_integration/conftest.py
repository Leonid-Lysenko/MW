# C:\Users\Leo\PyWork\MedWebsite\backend\diagnosis\tests\test_integration\conftest.py
"""
Фикстуры для интеграционных тестов приложения diagnosis.

Содержит набор фикстур для тестирования интеграционных сценариев и потоков данных,
включая фикстуры для NLP модуля.
"""

import sys
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from django.test import Client


# ============================================================================
# Глобальный мок тяжёлых модулей
# ============================================================================

# Мокаем все тяжёлые модули на уровне sys.modules
mock_ml = MagicMock()
mock_ml.nlp = MagicMock()
mock_ml.nlp.extractors = MagicMock()
mock_ml.nlp.extractors.hybrid = MagicMock()
mock_ml.nlp.extractors.rule_based = MagicMock()
mock_ml.nlp.extractors.semantic_search = MagicMock()
mock_ml.nlp.extractors.ner_rubio_finetuned = MagicMock()

# Создаём мок для HybridExtractor
mock_hybrid_instance = MagicMock()
mock_hybrid_instance.extract.return_value = [
    {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
    {"canonical_name": "кашель", "status": "present", "confidence": 0.90},
]
mock_hybrid_instance.semantic = MagicMock()
mock_hybrid_instance.semantic.find_similar.return_value = []

mock_hybrid_class = MagicMock()
mock_hybrid_class.return_value = mock_hybrid_instance
mock_ml.nlp.extractors.hybrid.HybridExtractor = mock_hybrid_class

# Подменяем модули в sys.modules
sys.modules['ml'] = mock_ml
sys.modules['ml.nlp'] = mock_ml.nlp
sys.modules['ml.nlp.extractors'] = mock_ml.nlp.extractors
sys.modules['ml.nlp.extractors.hybrid'] = mock_ml.nlp.extractors.hybrid
sys.modules['ml.nlp.extractors.rule_based'] = mock_ml.nlp.extractors.rule_based
sys.modules['ml.nlp.extractors.semantic_search'] = mock_ml.nlp.extractors.semantic_search
sys.modules['ml.nlp.extractors.ner_rubio_finetuned'] = mock_ml.nlp.extractors.ner_rubio_finetuned

# Также мокаем transformers и torch
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()


@pytest.fixture(autouse=True)
def mock_nlp_dependencies():
    """Автоматически мокаем NLP зависимости для всех интеграционных тестов."""
    with patch("diagnosis.views.hybrid_extractor") as mock_extractor, \
         patch("diagnosis.views.nlp_loaded_successfully", True), \
         patch("diagnosis.views.get_ml_symptoms") as mock_get_symptoms:
        
        mock_extractor.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
            {"canonical_name": "кашель", "status": "present", "confidence": 0.90},
        ]
        mock_extractor.semantic = Mock()
        mock_extractor.semantic.find_similar.return_value = [
            {"canonical_name": "мигрень", "confidence": 0.85},
        ]
        mock_get_symptoms.return_value = ["головная боль", "кашель", "температура"]
        
        yield


# ============================================================================
# Остальные фикстуры
# ============================================================================

@pytest.fixture
def client():
    """Фикстура Django test client для тестирования HTTP-запросов."""
    return Client()


@pytest.fixture
def common_symptoms():
    """Фикстура распространенных симптомов для тестирования диагностики."""
    return ["Кашель", "Высокая температура", "Головная боль"]


@pytest.fixture
def respiratory_symptoms():
    """Фикстура симптомов респираторных заболеваний для специализированного тестирования."""
    return ["Кашель", "Насморк", "Боль в горле", "Чихание"]


@pytest.fixture
def minimal_symptoms():
    """Фикстура минимального набора симптомов для тестирования граничных случаев."""
    return ["Усталость"]


@pytest.fixture
def navigation_urls():
    """Фикстура всех основных URL приложения для тестирования навигации."""
    return {
        "home": "/",
        "predict": "/predict/",
        "about": "/about/",
        "how_to_use": "/how-to-use/",
        "knowledge_base": "/knowledge-base/",
        "disease_detail": "/disease/{}/",
        "disease_detail_kb": "/knowledge-base/disease/{}/",
        "extract_api": "/api/extract-from-text/",
    }


@pytest.fixture
def common_disease_names():
    """Фикстура распространенных названий заболеваний для тестирования страниц заболеваний."""
    return ["Грипп", "Простуда", "COVID-19", "Ангина"]


@pytest.fixture
def error_scenarios():
    """Фикстура сценариев которые могут вызвать ошибки для тестирования обработки исключений."""
    return {
        "invalid_symptoms": ["НесуществующийСимптом123", "AnotherFakeSymptom"],
        "special_chars": ["Симп%том@", "Тест#1", "Сим&птом"],
        "empty_data": [],
        "very_long_name": ["ОченьДлинноеНазваниеСимптома" * 10],
        "sql_injection": ["'; DROP TABLE diseases; --", "OR 1=1"],
        "xss_attempt": [
            '<script>alert("xss")</script>',
            "<img src=x onerror=alert(1)>",
        ],
    }


@pytest.fixture
def nonexistent_diseases():
    """Фикстура несуществующих названий заболеваний для тестирования обработки ошибок."""
    return [
        "НесуществующаяБолезнь123",
        "FakeDiseaseXYZ",
        "ТестоваяБолезньНеНайдена",
        "NonExistentDiseaseName",
    ]


@pytest.fixture
def invalid_urls():
    """Фикстура некорректных URL для тестирования обработки ошибок маршрутизации."""
    return [
        "/disease//",
        "/predict//",
        "/knowledge-base/disease//",
        "/invalid-page/",
        "/../../etc/passwd",
        "/disease/<script>alert(1)</script>/",
    ]


@pytest.fixture
def nlp_test_texts():
    """Фикстура тестовых текстов для NLP API."""
    return {
        "simple": "болит голова",
        "multiple": "болит голова и кашель",
        "with_negation": "голова болит, кашля нет",
        "complex": "сильная головная боль, температура 38, слабость и ломота в теле",
        "with_typo": "тммпература поднялась до 39",
        "empty": "",
        "long": "У меня болит голова уже третий день, особенно в висках. Также беспокоит кашель, сухой такой, и заложенность носа. Температуры вроде нет, но чувствую слабость и ломоту в теле.",
    }


@pytest.fixture
def mock_nlp_success_response():
    """Фикстура мок-ответа успешного распознавания симптомов."""
    return {
        "success": True,
        "extracted_symptoms": [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
            {"canonical_name": "кашель", "status": "present", "confidence": 0.90},
        ],
        "suggested_symptoms": [
            {"name": "мигрень", "similarity": 0.85},
            {"name": "головокружение", "similarity": 0.72},
        ],
        "processing_time": 1.23,
    }


@pytest.fixture
def mock_nlp_error_response():
    """Фикстура мок-ответа с ошибкой."""
    return {"error": "Ошибка при анализе текста"}


@pytest.fixture
def nlp_api_headers():
    """Фикстура заголовков для NLP API запросов."""
    return {"content_type": "application/json"}