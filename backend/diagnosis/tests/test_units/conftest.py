"""
Фикстуры для unit-тестов приложения diagnosis.

Содержит набор фикстур для тестирования views, disease_data и других модулей.
"""

# В самом начале conftest.py, до всех остальных импортов
import sys
from unittest.mock import MagicMock

# Мокаем все тяжёлые модули на уровне sys.modules до того, как они будут импортированы
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

# Подменяем модули в sys.modules до любого импорта
sys.modules["ml"] = mock_ml
sys.modules["ml.nlp"] = mock_ml.nlp
sys.modules["ml.nlp.extractors"] = mock_ml.nlp.extractors
sys.modules["ml.nlp.extractors.hybrid"] = mock_ml.nlp.extractors.hybrid
sys.modules["ml.nlp.extractors.rule_based"] = mock_ml.nlp.extractors.rule_based
sys.modules["ml.nlp.extractors.semantic_search"] = mock_ml.nlp.extractors.semantic_search
sys.modules["ml.nlp.extractors.ner_rubio_finetuned"] = mock_ml.nlp.extractors.ner_rubio_finetuned

# Также мокаем transformers и torch
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()

import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from django.test import Client, RequestFactory

# Добавляем путь к ml модулю
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def client():
    """Фикстура Django test client для тестирования HTTP-запросов."""
    return Client()


@pytest.fixture
def factory():
    """Фикстура RequestFactory для создания тестовых запросов."""
    return RequestFactory()


@pytest.fixture
def mock_disease_info():
    """Фикстура мока информации о заболевании для тестирования."""
    return {
        "description": "Тестовое описание болезни",
        "treatment": "Тестовое лечение болезни",
        "symptoms": ["симптом1", "симптом2"],
        "severity": "medium",
        "specialist": "Терапевт",
        "category": "Инфекционные",
    }


@pytest.fixture
def mock_file_content():
    """Фикстура содержимого файла симптомов для тестирования загрузки данных."""
    return ["headache\n", "fever\n", "cough\n", "fatigue\n"]


@pytest.fixture
def predict_request_data():
    """Фикстура данных для POST запроса predict view."""
    return {"symptoms": ["кашель", "температура"]}


@pytest.fixture
def severity_test_cases():
    """Фикстура тестовых случаев для преобразования кодов серьезности в текст."""
    return [
        ("high", "ВЫСОКАЯ"),
        ("medium", "СРЕДНЯЯ"),
        ("low", "НИЗКАЯ"),
        ("unknown", "НЕОПРЕДЕЛЕНА"),
        ("variable", "ЗАВИСИТ ОТ СТАДИИ"),
        ("invalid", "НЕОПРЕДЕЛЕНА"),
    ]


@pytest.fixture
def symptom_combinations():
    """Фикстура различных комбинаций симптомов для тестирования предсказаний."""
    return [
        ["кашель"],  # один симптом
        ["кашель", "температура"],  # два симптома
        ["кашель", "температура", "головная боль"],  # три симптома
    ]


@pytest.fixture
def disease_paths():
    """Фикстура путей для тестирования disease_detail view из разных источников."""
    return [
        "/disease/Грипп/",  # из результатов диагностики
        "/knowledge-base/disease/Грипп/",  # из базы знаний
    ]


@pytest.fixture
def mock_model_classes():
    """Фикстура для мока classes_ атрибута ML модели."""
    mock_array = Mock()
    mock_array.tolist.return_value = [
        "Грипп",
        "Грипп A",
        "Грипп B",
        "Простуда",
        "COVID-19",
    ]
    return mock_array


@pytest.fixture
def probability_test_cases():
    """Фикстура тестовых случаев вероятностей заболеваний от ML модели."""
    return np.array([0.854321, 0.123456, 0.022223])


@pytest.fixture
def mock_file_operations():
    """Фикстура для моков файловых операций при тестировании загрузки симптомов."""

    class FileMocks:
        def __init__(self):
            self.mock_exists = Mock()
            self.mock_open = Mock()
            self.mock_file = MagicMock()

    mocks = FileMocks()
    mocks.mock_file.__enter__.return_value = mocks.mock_file
    mocks.mock_file.__exit__.return_value = None
    return mocks


# Фикстуры для test_disease_data_units.py


@pytest.fixture
def edge_case_disease_names():
    """Фикстура с пограничными случаями названий заболеваний для тестирования обработки."""
    return [
        "",  # пустая строка
        "   ",  # пробелы
        "НесуществующееЗаболевание123",  # несуществующее заболевание
        "ГРИПП",  # верхний регистр
        "грипп",  # нижний регистр
        " Грипп ",  # с пробелами
        "Covid-19",  # другая раскладка
    ]


@pytest.fixture
def mock_symptoms_with_special_chars():
    """Фикстура для симптомов с цифрами и специальными символами."""
    return [
        "1 стадия",
        "2-я стадия",
        "Боль 1 степени",
        "Боль 2 степени",
        "Боль 3 степени",
        "Боль в 1 пальце",
        "Абдоминальная боль",
        "Боль в горле",
        "Головная боль",
        "#симптом1",
        "@симптом",
    ]


@pytest.fixture
def mock_symptoms_list_fixture():
    """Фикстура для списка симптомов (альтернативное имя)."""
    return [
        "Абдоминальная боль",
        "Анорексия",
        "Апатия",
        "Боль в горле",
        "Боль в груди",
        "Боль в животе",
        "Боль в спине",
        "Боль в суставах",
        "Вздутие живота",
        "Головная боль",
        "Головокружение",
        "Диарея",
        "Жар",
        "Запор",
        "Кашель",
        "Лихорадка",
        "Насморк",
        "Одышка",
        "Озноб",
        "Рвота",
        "Слабость",
        "Сонливость",
        "Температура",
        "Тошнота",
        "Усталость",
    ]


# ============================================================================
# Фикстуры для тестирования rule-based экстрактора
# ============================================================================


@pytest.fixture
def mock_symptom_db():
    """Мок базы данных симптомов."""
    with patch("ml.nlp.extractors.rule_based.connection") as mock_conn:
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "головная боль"),
            (2, "кашель"),
            (3, "тошнота"),
            (4, "температура"),
            (5, "слабость"),
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        yield mock_conn


@pytest.fixture
def rule_based_extractor(mock_symptom_db):
    """Фикстура для создания rule-based экстрактора с моком БД."""
    from ml.nlp.extractors.rule_based import RuleBasedExtractor

    with patch("ml.nlp.extractors.rule_based.open", create=True) as mock_open:
        # Мок JSON файла с синонимами
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(
            {
                "головная боль": ["болит голова", "голова болит", "мигрень"],
                "кашель": ["кашляю", "кашель", "покашливание"],
                "тошнота": ["тошнит", "подташнивает", "мутит"],
                "температура": ["жар", "температурит", "высокая температура"],
                "слабость": ["усталость", "разбитость", "нет сил"],
            }
        )

        extractor = RuleBasedExtractor()
        yield extractor


# ============================================================================
# Фикстуры для тестирования semantic search экстрактора
# ============================================================================


@pytest.fixture
def mock_semantic_model():
    """Мок модели семантического поиска."""
    with patch("ml.nlp.extractors.semantic_search.SentenceTransformer") as MockTransformer:
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        MockTransformer.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_semantic_db():
    """Мок базы данных для семантического поиска."""
    with patch("ml.nlp.extractors.semantic_search.connection") as mock_conn:
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "головная боль", 0.95),
            (2, "кашель", 0.85),
            (3, "тошнота", 0.70),
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        yield mock_conn


@pytest.fixture
def semantic_extractor(mock_semantic_model, mock_semantic_db):
    """Фикстура для создания semantic search экстрактора."""
    from ml.nlp.extractors.semantic_search import SemanticSearchExtractor

    extractor = SemanticSearchExtractor(confidence_threshold=0.6)
    yield extractor


# ============================================================================
# Фикстуры для тестирования NER экстрактора
# ============================================================================


@pytest.fixture
def mock_ner_model():
    """Мок NER модели."""
    with (
        patch("ml.nlp.extractors.ner_rubio_finetuned.AutoTokenizer") as MockTokenizer,
        patch("ml.nlp.extractors.ner_rubio_finetuned.AutoModelForTokenClassification") as MockModel,
    ):

        mock_tokenizer = Mock()
        mock_tokenizer.convert_tokens_to_string.return_value = "тест"
        mock_tokenizer.convert_ids_to_tokens.return_value = ["болит", "го", "лова"]
        MockTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.config.id2label = {0: "O", 1: "B-SYMP", 2: "I-SYMP"}
        mock_model.eval.return_value = None
        MockModel.from_pretrained.return_value = mock_model

        yield mock_model


@pytest.fixture
def ner_extractor(mock_ner_model):
    """Фикстура для создания NER экстрактора."""
    from ml.nlp.extractors.ner_rubio_finetuned import RuBioRobertaNERExtractor

    with patch("ml.nlp.extractors.ner_rubio_finetuned.connection") as mock_conn:
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "головная боль"),
            (2, "кашель"),
            (3, "тошнота"),
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch("builtins.open", create=True):
            extractor = RuBioRobertaNERExtractor()
            yield extractor


# ============================================================================
# Фикстуры для тестирования hybrid экстрактора
# ============================================================================


@pytest.fixture
def mock_hybrid_components(rule_based_extractor, semantic_extractor, ner_extractor):
    """Мок компонентов гибридного экстрактора."""
    with (
        patch("ml.nlp.extractors.hybrid.RuleBasedExtractor") as MockRule,
        patch("ml.nlp.extractors.hybrid.RuBioRobertaNERExtractor") as MockNER,
        patch("ml.nlp.extractors.hybrid.SemanticSearchExtractor") as MockSemantic,
    ):

        MockRule.return_value = rule_based_extractor
        MockNER.return_value = ner_extractor
        MockSemantic.return_value = semantic_extractor

        yield {"rule": rule_based_extractor, "ner": ner_extractor, "semantic": semantic_extractor}


@pytest.fixture
def hybrid_extractor(mock_hybrid_components):
    """Фикстура для создания гибридного экстрактора."""
    from ml.nlp.extractors.hybrid import HybridExtractor

    with patch("ml.nlp.extractors.hybrid.connection") as mock_conn, patch("builtins.open", create=True):
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [(1, "головная боль"), (2, "кашель")]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        extractor = HybridExtractor()
        yield extractor


# ============================================================================
# Фикстуры для тестирования views
# ============================================================================


@pytest.fixture
def mock_hybrid_extractor_for_views():
    """Мок гибридного экстрактора для тестирования views."""
    with patch("diagnosis.views.hybrid_extractor") as mock_extractor:
        mock_extractor.extract.return_value = [
            {"canonical_name": "головная боль", "status": "present", "confidence": 0.95},
            {"canonical_name": "кашель", "status": "present", "confidence": 0.90},
        ]
        mock_extractor.semantic.find_similar.return_value = [
            {"canonical_name": "мигрень", "confidence": 0.85},
        ]
        yield mock_extractor


@pytest.fixture
def mock_nlp_loaded():
    """Мок состояния загрузки NLP модуля."""
    with patch("diagnosis.views.nlp_loaded_successfully", True), patch("diagnosis.views.hybrid_extractor", Mock()):
        yield


@pytest.fixture
def mock_nlp_not_loaded():
    """Мок состояния, когда NLP модуль не загружен."""
    with patch("diagnosis.views.nlp_loaded_successfully", False), patch("diagnosis.views.hybrid_extractor", None):
        yield


# ============================================================================
# Фикстуры для тестирования URL
# ============================================================================


@pytest.fixture
def url_names():
    """Фикстура с названиями URL для NLP."""
    return {
        "extract_api": "extract_from_text_api",
    }


# Добавьте в конец файла conftest.py


@pytest.fixture
def mock_django_dependencies():
    """Мок Django зависимостей для тестирования views."""
    with (
        patch("diagnosis.views.Symptom") as mock_symptom,
        patch("diagnosis.views.model") as mock_model,
        patch("diagnosis.views.model_loaded_successfully", True),
        patch("diagnosis.views.diseases_list", ["Грипп", "Простуда"]),
        patch("diagnosis.views.nlp_loaded_successfully", True),
        patch("diagnosis.views.hybrid_extractor") as mock_hybrid_extractor,
    ):

        # Мокаем Symptom.objects
        mock_symptom_manager = Mock()
        mock_symptom.objects = mock_symptom_manager
        mock_symptom_manager.all.return_value.order_by.return_value.values_list.return_value = [
            "головная боль",
            "кашель",
        ]

        yield {"mock_hybrid_extractor": mock_hybrid_extractor, "mock_model": mock_model, "mock_symptom": mock_symptom}
