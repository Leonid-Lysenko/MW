# flake8: noqa: E713
"""
Unit-тесты для модуля urls.py приложения diagnosis.

Тестирует разрешение URL, генерацию обратных ссылок и корректность маршрутизации.
"""

import pytest
from django.urls import resolve, reverse

from diagnosis import urls as diagnosis_urls
from diagnosis import views

# ===== Тесты импортов и базовой структуры =====


@pytest.mark.unit
def test_urls_module_importable():
    """Тест что модуль urls импортируется без ошибок."""
    from diagnosis import urls

    assert urls is not None


@pytest.mark.unit
def test_urlpatterns_exists():
    """Тест что urlpatterns существует."""
    assert hasattr(diagnosis_urls, "urlpatterns")
    assert isinstance(diagnosis_urls.urlpatterns, list)


@pytest.mark.unit
def test_urlpatterns_not_empty():
    """Тест что urlpatterns не пустой."""
    assert len(diagnosis_urls.urlpatterns) > 0


# ===== Тесты разрешения URL =====


@pytest.mark.unit
def test_home_url_resolves():
    """Тест что корневой URL разрешается в home view."""
    resolver = resolve("/")
    assert resolver.func == views.home
    assert resolver.url_name == "home"


@pytest.mark.unit
def test_predict_url_resolves():
    """Тест что URL predict разрешается в predict view."""
    resolver = resolve("/predict/")
    assert resolver.func == views.predict
    assert resolver.url_name == "predict"


@pytest.mark.unit
def test_about_url_resolves():
    """Тест что URL about разрешается в about view."""
    resolver = resolve("/about/")
    assert resolver.func == views.about
    assert resolver.url_name == "about"


@pytest.mark.unit
def test_how_to_use_url_resolves():
    """Тест что URL how_to_use разрешается в how_to_use view."""
    resolver = resolve("/how-to-use/")
    assert resolver.func == views.how_to_use
    assert resolver.url_name == "how_to_use"


@pytest.mark.unit
def test_knowledge_base_url_resolves():
    """Тест что URL knowledge_base разрешается в knowledge_base view."""
    resolver = resolve("/knowledge-base/")
    assert resolver.func == views.knowledge_base
    assert resolver.url_name == "knowledge_base"


@pytest.mark.unit
@pytest.mark.parametrize("disease_name", ["Грипп", "COVID-19", "Мигрень"])
def test_disease_detail_url_resolves(disease_name):
    """Тест что URL disease_detail разрешается с разными названиями заболеваний."""
    url = f"/disease/{disease_name}/"
    resolver = resolve(url)
    assert resolver.func == views.disease_detail
    assert resolver.url_name == "disease_detail"
    assert resolver.kwargs["disease_name"] == disease_name


@pytest.mark.unit
@pytest.mark.parametrize("disease_name", ["Грипп", "COVID-19", "Мигрень"])
def test_disease_detail_kb_url_resolves(disease_name):
    """Тест что URL disease_detail_kb разрешается с разными названиями заболеваний."""
    url = f"/knowledge-base/disease/{disease_name}/"
    resolver = resolve(url)
    assert resolver.func == views.disease_detail
    assert resolver.url_name == "disease_detail_kb"
    assert resolver.kwargs["disease_name"] == disease_name


# ===== Тесты reverse (генерации URL) =====


@pytest.mark.unit
def test_home_reverse():
    """Тест reverse для home."""
    url = reverse("home")
    assert url == "/"


@pytest.mark.unit
def test_predict_reverse():
    """Тест reverse для predict."""
    url = reverse("predict")
    assert url == "/predict/"


@pytest.mark.unit
def test_about_reverse():
    """Тест reverse для about."""
    url = reverse("about")
    assert url == "/about/"


@pytest.mark.unit
def test_how_to_use_reverse():
    """Тест reverse для how_to_use."""
    url = reverse("how_to_use")
    assert url == "/how-to-use/"


@pytest.mark.unit
def test_knowledge_base_reverse():
    """Тест reverse для knowledge_base."""
    url = reverse("knowledge_base")
    assert url == "/knowledge-base/"


@pytest.mark.unit
@pytest.mark.parametrize("disease_name", ["Грипп", "COVID-19", "Мигрень"])
def test_disease_detail_reverse(disease_name):
    """Тест reverse для disease_detail с разными названиями."""
    url = reverse("disease_detail", kwargs={"disease_name": disease_name})
    assert url.startswith("/disease/")
    assert url.endswith("/")
    assert len(url) > len("/disease//")


@pytest.mark.unit
@pytest.mark.parametrize("disease_name", ["Грипп", "COVID-19", "Мигрень"])
def test_disease_detail_kb_reverse(disease_name):
    """Тест reverse для disease_detail_kb с разными названиями."""
    url = reverse("disease_detail_kb", kwargs={"disease_name": disease_name})
    assert url.startswith("/knowledge-base/disease/")
    assert url.endswith("/")
    assert len(url) > len("/knowledge-base/disease//")


# ===== Тесты всех URL паттернов =====


@pytest.mark.unit
def test_all_url_patterns_have_names():
    """Тест что все URL паттерны имеют имена."""
    url_names = [
        "home", 
        "predict", 
        "disease_detail", 
        "disease_detail_kb", 
        "about", 
        "how_to_use", 
        "knowledge_base",
        "extract_from_text_api",  # Добавлен новый URL для NLP модуля
    ]

    for pattern in diagnosis_urls.urlpatterns:
        assert hasattr(pattern, "name")
        if pattern.name:
            assert pattern.name in url_names


@pytest.mark.unit
def test_all_views_are_functions():
    """Тест что все view являются функциями (не классами)."""
    for pattern in diagnosis_urls.urlpatterns:
        assert hasattr(pattern, "callback")
        assert callable(pattern.callback)


# ===== Тесты корректности путей =====


@pytest.mark.unit
def test_no_trailing_slashes_missing():
    """Тест что нет пропущенных trailing slashes."""
    for pattern in diagnosis_urls.urlpatterns:
        pattern_str = str(pattern.pattern)
        if not pattern_str.endswith("<path:") and not "<str:" in pattern_str:
            assert pattern_str.endswith("/") or pattern_str == ""


@pytest.mark.unit
def test_url_patterns_ordering():
    """Тест порядка URL паттернов."""
    patterns = [str(pattern.pattern) for pattern in diagnosis_urls.urlpatterns]

    expected_patterns = ["", "predict/", "disease/", "about/", "how-to-use/", "knowledge-base/"]
    for expected in expected_patterns:
        assert any(expected in pattern for pattern in patterns)


# ===== Тесты параметров URL =====


@pytest.mark.unit
def test_disease_name_parameter_format():
    """Тест формата параметра disease_name."""
    for pattern in diagnosis_urls.urlpatterns:
        pattern_str = str(pattern.pattern)
        if "<str:disease_name>" in pattern_str:
            assert "<str:disease_name>" in pattern_str
            assert "<path:" not in pattern_str


@pytest.mark.unit
def test_no_duplicate_url_names():
    """Тест что нет дублирующихся имен URL."""
    url_names = [pattern.name for pattern in diagnosis_urls.urlpatterns if pattern.name]
    assert len(url_names) == len(set(url_names))


# ===== Тесты соответствия view функций =====


@pytest.mark.unit
def test_urls_map_to_correct_views():
    """Тест что URL соответствуют правильным view функциям."""
    url_view_mapping = {
        "home": views.home,
        "predict": views.predict,
        "disease_detail": views.disease_detail,
        "about": views.about,
        "how_to_use": views.how_to_use,
        "knowledge_base": views.knowledge_base,
    }

    for pattern in diagnosis_urls.urlpatterns:
        if pattern.name in url_view_mapping:
            assert pattern.callback == url_view_mapping[pattern.name]


# ===== Smoke тесты =====


@pytest.mark.unit
def test_all_urls_reversible():
    """Тест что все именованные URL могут быть reversed."""
    for pattern in diagnosis_urls.urlpatterns:
        if pattern.name:
            try:
                if "disease_name" in str(pattern.pattern):
                    url = reverse(pattern.name, kwargs={"disease_name": "Тест"})
                else:
                    url = reverse(pattern.name)
                assert url is not None
            except Exception as e:
                pytest.fail(f"URL {pattern.name} cannot be reversed: {e}")
