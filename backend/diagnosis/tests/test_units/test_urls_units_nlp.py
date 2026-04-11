# C:\Users\Leo\PyWork\MedWebsite\backend\diagnosis\tests\test_units\test_urls_units_nlp.py
"""Unit-тесты для URL маршрутов NLP модуля."""

import pytest
from django.urls import reverse, resolve, get_resolver


@pytest.mark.unit
def test_extract_api_url_resolves():
    """Тест: URL API извлечения симптомов резолвится в правильную view."""
    url = reverse("extract_from_text_api")
    resolver = resolve(url)
    assert resolver.func.__name__ == "extract_from_text_api"


@pytest.mark.unit
def test_extract_api_url_path():
    """Тест: правильный путь для API извлечения симптомов."""
    url = reverse("extract_from_text_api")
    assert url == "/api/extract-from-text/"


@pytest.mark.unit
def test_extract_api_url_exists_in_urlpatterns():
    """Тест: URL присутствует в конфигурации маршрутов."""
    # Получаем все URL паттерны из корневого URLconf
    resolver = get_resolver()

    # Рекурсивно ищем URL по имени
    found = False

    def check_urlpatterns(urlpatterns, prefix=""):
        nonlocal found
        for pattern in urlpatterns:
            if hasattr(pattern, "url_patterns"):
                # Это URLResolver - рекурсивно обходим вложенные паттерны
                check_urlpatterns(pattern.url_patterns, prefix + str(pattern.pattern))
            elif hasattr(pattern, "name") and pattern.name == "extract_from_text_api":
                found = True
                return
            elif hasattr(pattern, "callback"):
                # Проверяем по имени view
                if hasattr(pattern, "name") and pattern.name == "extract_from_text_api":
                    found = True
                    return

    check_urlpatterns(resolver.url_patterns)

    assert found, "URL с именем 'extract_from_text_api' не найден в URLconf"
