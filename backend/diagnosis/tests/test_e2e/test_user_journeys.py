# C:\Users\Leo\PyWork\MedWebsite\backend\diagnosis\tests\test_e2e\test_user_journeys.py
"""
E2E тесты полных пользовательских сценариев приложения diagnosis.
"""

import time
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException


def safe_click(driver, element, max_attempts=3):
    """Безопасный клик по элементу с обработкой перекрытий."""
    for attempt in range(max_attempts):
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.5)
            element.click()
            return True
        except ElementClickInterceptedException:
            time.sleep(1)
            try:
                driver.execute_script("arguments[0].click();", element)
                return True
            except:
                pass
    return False


def wait_for_results_page(browser, wait, timeout=10):
    """Ожидание загрузки страницы результатов с разными селекторами."""
    selectors = [
        ".result-card",
        ".results-container",
        ".disease-card",
        "h2:contains('Результаты')",
        "h2:contains('Возможные заболевания')",
    ]
    
    # Ждём изменения URL
    start_url = browser.current_url
    for _ in range(timeout):
        if browser.current_url != start_url:
            time.sleep(1)
            return True
        time.sleep(0.5)
    
    # Если URL не изменился, проверяем наличие элементов
    for selector in selectors:
        if selector.startswith("h2:"):
            continue  # Selenium не поддерживает :contains
        try:
            wait(browser).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            return True
        except:
            pass
    
    return False


@pytest.mark.e2e
def test_nlp_text_input_flow(browser, live_server_url, wait):
    """
    E2E тест: ввод текста → распознавание симптомов → диагностика.
    """
    browser.get(live_server_url)
    wait(browser).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # 1. Ввод текста
    textarea = wait(browser).until(EC.presence_of_element_located((By.ID, "freeTextInput")))
    textarea.send_keys("болит голова и кашель")

    # 2. Нажатие кнопки распознавания
    extract_btn = wait(browser).until(EC.element_to_be_clickable((By.ID, "extractFromTextBtn")))
    extract_btn.click()

    # 3. Проверка результатов распознавания
    wait(browser).until(EC.presence_of_element_located((By.ID, "extractionResult")))
    present_badges = browser.find_elements(By.CSS_SELECTOR, ".symptom-badge.present")
    assert len(present_badges) > 0, "Не найдены распознанные симптомы"

    # 4. Нажатие кнопки анализа (переход к результатам)
    analyze_btn = wait(browser).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'button[type="submit"]'))
    )
    
    success = safe_click(browser, analyze_btn)
    assert success, "Не удалось нажать кнопку анализа"

    # 5. Ожидание перехода на страницу результатов
    time.sleep(2)
    
    # Проверяем, что мы перешли на другую страницу
    assert browser.current_url != live_server_url, "URL не изменился, переход не произошёл"
    
    # Проверяем, что на странице есть текст
    page_content = browser.page_source.lower()
    assert "результат" in page_content or "вероятн" in page_content or "заболеван" in page_content, \
        "Не перешли на страницу результатов"


@pytest.mark.e2e
def test_manual_symptom_selection(browser, live_server_url, wait):
    """
    E2E тест: ручной выбор симптомов → диагностика.
    """
    browser.get(live_server_url)
    wait(browser).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # 1. Раскрываем секцию симптомов
    symptoms_toggle = wait(browser).until(EC.element_to_be_clickable((By.ID, "symptomsToggle")))
    browser.execute_script("arguments[0].click();", symptoms_toggle)
    time.sleep(0.5)

    # 2. Ждём, пока секция раскроется
    wait(browser).until(EC.visibility_of_element_located((By.ID, "symptomsCollapse")))

    # 3. Выбираем симптомы
    checkboxes = browser.find_elements(By.CSS_SELECTOR, 'input[name="symptoms"]')
    assert len(checkboxes) > 0, "Симптомы не найдены"

    selected = 0
    for cb in checkboxes[:3]:
        if cb.is_displayed() and cb.is_enabled():
            browser.execute_script("arguments[0].click();", cb)
            selected += 1
            time.sleep(0.2)

    assert selected > 0, "Не удалось выбрать симптомы"

    # 4. Проверяем, что симптомы выбрались
    checked_checkboxes = browser.find_elements(By.CSS_SELECTOR, 'input[name="symptoms"]:checked')
    assert len(checked_checkboxes) == selected, "Не все выбранные симптомы отмечены"

    # 5. Проверяем баннер выбранных симптомов
    banner = browser.find_element(By.ID, "selectedSymptomsBanner")
    assert banner.is_displayed(), "Баннер выбранных симптомов не отображается"

    # 6. Нажатие кнопки анализа (переход к результатам)
    analyze_btn = wait(browser).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'button[type="submit"]'))
    )
    
    success = safe_click(browser, analyze_btn)
    assert success, "Не удалось нажать кнопку анализа"

    # 7. Ожидание перехода на страницу результатов
    time.sleep(2)
    
    # Проверяем, что мы перешли на другую страницу
    assert browser.current_url != live_server_url, "URL не изменился, переход не произошёл"
    
    # 8. Проверка страницы результатов
    page_content = browser.page_source.lower()
    assert "результат" in page_content or "вероятн" in page_content or "заболеван" in page_content, \
        "Не перешли на страницу результатов"


@pytest.mark.e2e
def test_knowledge_base_navigation(browser, live_server_url, wait):
    """E2E тест: база знаний."""
    browser.get(f"{live_server_url}/knowledge-base/")
    wait(browser).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    assert "база знаний" in browser.page_source.lower() or "заболеван" in browser.page_source.lower()


@pytest.mark.e2e
def test_e2e_setup_verification(browser, live_server_url):
    """Тест проверки настроек E2E окружения."""
    assert browser is not None
    browser.get(live_server_url)
    assert "Медицинский" in browser.page_source