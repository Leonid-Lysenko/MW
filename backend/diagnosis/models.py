from django.db import models

from .disease_data import DISEASE_DATABASE
from .symptoms_data import SYMPTOMS_LIST


class Disease(models.Model):
    """Модель для хранения детальной информации о заболевании."""

    name = models.CharField(max_length=255, unique=True, verbose_name="Название заболевания")
    description = models.TextField(verbose_name="Описание")
    treatment = models.TextField(verbose_name="Лечение")
    symptoms = models.JSONField(verbose_name="Симптомы")
    severity = models.CharField(max_length=50, verbose_name="Степень серьезности")
    specialist = models.CharField(max_length=100, verbose_name="Специалист")
    category = models.CharField(max_length=100, verbose_name="Категория")

    class Meta:
        managed = False
        verbose_name = "Заболевание"
        verbose_name_plural = "Заболевания"

    def __str__(self):
        return self.name


class Symptom(models.Model):
    name = models.CharField(max_length=255, unique=True, verbose_name="Название симптома")

    class Meta:
        managed = False
        verbose_name = "Симптом"
        verbose_name_plural = "Симптомы"
        ordering = ["id"]

    def __str__(self):
        return self.name
