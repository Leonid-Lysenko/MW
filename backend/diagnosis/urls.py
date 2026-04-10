from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
    path("api/extract-from-text/", views.extract_from_text_api, name="extract_from_text_api"),
    path("disease/<str:disease_name>/", views.disease_detail, name="disease_detail"),
    path(
        "knowledge-base/disease/<str:disease_name>/",
        views.disease_detail,
        name="disease_detail_kb",
    ),
    path("about/", views.about, name="about"),
    path("how-to-use/", views.how_to_use, name="how_to_use"),
    path("knowledge-base/", views.knowledge_base, name="knowledge_base"),
]
