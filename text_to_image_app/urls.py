# text_to_image_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
]