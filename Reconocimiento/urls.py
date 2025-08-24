from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register_person, name='register_person'),
    path('recognize/', views.recognize_face, name='recognize_face'),
    path('history/', views.recognition_history, name='history'),
    path('persons/', views.person_list, name='person_list'),
]