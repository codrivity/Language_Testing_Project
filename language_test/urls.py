

from django.urls import path
from . import views

urlpatterns = [
    path("english/get-answer/",views.english_test_get_answer),
    path("spanish/get-answer/",views.spanish_test_get_answer),
    path("dutch/get-answer/",views.dutch_test_get_answer),
    path("english/get-conversation/",views.english_conversation_speech_to_text),
    # path('get-total-test-count/', views.get_total_test_count, name='get_total_test_count')
]