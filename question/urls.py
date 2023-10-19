from django.urls import path
from . import views

urlpatterns = [
    path('add-question/',views.add_question),
    path('add-image-question/',views.add_image_question),
    path('get-question/',views.get_question),
    path('get-question-image/<str:url>',views.get_image)
]
