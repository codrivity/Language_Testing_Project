

from django.urls import path
from . import views


urlpatterns = [
    path('get-csrf-token/',views.get_csrf_token),
    path("user-details/",views.user_detail),
    path("add-user/",views.add_user),
    path("login/<str:email>/<str:password>/",views.login),
    path('change-password/',views.change_password),
    path('user_count/', views.user_count, name='user_count'),
    path('upload_cv/', views.upload_cv, name='upload-cv'),
    path('count_cv/', views.count_cv, name='upload-cv'),
    path('pi_test/', views.pi_test, name='predictive-index-test'),
    path('send_email_otp/<str:email>/', views.send_email_otp , name='send_email_otp'),
    path('check_score_id/<str:email>/<str:score>/', views.check_score_id , name='check_score_id'),
    
]

