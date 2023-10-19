from django.urls import path
from . import views

urlpatterns = [
    path('save-report/',views.save_report),
    path('get-report/<int:userid>',views.get_report),
    path('get_total_report/',views.get_total_report_count)
 
]
