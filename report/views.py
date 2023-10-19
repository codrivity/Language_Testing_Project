from django.shortcuts import render
from django.http import HttpResponse
from user.models import User
from .models import Report
from rest_framework.parsers import JSONParser
import json as js
from .serializer import ReportSerializer
from rest_framework.renderers import JSONRenderer
from django.http import JsonResponse

# Create your views here.



def save_report(request):
    if request.method=='POST':
        try:
            data = JSONParser().parse(request)
            user_id = data.get("user_id")
            report_details = data.get("report_details")
            report_name=data.get("report_name")
            user_instance = User.objects.get(id=user_id)
            report = Report(user_id=user_instance,details=report_details,name=report_name)
            report.save()
            return HttpResponse("Report Saved!")
        except:
            return HttpResponse("Something went wrong")
        

def get_report(request,userid):
    if request.method=='GET':
        try:
            lst_report = Report.objects.filter(user_id=userid)
            serializer = ReportSerializer(lst_report, many=True)
            json_data = JSONRenderer().render(serializer.data)
            lst = js.loads(json_data)
            return HttpResponse(js.dumps(lst), content_type='application/json')
        except User.DoesNotExist:
            return HttpResponse("Report Not Found")
    else:
        return HttpResponse("Invalid Request Method")
    
def get_total_report_count(request):
    if request.method == 'GET':
        total_reports = Report.objects.all().count()
        print(total_reports)
        return JsonResponse({"total_reports": total_reports})
    else:
        return HttpResponse("Invalid Request Method")