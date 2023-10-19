import os
import json as js
from django.http import HttpResponse
from django.http import JsonResponse
from .models import User
from .serializers import UserSerializer
from rest_framework.renderers import JSONRenderer
from django.middleware.csrf import get_token
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
from .models import PredictiveIndex
import json, ast
from django.core.mail import send_mail
from django.conf import settings
import random
def get_csrf_token(request):
    token = get_token(request)
    return JsonResponse({'csrfToken': token})

def user_detail(request):
    if request.method=='GET':
        lst = []
        user_obj = User.objects.all()
        
        for i in user_obj:
            serializer = UserSerializer(i)
            json_data = JSONRenderer().render(serializer.data)
            lst.append(js.loads(json_data))
        
        lst_json = js.dumps(lst)
        return HttpResponse(lst_json, content_type='application/json')
 
@csrf_exempt   
def upload_cv(request):
    response   = {}
    if request.method == 'POST':
        uploaded_file = request.FILES.get('cv') 
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        pri_education = request.POST.get('primary_education')
        sec_education = request.POST.get('secondary_education')
        high_education = request.POST.get('higher_education')
        job_title = request.POST.get("job_title")
        yoe = request.POST.get("yoe")
        com_name = request.POST.get("com_name")
        job_desc = request.POST.get("job_desc")
        primary_score = request.POST.get("primary_score")
        secondary_score = request.POST.get("secondary_score")
        high_score = request.POST.get("high_score")
        if uploaded_file:
            folder_path = 'cv_files/'
            fs = FileSystemStorage(location=folder_path)
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_url = fs.url(filename)
            cv = User(cv=file_url) 
            cv.save()
            response = {
                "message": 'File uploaded and saved successfully',
                "file_url": file_url,
                "First Name": fname,
                "Last Name": lname,
                "Email" : email,
                "Phone Number" : phone,
                "Address" : address,
                "Gender" : gender,
                "Primary Education" : pri_education,
                "Primary Score" : primary_score,
                "Secondary Education" : sec_education,
                "Secondary Score" : secondary_score,
                "High Education" : high_education,
                "High Score" : high_score,
                "Job_title" : job_title,
                "Company Name" : com_name,
                "Year of Experience" : yoe,
                "Job Description" : job_desc
            }
        return JsonResponse(response)

def count_cv(request):
    if request.method == "GET":
        cv_folder = 'cv_files/'
        if os.path.exists(cv_folder) and os.path.isdir(cv_folder):
            files = os.listdir(cv_folder)
            cv_count = len(files)
        else:
            cv_count = 0
        return JsonResponse({"cv_count": cv_count})
def user_count(request):
    if request.method=='GET':
        total_users = User.objects.count()
        return JsonResponse({"total_users": total_users})
@csrf_exempt
def add_user(request):
    if request.method == 'POST':
        try:
            data = JSONParser().parse(request)
            email = data.get("email")
            password = data.get("password")
            name = data.get("name")
            try:
                user_obj = User.objects.get(email=email)
                return JsonResponse({"message": "Email already taken"}, status=200)
            except User.DoesNotExist:
                user = User(email=email, password=password, name=name)
                user.save()
                user_obj = User.objects.get(email=email, password=password)
                user_dict = {
                    'id': user_obj.id,
                    'name': user_obj.name,
                    'email': user_obj.email,
                    'password': user_obj.password
                }
                json_data = js.dumps(user_dict)
                return HttpResponse(json_data, content_type='application/json')
        except MultiValueDictKeyError:
            return JsonResponse({"message": "Missing required fields"}, status=400)
        except Exception as e:
            return JsonResponse({"message": str(e)}, status=500)
    else:
        return JsonResponse({"message": "Invalid Request Method"}, status=405)
def login(request, email, password):
    if request.method == 'GET':
        try:
            user_obj = User.objects.get(email=email, password=password)
            if user_obj is not None:
                user_dict = {
                    'id': user_obj.id,
                    'name': user_obj.name,
                    'email': user_obj.email,
                    'password': user_obj.password
                }
                json_data = js.dumps(user_dict)
                return HttpResponse(json_data, content_type='application/json')
            else:
                return HttpResponse("User Not Found")
        except User.DoesNotExist:
            return HttpResponse("User Not Found")
    else:
        return HttpResponse("Invalid Request Method")
def change_password(request):
    if request.method == 'POST':
        try:
            data = JSONParser().parse(request)
            user_obj = User.objects.get(email=data['email'])
            if user_obj is not None:
                user_obj.password = data['password']
                user_obj.save()
                return HttpResponse("Password Updated")
            else:
                return HttpResponse("Invalid Email")
        except User.DoesNotExist:
            return HttpResponse("Invalid Email")
    else:
        return HttpResponse("Invalid Request Method")
@csrf_exempt
def pi_test(request):
    piResponse = {}
    if request.method == 'POST':
        language = request.POST.get('language')
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        mname = request.POST.get('mname')
        email = request.POST.get('email')
        pi_id = request.POST.get('pi_id')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        race = request.POST.get('race')
        education = request.POST.get('education')
        radio11 = request.POST.get('radio11')
        radio12 = request.POST.get('radio12')
        radio13 = request.POST.get('radio13')
        checkbox1_pi = request.POST.get('checkbox1_pi', '[]')
        checkbox1_pi_list = ast.literal_eval(checkbox1_pi)
        checkbox2_pi = request.POST.get('checkbox2_pi')
        checkbox2_pi_list = ast.literal_eval(checkbox2_pi)
        piResponse = {
            "First Name" : fname,
            "Middle Name" : mname,
            "Last Name" : lname,
            "Email" : email,
            "Predictive Id" : pi_id,
            "Age" : age,
            "Gender" : gender,
            "Race" : race,
            "Education" : education,
            "Language" : language,
            "Question 1" : radio11,
            "Question 2" : radio12,
            "Question 3" : radio13,
            "Question 4" : checkbox1_pi_list,
            "Question 5" : checkbox2_pi_list,
        }
        return JsonResponse({"Success": piResponse})
    else:
        return JsonResponse({"Error": "Invalid Request"})
def generate_random_otp():
    otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return otp
otp_storage = {}
@csrf_exempt   
def send_email_otp(request, email):
    try:
        user = User.objects.get(email=email)
        user_email = user.email  
        print(user_email)  
        if request.method == "GET":
            otp = generate_random_otp()
            otp_storage[email] = otp  
            subject = "Checking"
            message = f"This is a dummy message {otp}"
            from_email = settings.EMAIL_HOST_USER
            recipient_list = [user_email]  
            send_mail(subject, message, from_email, recipient_list)
            return JsonResponse({"Status": "Mail Sent"})

    except User.DoesNotExist:
        return JsonResponse({"Status": "User not found"}, status=404)
    

def check_score_id(request, email,score):
    print(score)
    print(type(score))
    if request.method == "GET":
        stored_otp = otp_storage.get(email)
        print("Hello")
        print(stored_otp)
        if stored_otp and score == stored_otp:
            status = True
            return JsonResponse({"Status" : status})
        else:
            status = False
            return JsonResponse({"Status" : status})
            