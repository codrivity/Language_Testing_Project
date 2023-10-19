
import json as js
import os
from django.http import HttpResponse, JsonResponse
from question.serializer import QuestionSerializer
from .models import Question
from rest_framework.renderers import JSONRenderer
from django.middleware.csrf import get_token
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import time
import random
# Create your views here.



def handle_uploaded_image(f):
        img = Image.open(f) 
        t = time.time()
        print(f)
        file_path = f"question/images/{t}{f}"
        image_path = f"{t}{f}"
        print(file_path)
        img.save(file_path)
        return image_path

def add_question(request):
    if request.method=='POST':
        data = JSONParser().parse(request)
        if data.get("type")=="text":
            question_obj = Question(type=data.get("type"),details=data.get('details'),answer=data.get('answer'),instruction=data.get('instruction'))
            question_obj.save()
            return HttpResponse("Question Added")
    else:
        return JsonResponse({"message": "Invalid Request Method"}, status=405)
    

# @csrf_exempt
def add_image_question(request):
    if request.method=='POST':
        if request.POST["type"]=="image":
            file = request.FILES['image']
            file_path= handle_uploaded_image(file)
            question_obj = Question(type=request.POST["type"],details=file_path,answer=request.POST['answer'],instruction=request.POST['instruction'])
            question_obj.save()
            return HttpResponse("Question Added")
    else:
        return JsonResponse({"message": "Invalid Request Method"}, status=405)        
       
        
def get_question(request):
    if request.method=='GET':
        try:
            lst = []
            user_obj = Question.objects.all()
            
            for i in user_obj:
                serializer = QuestionSerializer(i)
                json_data = JSONRenderer().render(serializer.data)
                lst.append(js.loads(json_data))


            random.shuffle(lst)
            lst_json = js.dumps(lst)
            
            return HttpResponse(lst_json, content_type='application/json')
        except:
            return HttpResponse("Something went wrong")
    else:
        return JsonResponse({"message": "Invalid Request Method"}, status=405)
    

def get_image(request,url):
    
    if request.method == 'GET' and url is not None:
        image_path = os.path.join("question", "images", url)
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                response = HttpResponse(image_file.read(), content_type="image/jpeg")  # Update the content type accordingly
                return response
        else:
            return HttpResponse("Image not found", status=404)
    else:
        return JsonResponse({"message": "Invalid Request Method"}, status=405)