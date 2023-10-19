from django.db import models

# Create your models here.
class Question(models.Model):
    id=models.AutoField(primary_key=True)
    type=models.CharField(max_length=100)
    details=models.CharField()
    answer=models.CharField()
    instruction=models.CharField()