from django.db import models
from user.models import User
# Create your models here.
class Report(models.Model):
    id=models.AutoField(primary_key=True)
    user_id=models.ForeignKey(User,on_delete=models.CASCADE)
    name=models.CharField()
    details=models.JSONField()

class CV(models.Model):
    id = models.AutoField(primary_key=True)