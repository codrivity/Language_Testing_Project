

from rest_framework import serializers
from user.models import User

class ReportSerializer(serializers.Serializer):
    id=serializers.IntegerField()
    user_id = serializers.PKOnlyObject(pk=User)
    name=serializers.CharField()
    details = serializers.JSONField()
   