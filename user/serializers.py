

from rest_framework import serializers
from .models import User

class UserSerializer(serializers.Serializer):
    id=serializers.IntegerField()
    name = serializers.CharField(max_length=255)
    email = serializers.CharField(max_length=255)
    password = serializers.CharField(max_length=255)
    class Meta:
        model = User
        fields = ('id', 'name', 'email', 'password', 'cv')