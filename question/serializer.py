

from rest_framework import serializers


class QuestionSerializer(serializers.Serializer):
    id=serializers.IntegerField()
    type = serializers.CharField(max_length=100)
    details = serializers.CharField()
    answer = serializers.CharField()
    instruction=serializers.CharField()