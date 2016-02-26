from rest_framework import serializers
from CBIR.models import Image

class ImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Image
        fields = ('img',)

# class DatasetSerializer(serializers.ModelSerializer):
#
#     class Meta:
#         model = Dataset
#         fields = ('name',)
