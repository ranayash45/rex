from django.core.files.base import ContentFile
from django.shortcuts import render
from rex.settings import MEDIA_URL
from tools.models import Upload
from .Learning.ImageClassification import IdentifyMango
import cv2
import numpy as np


# Create your views here.
def Classification(request):
    return render(request,'tools/Classification/views/Mango.html')

def ListClassifiers(request):
    return render(request,'tools/Classification/views/ListClassifier.html')

def ClassificationUploadMethod(request):
    if request.method == 'POST' and request.FILES['classificationupload']:
        myfile = request.FILES['classificationupload']
        if myfile.name.lower().endswith('.jpeg') or myfile.name.lower().endswith('.jpg'):
            #print(myfile)
            filedata = myfile.read()
            img_np = cv2.imdecode(np.asarray(bytearray(filedata),dtype=np.uint8),-1)
            #img_np = cv2.imread(MEDIA_URL +'/'+filename)
            Result = IdentifyMango(img_np)
            image_string = cv2.imencode('.jpg',Result['img'])
            upload = Upload()
            upload.file.save(myfile.name,ContentFile(image_string[1]))
            upload.save()
            Result['uploaded_file_url'] = MEDIA_URL + upload.file.name
            return render(request, 'tools/Classification/views/ClassificationUpload.html',Result)
        else:
            return render(request, 'tools/Classification/views/ClassificationUpload.html', {
                'error_message': 'Please provide image in jpeg format'
            })

    return render(request,'tools/Classification/views/ClassificationUpload.html')

