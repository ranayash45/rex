from django.core.files.base import ContentFile
from django.shortcuts import render
from django.http import JsonResponse
from rex.settings import MEDIA_URL
from tools.models import Upload
from .Learning.ImageClassification import IdentifyMango
from .Learning.DeepClassifier import PredictImage
import cv2
import numpy as np
from os import path


# Create your views here.

def Index(request):
    return render(request,'tools/Classification/views/Home.html')

def Classification(request):
    return render(request,'tools/Classification/views/Mango.html')

def ListClassifiers(request):
    return render(request,'tools/Classification/views/ListClassifier.html')

def ClassificationUploadJquery(request):
    if request.method == "POST":
        myfile = request.FILES['classificationupload']
        filedata = myfile.read()

        img_np = cv2.imdecode(np.asarray(bytearray(filedata), dtype=np.uint8), -1)
        scale_percent = 400 / img_np.shape[0] * 100

        # calculate the 50 percent of original dimensions
        width = int(img_np.shape[1] * scale_percent / 100)
        height = int(img_np.shape[0] * scale_percent / 100)

        dsize = (height,width)
        img_np = cv2.resize(img_np,dsize)

        predict = PredictImage(img_np)
        Result = IdentifyMango(img_np)
        image_string = cv2.imencode('.jpg', Result['img'])

        upload = Upload()
        upload.file.save(myfile.name, ContentFile(image_string[1]))
        upload.save()

        data = {
            "predictions":predict,
            "image_url":MEDIA_URL+upload.file.name,
            "breed_type":{
                "svm":Result["svm"],
                "dtree":Result["des"],
                "perc":Result["pes"]
            },
            "condition":{
                "svm":Result["tsvm"],
                "dtree":Result["tdes"],
                "perc":Result["tpes"]
            }
        }
        return JsonResponse(data)
    else:
        return render(request,'tools/Classification/views/NewClassificationUpload.html')

def ClassificationUploadMethod(request):
    if request.method == 'POST' and request.FILES['classificationupload']:
        myfile = request.FILES['classificationupload']
        if myfile.name.lower().endswith('.jpeg') or myfile.name.lower().endswith('.jpg'):
            #print(myfile)
            filedata = myfile.read()
            img_np = cv2.imdecode(np.asarray(bytearray(filedata),dtype=np.uint8),-1)
            predict = PredictImage(img_np)
            #img_np = cv2.imread(MEDIA_URL +'/'+filename)
            Result = IdentifyMango(img_np)
            Result['cnnpredict'] = predict
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

