from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
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
        fs = FileSystemStorage()
        if myfile.name.lower().endswith('.jpeg') or myfile.name.lower().endswith('.jpg'):
            fs.delete('test.jpeg')
            filename = fs.save('test.jpeg', myfile)
            img_np = cv2.imread(fs.path(filename))
            Result = IdentifyMango(img_np)
            print(Result)
            cv2.imwrite(fs.path(filename),Result['img'])

            uploaded_file_url = fs.url(filename)
            Result['uploaded_file_url'] = uploaded_file_url
            return render(request, 'tools/Classification/views/ClassificationUpload.html',Result)
        else:
            return render(request, 'tools/Classification/views/ClassificationUpload.html', {
                'error_message': 'Please provide image in jpeg format'
            })

    return render(request,'tools/Classification/views/ClassificationUpload.html')

