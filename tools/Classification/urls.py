from django.urls import path,include
import tools.Classification.views
urlpatterns = [
    path('Mango/',tools.Classification.views.Classification),
    path('Mango/Upload',tools.Classification.views.ClassificationUploadMethod),
    path('ListClassifier/',tools.Classification.views.ListClassifiers)
]