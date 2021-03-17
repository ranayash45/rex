from django.urls import path,include
from . import views
urlpatterns = [
    path('',views.Index),
    path('Mango/',views.Classification),
    path('Mango/Upload',views.ClassificationUploadMethod),
    path('Mango/Upload2',views.ClassificationUploadJquery),
    path('ListClassifier/',views.ListClassifiers)
]