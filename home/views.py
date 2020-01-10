from django.shortcuts import render
from django.http import HttpResponseNotFound
# Create your views here.
def PageNotFound(request):
    return HttpResponseNotFound('<h1>Page Not Found</h1>')

def Home(request):
    return render(request,'Home.html')