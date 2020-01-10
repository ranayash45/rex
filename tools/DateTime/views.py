from django.shortcuts import render
from django.http import HttpResponse
import datetime
# Create your views here.

def current_datetime(request):
    now = datetime.datetime.now()
    html = "<Html><body>It is now %s. </body> </html>"%(now)
    return HttpResponse(html)