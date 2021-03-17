from django.urls import include,path
from . import views

urlpatterns = [
    path('',views.Index),
    path('DateTime/',include('tools.DateTime.urls')),
    path('Classification/',include('tools.Classification.urls'))
]