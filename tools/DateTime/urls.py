from django.urls import include,path
from . import views
urlpatterns = [
    path('CurrentTime/',views.current_datetime)
]