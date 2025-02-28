from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import capture_attendance
from .views import process_attendance  

urlpatterns = [
    path('', views.home, name='home'),
    path('attendance', views.cam_home, name='cam_home'),
    path('capture/', capture_attendance, name='capture_attendance'),
    path('process_attendance/', process_attendance, name='process_attendance'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

