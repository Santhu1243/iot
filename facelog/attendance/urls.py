from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
<<<<<<< HEAD
from .views import capture_attendance
from .views import process_attendance  

urlpatterns = [
    path('', views.home, name='home'),
    path('attendance', views.cam_home, name='cam_home'),
    path('capture/', capture_attendance, name='capture_attendance'),
    path('process_attendance/', process_attendance, name='process_attendance'),
=======

from django.urls import path



app_name = "attendance"

urlpatterns = [
    path('', views.home, name='home'),
    path('employee-login/', views.employee_login_view, name='employee_login'),
   path('hr-dashboard/', views.hr_dashboard_view, name='hr_dashboard'),  # Use the correct function name
>>>>>>> 1c8af8bbcef1c2e3799d66bfe8f95f77c124bd63
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



