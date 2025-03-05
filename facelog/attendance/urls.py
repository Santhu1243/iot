from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import capture_attendance
from .views import process_attendance  
from .views import add_employee
from .views import employee_list  # Import your employee list view



urlpatterns = [
    path('', views.home, name='home'),
    path('employee-login/', views.employee_login_view, name='employee_login'),
    path('hr-dashboard/', views.hr_dashboard_view, name='hr_dashboard'),  
    path('attendance', views.cam_home, name='cam_home'),
    path('capture/', capture_attendance, name='capture_attendance'),
    path('process_attendance/', process_attendance, name='process_attendance'),
    path('add-employee/', add_employee, name='add_employee'),
    path('employees/', employee_list, name='employee_list'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



