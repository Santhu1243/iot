from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
# from .views import process_attendance  
from .views import get_detected_employee
from .views import add_employee
from .views import employee_list, video_feed  # Import your employee list view
from .views import get_employees
from .views import api_attendance_list


from .views import download_attendance




urlpatterns = [
    path('', views.home, name='home'),
    path('employee-login/', views.employee_login_view, name='camera_page'),
    path('hr-dashboard/', views.hr_dashboard_view, name='hr_dashboard'),  

    path('attendance/', views.cam_home, name='cam_home'),  # Fixed missing "/"
    path('add-employee/', views.add_employee, name='add_employee'),
    path('employees/', views.employee_list, name='employee_list'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('detected_employee/', views.get_detected_employee, name='detected_employee'),
    # path('process_attendance/', process_attendance, name='process_attendance'),
    path('add-employee/', add_employee, name='add_employee'),
    path('employees/', employee_list, name='employee_list'),
    path('video_feed/', video_feed, name='video_feed'),
    path('detected_employee/', get_detected_employee, name='detected_employee'),
   
    path("api/employees/", get_employees, name="get_employees"),
    path("api/attendance_list/", api_attendance_list, name="api_attendance_list"),
    path('download/', views.download_attendance, name='download_attendance'),  


] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


