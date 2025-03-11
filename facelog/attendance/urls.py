from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
# from .views import process_attendance  
from .views import get_detected_employee
from .views import add_employee
from .views import employee_list, video_feed  # Import your employee list view
from .views import attendance_list



urlpatterns = [
    path('', views.home, name='home'),
    path('employee-login/', views.employee_login_view, name='camera_page'),
    path('hr-dashboard/', views.hr_dashboard_view, name='hr_dashboard'),  
    path('attendance', views.cam_home, name='cam_home'),
    # path('process_attendance/', process_attendance, name='process_attendance'),
    path('add-employee/', add_employee, name='add_employee'),
    path('employees/', employee_list, name='employee_list'),
    path('video_feed/', video_feed, name='video_feed'),
    path('detected_employee/', get_detected_employee, name='detected_employee'),
    path('attendance-list/', attendance_list, name='attendance_list'),



] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



