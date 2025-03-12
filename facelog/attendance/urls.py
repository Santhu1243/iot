from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views  

urlpatterns = [
    path('', views.home, name='home'),
    path('employee-login/', views.employee_login_view, name='camera_page'),
    path('hr-dashboard/', views.hr_dashboard_view, name='hr_dashboard'),  
    path('attendance/', views.cam_home, name='cam_home'),  # Fixed missing "/"
    path('add-employee/', views.add_employee, name='add_employee'),
    path('employees/', views.employee_list, name='employee_list'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('detected_employee/', views.get_detected_employee, name='detected_employee'),
] 

# Serve static & media files in development
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)  
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
