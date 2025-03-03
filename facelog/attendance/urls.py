from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

from django.urls import path



app_name = "attendance"

urlpatterns = [
    path('', views.home, name='home'),
    path('employee-login/', views.employee_login_view, name='employee_login'),
   path('hr-dashboard/', views.hr_dashboard_view, name='hr_dashboard'),  # Use the correct function name
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



