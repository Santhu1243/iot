from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html')




from django.shortcuts import render

def employee_login_view(request):
    return render(request, 'cam_home.html')

def hr_dashboard_view(request):
    return render(request, 'hr_dashboard.html')
