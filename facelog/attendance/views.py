from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html')

def cam_home(request):
    return render(request, 'cam_home.html')