from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.urls import reverse
from .models import Employee, Attendance
from .forms import EmployeeForm

import cv2
import numpy as np
import base64
import face_recognition
import json
import os
from datetime import datetime

# ========================= Home Views =========================
def home(request):
    return render(request, 'home.html')

def cam_home(request):
    return render(request, 'cam_home.html')

def hr_dashboard_view(request):
    return render(request, 'hr_dashboard.html')

def employee_login_view(request):
    return render(request, 'cam_home.html')

# ========================= Employee Views =========================
def add_employee(request):
    if request.method == "POST":
        print("FILES:", request.FILES)  # Debugging: See if the file is being uploaded
        form = EmployeeForm(request.POST, request.FILES)
        if form.is_valid():
            employee = form.save()
            print("Saved file path:", employee.image.path)  # Debugging: Check where it's saved
            return redirect('employee_list')  
        else:
            print("Form errors:", form.errors)  # Debugging: See form errors

    else:
        form = EmployeeForm()
    return render(request, 'add_employee.html', {'form': form})


def employee_list(request):
    employees = Employee.objects.all()
    return render(request, 'employee_list.html', {'employees': employees})

# ========================= Face Recognition Setup =========================
known_faces = []
known_names = []  # FIXED: Changed from dict to list
employee_data = {}

def load_known_faces():
    global known_faces, known_names, employee_data
    known_faces.clear()
    known_names.clear()
    employee_data.clear()

    KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, "employees")

    for employee in Employee.objects.all():
        if employee.image:
            image_path = os.path.join(KNOWN_FACES_DIR, employee.image.name)
            print(f"🖼️ Checking Image: {image_path}")

            if os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(employee.name)
                    employee_data[employee.name] = {
                        "emp_id": employee.emp_id,
                        "designation": employee.designation,
                        "image_url": f"{settings.MEDIA_URL}{employee.image.name}"
                    }
                    print(f"✔️ Face loaded for {employee.name}")
                else:
                    print(f"❌ No face found in {image_path}")
            else:
                print(f"❌ Image not found: {image_path}")


# ========================= Attendance Processing =========================
# def process_attendance(frame):
#     global known_faces, known_names, employee_data

#     # Convert frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect face locations and encodings
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for face_encoding in face_encodings:
#         print("\n🔍 Checking Match for Face Encoding:", face_encoding)  # Debugging

#         # Compare face encodings with known faces
#         matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
#         face_distances = face_recognition.face_distance(known_faces, face_encoding)

#         print("🎯 Matches:", matches)  # Debugging
#         print("📏 Face Distances:", face_distances)  # Debugging

#         if True in matches:
#             best_match_index = np.argmin(face_distances)
#             name = known_names[best_match_index]
#             print(f"✅ Match Found: {name}")
#         else:
#             name = "Unknown"
#             print("❌ No Match Found")

#     return name  # Ensure the function returns name properly


# ========================= Camera Capture =========================
@csrf_exempt
def capture_attendance(request):
    """Captures image from the camera (for testing only)."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return JsonResponse({'message': "Face recognized and attendance marked"})
    
    return JsonResponse({'message': 'Camera not accessible'}, status=500)

# ========================= Live Video Feed =========================
def generate_frames():
    camera = cv2.VideoCapture(0)  # FIXED: Initialize inside function
    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if known_faces:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    camera.release()  # FIXED: Release camera after use

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# ========================= Employee Detection =========================
def get_detected_employee(request):
    emp_id = request.GET.get('emp_id')

    if not emp_id:
        return JsonResponse({'error': 'No employee ID provided'}, status=400)

    try:
        employee = Employee.objects.get(emp_id=emp_id)
        return JsonResponse({
            'status': 'success',
            'emp_id': employee.emp_id,
            'name': employee.name,
            'designation': employee.designation,
            'image_url': f"{settings.MEDIA_URL}{employee.image.name}"
        })
    except Employee.DoesNotExist:
        return JsonResponse({'error': 'Employee not found'}, status=404)

def camera_page(request):
    return render(request, "camera_home.html")
