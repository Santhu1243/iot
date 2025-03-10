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
            return redirect(reverse('employee_list'))
        else:
            print("Form errors:", form.errors)  # Debugging: See form errors

    else:
        form = EmployeeForm()
    return render(request, 'add_employee.html', {'form': form})


def employee_list(request):
    employees = Employee.objects.all()
    return render(request, 'employee_list.html', {'employees': employees})

# ========================= Face Recognition Setup =========================
# import os
# import face_recognition
# from django.conf import settings
# from .models import Employee

# KNOWN_FACES_DIR = "C:\\Users\\santh\\OneDrive\\Desktop\\chezzion-iot\\facelog\\media\\employees"  

# known_faces = []
# known_names = []
# employee_data = {}

# def load_known_faces():
#     global known_faces, known_names, employee_data
#     known_faces.clear()
#     known_names.clear()
#     employee_data.clear()

#     # ✅ Load known faces from directory
#     for filename in os.listdir(KNOWN_FACES_DIR):
#         if filename.endswith((".jpg", ".jpeg", ".png")):
#             file_path = os.path.join(KNOWN_FACES_DIR, filename)
#             image = face_recognition.load_image_file(file_path)
#             encodings = face_recognition.face_encodings(image)

#             if encodings:
#                 known_faces.append(encodings[0])  # ✅ Only take the first encoding
#                 known_names.append(os.path.splitext(filename)[0])

#     print(f"✅ Loaded {len(known_faces)} known faces from directory")  # Debugging

#     # ✅ Load employee faces from database
#     for employee in Employee.objects.all():
#         if employee.image:
#             image_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, employee.image.name))
#             image_url = f"{settings.MEDIA_URL}{employee.image.name}"

#             print(f"🖼️ Checking Image: {image_path}")

#             if os.path.exists(image_path):
#                 image = face_recognition.load_image_file(image_path)
#                 encodings = face_recognition.face_encodings(image)

#                 if encodings:
#                     known_faces.append(encodings[0])
#                     known_names.append(employee.name)
#                     employee_data[employee.name] = {
#                         "emp_id": employee.emp_id,
#                         "designation": employee.designation,
#                         "image_url": image_url
#                     }
#                     print(f"✔️ Face loaded for {employee.name}")
#                 else:
#                     print(f"❌ No face found in {image_path}")
#             else:
#                 print(f"❌ Image not found: {image_path}")

#     print(f"✅ Total Faces Loaded: {len(known_faces)}")



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




import cv2
import face_recognition
import numpy as np
import os
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings
from .models import Employee  # Assuming Employee model exists

# Global variables to store known faces and employee data
known_faces = []
known_names = []
employee_data = {}

# ========================= Load Known Faces =========================
def load_known_faces():
    global known_faces, known_names, employee_data
    known_faces = []
    known_names = []
    employee_data = {}

    faces_dir = os.path.join(settings.MEDIA_ROOT, "employees")  # Path where known faces are stored

    if not os.path.exists(faces_dir):
        print(f"Directory {faces_dir} does not exist. Please add known face images.")
        return

    for filename in os.listdir(faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            if (encoding := face_recognition.face_encodings(image)):
                known_faces.append(encoding[0])
                name = os.path.splitext(filename)[0]  # Get filename without extension
                known_names.append(name)

                # Fetch employee data if available
                try:
                    employee = Employee.objects.get(name=name)
                    employee_data[name] = {"emp_id": employee.emp_id, "designation": employee.designation}
                except Employee.DoesNotExist:
                    employee_data[name] = {"emp_id": "Unknown", "designation": "Unknown"}

    print(f"Loaded {len(known_faces)} known faces: {known_names}")

# ========================= Live Video Feed =========================
import cv2
import numpy as np
import face_recognition

last_detected_employee = None  # Store last detected employee globally
known_faces = []  # Load known face encodings here
known_names = []  # Load corresponding employee names here

def generate_frames():
    global last_detected_employee
    load_known_faces()
    
    camera = cv2.VideoCapture(0)  # Change index to 1 or 2 if needed
    if not camera.isOpened():
        print("❌ Error: Camera not opening.")
        return

    process_this_frame = True  # Skip frames to improve performance

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Error: Failed to capture frame.")
            break
        
        # ✅ Reduce frame size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # ✅ Process only every alternate frame (skip frames)
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
                name = "Unknown"

                if any(matches):
                    face_distances = face_recognition.face_distance(known_faces, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        last_detected_employee = {"name": name}  # Store detected employee details

                # ✅ Scale face coordinates back to full frame size
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        process_this_frame = not process_this_frame  # Skip every alternate frame

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    camera.release()






def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# ========================= Employee Detection =========================
from django.conf import settings
from django.http import JsonResponse
from django.utils.html import escape
import os

from django.http import JsonResponse

from django.http import JsonResponse
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_detected_employee(request):
    global last_detected_employee

    if last_detected_employee is None:
        logger.warning("No employee data available")
        return JsonResponse({"error": "No employee detected"}, status=404)

    if isinstance(last_detected_employee, dict):
        emp_id = last_detected_employee.get("name", "Unknown")
        image_url = last_detected_employee.get("image", "/media/employees/default_user.png")
        designation = last_detected_employee.get("designation", "Unknown")

        logger.info(f"Employee detected: {emp_id}, Image: {image_url}, Designation: {designation}")

        return JsonResponse({
            "emp_id": emp_id,
            "image": image_url,
            "designation": designation
        })

    logger.error("Invalid employee data format")
    return JsonResponse({"error": "Invalid employee data format"}, status=500)








# ========================= Camera Page =========================
def camera_page(request):
    return render(request, "camera_home.html")
