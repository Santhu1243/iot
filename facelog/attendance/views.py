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
import time
import threading
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings
from attendance.models import Employee  # Assuming Employee model exists

# ========================= Load Known Faces =========================
known_faces = []
known_names = []
employee_data = {}  # Store employee details dynamically
last_detected_employee = None  # Store last detected employee
last_detected_time = None  # Store timestamp of last detection

def load_known_faces():
    """Load employee images from the database and extract face encodings."""
    global known_faces, known_names, employee_data

    employees = Employee.objects.all()
    known_faces = []
    known_names = []
    employee_data = {}

    for emp in employees:
        image_path = emp.image.path  # Get the full image path
        try:
            emp_image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(emp_image)

            if encodings:
                emp_encoding = encodings[0]
                known_faces.append(emp_encoding)
                known_names.append(emp.name)
                employee_data[emp.name] = {
                    "emp_id": emp.emp_id,
                    "image": emp.image.url if emp.image else "/media/employees/default_user.png",
                    "designation": emp.designation,
                }
            else:
                print(f"⚠️ No face found in image for {emp.name}")

        except Exception as e:
            print(f"⚠️ Error processing image for {emp.name}: {e}")

# Load known faces initially
load_known_faces()

# Refresh known faces every 5 minutes
def periodic_refresh():
    while True:
        load_known_faces()
        threading.Event().wait(300)  # Refresh every 5 minutes

threading.Thread(target=periodic_refresh, daemon=True).start()

# ========================= Live Video Feed =========================
def generate_frames():
    """Generate frames from the camera feed and process face recognition."""
    global last_detected_employee, last_detected_time

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Error: Camera not opening.")
        return

    process_this_frame = True  # Skip every alternate frame to improve speed

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Error: Failed to capture frame.")
            break

        # Reduce frame size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            detected = False  # Flag to check if a known face is detected

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.4)
                name, emp_id, image, designation = "Unknown", "Unknown", "/media/employees/default_user.png", "Unknown"

                if any(matches):
                    face_distances = face_recognition.face_distance(known_faces, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        emp_id = employee_data[name]["emp_id"]
                        image = employee_data[name]["image"]
                        designation = employee_data[name]["designation"]

                        last_detected_employee = {
                            "emp_id": emp_id,
                            "emp_name": name,
                            "image": image,
                            "designation": designation,
                        }
                        last_detected_time = time.time()  # Store detection time
                        detected = True

                # Scale back coordinates to full frame size
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
                
                # Draw a rectangle around detected face and add label
                color = (0, 255, 0) if detected else (0, 0, 255)  # Green for known, Red for unknown
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        process_this_frame = not process_this_frame  # Skip alternate frames

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    camera.release()

# ========================= Video Streaming View =========================
def video_feed(request):
    """Stream video feed with face recognition."""
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")

# ========================= Employee Detection API =========================
def get_detected_employee(request):
    """Return the last detected employee details as JSON and reset after 2 seconds."""
    global last_detected_employee, last_detected_time

    if last_detected_employee is None:
        return JsonResponse({"error": "No employee detected"}, status=404)

    # Check if 2 seconds have passed since last detection
    if last_detected_time and time.time() - last_detected_time > 2:
        last_detected_employee = None  # Reset employee details
        last_detected_time = None
        return JsonResponse({"error": "No employee detected"}, status=404)

    return JsonResponse(last_detected_employee)

# ========================= Camera Page =========================
def camera_page(request):
    """Render the camera feed page."""
    return render(request, "camera_home.html")
