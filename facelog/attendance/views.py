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
import threading
import time
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from .models import Employee  

# ========================= Load Known Faces =========================
known_faces = []
known_names = []
employee_data = {}

def load_known_faces():
    """Loads employee face encodings and details from the database."""
    global known_faces, known_names, employee_data

    employees = Employee.objects.all()
    known_faces.clear()
    known_names.clear()
    employee_data.clear()

    for emp in employees:
        try:
            emp_image = face_recognition.load_image_file(emp.image.path)
            emp_encoding = face_recognition.face_encodings(emp_image)[0]
            known_faces.append(emp_encoding)
            known_names.append(emp.name)
            employee_data[emp.name] = {
                "emp_name": emp.name,
                "emp_id": emp.emp_id,
                "image": emp.image.url if emp.image else "/media/employees/default_user.png",
                "designation": emp.designation
            }
        except Exception as e:
            print(f"⚠️ Error loading {emp.name}: {e}")

# ========================= Multi-threaded Camera Feed =========================
last_detected_employee = None
frame_queue = []
processed_faces = []
lock = threading.Lock()

cv2.setUseOptimized(True)  # 🔥 Enable OpenCV Optimizations

def generate_frames():
    """Captures video frames and overlays face recognition results in real time."""
    global frame_queue, processed_faces

    load_known_faces()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Error: Camera not opening.")
        return

    threading.Thread(target=process_faces, daemon=True).start()  # Run face recognition in background

    while True:
        success, frame = camera.read()
        if not success:
            break

        with lock:
            frame_queue.append(frame)

        # ✅ Overlay detected faces on frame
        with lock:
            for (top, right, bottom, left, name) in processed_faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    camera.release()

# ========================= Face Processing =========================
DEFAULT_EMPLOYEE = {
    "emp_id": "Unknown",
    "emp_name": "No Face Detected",
    "image": "/media/employees/default_user.png",
    "designation": "N/A"
}
last_detected_time = time.time()  # Store last detection time

def process_faces():
    """Processes frames for face recognition in a separate thread."""
    global last_detected_employee, processed_faces, last_detected_time

    while True:
        with lock:
            if not frame_queue:
                time.sleep(0.1)  # Prevent CPU overuse
                continue  # Skip processing if no frame available

            frame = frame_queue.pop(0)  # Get latest frame

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize for faster processing
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # 🔥 Use HOG (faster)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        detected_faces = []

        if face_encodings:  # If any face is detected
            last_detected_time = time.time()  # Update last detected time

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
            name = "Unknown"

            if any(matches):
                best_match_index = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
                name = known_names[best_match_index]
                last_detected_employee = employee_data.get(name, DEFAULT_EMPLOYEE)

            # ✅ Scale coordinates back to full size
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
            detected_faces.append((top, right, bottom, left, name))

        with lock:
            processed_faces = detected_faces  # ✅ Update global processed faces

        # 🕒 Auto-reset if no face detected for 10 sec
        if not detected_faces and time.time() - last_detected_time > 1:
            last_detected_employee = DEFAULT_EMPLOYEE

# ========================= Django Views =========================
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_detected_employee(request):
    return JsonResponse(last_detected_employee if last_detected_employee else DEFAULT_EMPLOYEE)

def camera_page(request):
    return render(request, "camera_home.html")
