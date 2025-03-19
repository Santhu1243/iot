from django.shortcuts import render, redirect, get_object_or_404
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
from django.utils.timezone import now
from pytz import timezone


# ========================= Home Views =========================
def home(request):
    return render(request, 'home.html')

def cam_home(request):
    return render(request, 'cam_home.html')


from django.shortcuts import render
from .models import Employee, Attendance
from .forms import EmployeeForm
from datetime import datetime

def hr_dashboard_view(request):
    employees = Employee.objects.all()
    
    # Handling Attendance Filtering
    date_filter = request.GET.get('date', 'today')
    selected_date = datetime.now().date()

    if date_filter == 'yesterday':
        selected_date -= timedelta(days=1)
    elif date_filter == 'custom':
        selected_date = request.GET.get('custom_date', selected_date)

    attendances = Attendance.objects.filter(timestamp__date=selected_date)

    # Handling Employee Addition
    if request.method == 'POST':
        form = EmployeeForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        form = EmployeeForm()

    context = {
        'employees': employees,
        'attendances': attendances,
        'form': form,
        'selected_date': selected_date,
        'date_filter': date_filter,
    }
    
    return render(request, 'hr_dashboard.html', context)


def employee_login_view(request):
    return render(request, 'cam_home.html')

# ========================= Employee Views =========================
# def add_employee(request):
#     if request.method == "POST":
#         print("FILES:", request.FILES)  # Debugging: See if the file is being uploaded
#         form = EmployeeForm(request.POST, request.FILES)
#         if form.is_valid():
#             employee = form.save()
#             print("Saved file path:", employee.image.path)  # Debugging: Check where it's saved
#             return redirect(reverse('employee_list'))
#         else:
#             print("Form errors:", form.errors)  # Debugging: See form errors

#     else:
#         form = EmployeeForm()
#     return render(request, 'add_employee.html', {'form': form})
from django.http import JsonResponse

def add_employee(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST, request.FILES)
        if form.is_valid():
            employee = form.save()
            return JsonResponse({
                "success": True,
                "name": employee.name,
                "emp_id": employee.emp_id,
                "designation": employee.designation,
                "image_url": employee.image.url if employee.image else None
            })
        else:
            return JsonResponse({"success": False, "errors": form.errors}, status=400)

    return JsonResponse({"success": False, "message": "Invalid request"}, status=400)



def employee_list(request):
    employees = Employee.objects.all()
    return render(request, 'employee_list.html', {'employees': employees})

def get_employees(request):
    employees = Employee.objects.all()
    employee_data = [
        {
            "name": emp.name,
            "designation": emp.designation,
            "image_url": emp.image.url if emp.image else "/static/default-image.jpg"
        }
        for emp in employees
    ]
    return JsonResponse({"employees": employee_data})
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
from attendance.models import Employee  
from datetime import timedelta

# ========================= Load Known Faces =========================
known_faces = []
known_names = []
employee_data = {}  
last_detected_employee = None  
last_detected_time = None  
def load_known_faces():
    """Load employee images from the database and extract face encodings."""
    global known_faces, known_names, employee_data

    employees = Employee.objects.all()
    known_faces = []
    known_names = []
    employee_data = {}

    for emp in employees:
        image_path = emp.image.path  
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

load_known_faces()

def periodic_refresh():
    while True:
        load_known_faces()
        threading.Event().wait(300)  

threading.Thread(target=periodic_refresh, daemon=True).start()

# ========================= Live Video Feed =========================
def generate_frames():
    global last_detected_employee, last_detected_time

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Error: Camera not opening.")
        return

    process_this_frame = True  

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Error: Failed to capture frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            detected = False  

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
                        last_detected_time = time.time()  
                        detected = True

                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
                
                color = (0, 255, 0) if detected else (0, 0, 255)  
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        process_this_frame = not process_this_frame  

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    camera.release()

# ========================= Video Streaming View =========================
def video_feed(request):
    """Stream video feed with face recognition."""
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")

# ========================= Employee Detection API =========================from django.utils.timezone import localtime, now
from django.utils.timezone import localtime, now
from attendance.models import Employee, Attendance

def get_detected_employee(request):
    global last_detected_employee, last_detected_time

    if last_detected_employee is None:
        return JsonResponse({"error": "No employee detected"}, status=404)

    if last_detected_time and (time.time() - last_detected_time > 1):
        last_detected_employee = None
        return JsonResponse({"error": "No employee detected"}, status=404)

    emp_id = last_detected_employee["emp_id"]
    employee = Employee.objects.get(emp_id=emp_id)

    today = localtime(now()).date()
    attendance_exists = Attendance.objects.filter(employee=employee, timestamp__date=today).exists()

    if not attendance_exists:
        Attendance.objects.create(employee=employee)
        attendance_status = "Attendance Marked ✅"
    else:
        attendance_status = "Attendance Already Marked ✅"

    last_detected_employee["attendance_status"] = attendance_status

    return JsonResponse(last_detected_employee)



# ========================= Camera Page =========================
def camera_page(request):
    """Render the camera feed page."""
    return render(request, "camera_home.html")

# ========================= attendance list Page =========================

from datetime import datetime, timedelta
from django.shortcuts import render
from django.utils.timezone import localtime, now
import pytz
from .models import Attendance

from django.shortcuts import render
from django.utils import timezone
from datetime import datetime, timedelta
import pytz
from .models import Attendance

def api_attendance_list(request):
    india_tz = pytz.timezone("Asia/Kolkata")
    current_time = timezone.now().astimezone(india_tz)

    date_filter = request.GET.get("date", "today")

    if date_filter == "today":
        selected_date = current_time.date()
    elif date_filter == "yesterday":
        selected_date = current_time.date() - timedelta(days=1)
    elif date_filter == "custom":
        custom_date_str = request.GET.get("custom_date")
        try:
            selected_date = datetime.strptime(custom_date_str, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            selected_date = current_time.date()
    else:
        selected_date = current_time.date()

    start_datetime = india_tz.localize(datetime.combine(selected_date, datetime.min.time()))
    end_datetime = india_tz.localize(datetime.combine(selected_date, datetime.max.time()))

    start_utc = start_datetime.astimezone(pytz.utc)
    end_utc = end_datetime.astimezone(pytz.utc)

    attendances = Attendance.objects.filter(timestamp__gte=start_utc, timestamp__lte=end_utc)

    data = [
        {
            "id": record.employee.id,
            "name": record.employee.name,
            "timestamp": record.timestamp.astimezone(india_tz).strftime("%Y-%m-%d %H:%M:%S")
        }
        for record in attendances
    ]
    return JsonResponse({"attendances": data})