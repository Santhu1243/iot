from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from django.urls import reverse_lazy  # Import reverse_lazy

# Create your views here.
def home(request):
    return render(request, 'home.html')




from django.shortcuts import render

def employee_login_view(request):
    return render(request, 'cam_home.html')


@csrf_exempt
def capture_attendance(request):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        result = recognize_and_mark_attendance(frame)
        return JsonResponse({'message': result})

    return JsonResponse({'message': 'Camera not accessible'})

import cv2
import numpy as np
import base64
import face_recognition
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Employee, Attendance
from datetime import datetime, timedelta

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import cv2
import base64
import face_recognition
from datetime import datetime
from .models import Employee, Attendance

@csrf_exempt
def process_attendance(request):
    if request.method == "POST":
        data = json.loads(request.body)
        image_data = data.get("image")
        use_usb_camera = data.get("use_usb_camera", False)

        if use_usb_camera:
            cap = cv2.VideoCapture(1)  # Open USB camera
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return JsonResponse({"status": "error", "message": "Failed to capture image from USB camera"}, status=500)
        else:
            if not image_data:
                return JsonResponse({"status": "error", "message": "No image received"}, status=400)

            # Decode base64 image
            image_data = image_data.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Load known employees
        known_faces = Employee.objects.exclude(face_encoding="").values('id', 'name', 'face_encoding', 'emp_id', 'designation')
        known_encodings = []
        known_ids = []
        
        for emp in known_faces:
            face_encoding = emp.get('face_encoding')
            if face_encoding:
                known_encodings.append(np.array(list(map(float, face_encoding.split(',')))))
                known_ids.append(emp['id'])

        # Detect face in current frame
        face_encodings = face_recognition.face_encodings(rgb_frame)
        print(f"Detected face encodings: {face_encodings}")  # Debugging

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                matched_id = known_ids[matches.index(True)]
                employee = Employee.objects.get(id=matched_id)

                # Check if attendance already exists for today
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                attendance_exists = Attendance.objects.filter(employee=employee, timestamp__gte=today_start).exists()

                if attendance_exists:
                    return JsonResponse({
                        "status": "success",
                        "name": employee.name,
                        "emp_id": employee.emp_id,
                        "designation": employee.designation,
                        "message": "Attendance Already Marked"
                    })

                # Mark Attendance
                Attendance.objects.create(employee=employee)

                return JsonResponse({
                    "status": "success",
                    "name": employee.name,
                    "emp_id": employee.emp_id,
                    "designation": employee.designation,
                    "message": f"Attendance marked @ {datetime.now().strftime('%I:%M %p')}"
                })

        return JsonResponse({"status": "error", "message": "Face not recognized"}, status=400)
    
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)


def hr_dashboard_view(request):
    return render(request, 'hr_dashboard.html')

def cam_home(request):
    return render(request, 'cam_home.html')


from django.shortcuts import render, redirect
from .forms import EmployeeForm
from .models import Employee

def add_employee(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect(reverse_lazy('employee_list'))  # Redirect to a list view (create if needed)
    else:
        form = EmployeeForm()
    
    return render(request, 'add_employee.html', {'form': form})

def employee_list(request):
    employees = Employee.objects.all()
    return render(request, 'employee_list.html', {'employees': employees})


import cv2
import face_recognition
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Attendance  # Import the Attendance model
from django.utils.timezone import now

# Paths for known faces and captured images
KNOWN_FACES_DIR = r"facelog/employee_faces"
SAVE_FOLDER = r"facelog/employee_faces"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        if encoding := face_recognition.face_encodings(image):  # Only add if encoding exists
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name
        else:
            print(f"❌ No face found in {filename}")


@csrf_exempt
def process_attendance(request):
    if request.method == "GET":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return JsonResponse({"error": "Failed to access the camera"}, status=500)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return JsonResponse({"error": "Could not capture frame"}, status=500)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_faces = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]

                # ✅ Mark Attendance
                Attendance.objects.create(name=name)

            recognized_faces.append({"name": name, "location": [top, right, bottom, left]})

            # Draw a rectangle & label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save captured image
        image_path = os.path.join(SAVE_FOLDER, "captured_face.jpg")
        cv2.imwrite(image_path, frame)

        return JsonResponse({"faces": recognized_faces, "image": image_path})

    return JsonResponse({"error": "Invalid request method"}, status=400)


