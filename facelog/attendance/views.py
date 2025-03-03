from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
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

@csrf_exempt
def process_attendance(request):
    if request.method == "POST":
        data = json.loads(request.body)
        image_data = data.get("image")

        if not image_data:
            return JsonResponse({"status": "error", "message": "No image received"}, status=400)

        # Decode the base64 image
        image_data = image_data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Recognize face
        known_faces = Employee.objects.exclude(face_encoding="").values('id', 'name', 'face_encoding', 'emp_id', 'designation')
        known_encodings = [np.array(list(map(float, emp['face_encoding'].split(',')))) for emp in known_faces]
        known_ids = [emp['id'] for emp in known_faces]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                matched_id = known_ids[matches.index(True)]
                employee = Employee.objects.get(id=matched_id)

                # Check if attendance was already marked today
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

                # Mark attendance if not already marked
                Attendance.objects.create(employee=employee)

                return JsonResponse({
                    "status": "success",
                    "name": employee.name,
                    "emp_id": employee.emp_id,
                    "designation": employee.designation,
                    "message": f"Attendance marked @ {datetime.now().strftime('%I:%M %p')}"
                })

        return JsonResponse({"status": "error", "message": "Face not recognized"}, status=400)


def hr_dashboard_view(request):
    return render(request, 'hr_dashboard.html')

def cam_home(request):
    return render(request, 'cam_home.html')
