import json
import os
import numpy as np
import face_recognition
from django.conf import settings
from django.db import models
from django.utils.timezone import localdate

class Employee(models.Model):
    name = models.CharField(max_length=255)
    emp_id = models.CharField(max_length=50, unique=True)
    designation = models.CharField(max_length=100)
    image = models.ImageField(upload_to='employees/',null=True)  # Saves in 'media/employees/'
    face_encoding = models.TextField(blank=True, null=True)  # Store encoded face data

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)  # Save image first

        image_path = os.path.join(settings.MEDIA_ROOT, str(self.image))
        
        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            if (face_encodings := face_recognition.face_encodings(image)):
                self.face_encoding = json.dumps(face_encodings[0].tolist())  # Store as JSON
                super().save(update_fields=['face_encoding'])  # Update only this field
            else:
                print(f"No face detected in {image_path}")
        else:
            print(f"File not found: {image_path}")

    def get_encoding(self):
        """Returns face encoding as a NumPy array."""
        return np.array(json.loads(self.face_encoding)) if self.face_encoding else None

    def __str__(self):
        return self.name


class Attendance(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["employee", "timestamp"], name="unique_attendance_per_day"
            )
        ]

    def __str__(self):
        return f"{self.employee.name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"


