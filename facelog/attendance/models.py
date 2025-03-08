import json
import face_recognition
import numpy as np
from django.db import models

import os
import face_recognition
from django.conf import settings
from django.db import models

class Employee(models.Model):
    name = models.CharField(max_length=255)
    emp_id = models.CharField(max_length=50)
    designation = models.CharField(max_length=100)
    image = models.ImageField(upload_to='employees/')  # Saves in 'media/employees/'

    def save(self, *args, **kwargs):
        # Save the image first
        super().save(*args, **kwargs)

        # Ensure image path exists before using face_recognition
        image_path = os.path.join(settings.MEDIA_ROOT, str(self.image))
        if os.path.exists(image_path):  # Check if file exists before processing
            image = face_recognition.load_image_file(image_path)
            # Process the image here (e.g., facial recognition)
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
        unique_together = ("employee", "timestamp")

    def __str__(self):
        return f"{self.employee.name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
