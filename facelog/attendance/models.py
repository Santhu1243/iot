from django.db import models




class Employee(models.Model):
    name = models.CharField(max_length=100)
    emp_id = models.CharField(max_length=50, unique=True)
    designation = models.CharField(max_length=100)
    face_encoding = models.TextField(blank=True, null=True)  # Stores the face encoding
    image = models.ImageField(upload_to="employee_faces/", blank=True, null=True)

    def __str__(self):
        return self.name



class Attendance(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.employee.name} - {self.timestamp}"
