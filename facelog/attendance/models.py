from django.db import models



# class Employee(models.Model):
#     emp_id = models.CharField(max_length=20, unique=True)
#     name = models.CharField(max_length=100)
#     face_encoding = models.TextField()
#     designation = models.CharField(max_length=100, default=None)  


from django.db import models

class Employee(models.Model):
    name = models.CharField(max_length=100)
    employee_id = models.CharField(max_length=20, unique=True)
    designation = models.CharField(max_length=100)
    image = models.ImageField(upload_to='employee_images/')  # Store image in media/employee_images

    def __str__(self):
        return self.name


class Attendance(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.employee.name} - {self.timestamp}"