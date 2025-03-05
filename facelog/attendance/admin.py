from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Employee

class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('emp_id', 'name', 'designation')
    search_fields = ('name', 'emp_id')

admin.site.register(Employee, EmployeeAdmin)

from django.contrib import admin
from .models import Attendance

admin.site.register(Attendance)
