from django import forms
from .models import Employee


class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['name', 'emp_id', 'designation', 'image']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Name'}),
            'emp_id': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Employee ID'}),
            'designation': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Designation'}),
            'image': forms.FileInput(attrs={'class': 'form-control-file'}),
        }

