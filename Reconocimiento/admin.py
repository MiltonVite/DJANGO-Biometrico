from django.contrib import admin
from .models import Person, RecognitionLog

@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('name', 'email')

@admin.register(RecognitionLog)
class RecognitionLogAdmin(admin.ModelAdmin):
    list_display = ('person', 'recognition_time', 'confidence')
    list_filter = ('recognition_time', 'person')
    readonly_fields = ('recognition_time',)
