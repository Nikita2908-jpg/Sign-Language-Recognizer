from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('recognition/', views.recognition_view, name='recognition'),
    path('api/upload/', views.upload_api, name='api_upload'),
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
    path('history/', views.history_view, name='history'),
    path('feedback/', views.feedback_view, name='feedback'),
]
    
