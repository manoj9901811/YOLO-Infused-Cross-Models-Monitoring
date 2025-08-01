from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name='index'),
    path('upload', views.upload_and_result, name='upload_video'),
    path('live-camera/', views.live_camera, name='live_camera'),
    path('detect-fall-live/', views.detect_fall_live, name='detect_fall_live'),    
    path('upload_crash/', views.upload_crash, name='upload_crash'),  # ✅ Add this
    path('algorithms/', views.algorithms_view, name='algorithms_view'),
    path('object-tracking/', views.object_tracking_view, name='object_tracking_view'),


]
