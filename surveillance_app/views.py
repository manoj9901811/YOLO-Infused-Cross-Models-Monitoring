import os
import uuid
import cv2
import math
import torch
import subprocess
import time
import mimetypes
from datetime import datetime

from django.conf import settings
from django.shortcuts import render, redirect
from django.core.mail import EmailMessage
from .forms import VideoURLForm
from .models import VideoUpload
import yt_dlp

# ✅ Load YOLOv5 model (can be replaced with YOLOv8 if needed)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)


def send_fall_alert_email(images):
    subject = "⚠️ Fall Detected Alert"
    body = (
        "Dear Recipient,\n\n"
        "This is a system-generated alert from FallGuard AI. A fall incident has been detected.\n"
        "Please verify the location and take appropriate safety measures.\n\n"
        "If you need assistance, contact support@example.com.\n\n"
        "Best regards,\n"
        "FallGuard AI"
    )

    email = EmailMessage(
        subject=subject,
        body=body,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=[settings.DEFAULT_FROM_EMAIL],
    )

    for img_path in images:
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()
                img_name = os.path.basename(img_path)
                mime_type, _ = mimetypes.guess_type(img_path)
                email.attach(img_name, img_data, mime_type or 'application/octet-stream')
        except Exception as e:
            print(f"Error attaching {img_path}: {e}")

    email.send(fail_silently=False)


def index(request):
    return render(request, 'surveillance_app/index.html')


def download_video_from_url(url):
    filename = f"{uuid.uuid4()}.mp4"
    filepath = os.path.join(settings.MEDIA_ROOT, filename)

    ydl_opts = {
        'outtmpl': filepath,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': True,
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Download error: {e}")
        return None

    return filepath if os.path.isfile(filepath) else None


def fall_detection_with_overlay(input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None, "Failed to read video"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20

        temp_output = output_path.replace('.mp4', '_raw.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        total_falls_detected = 0
        fall_image_paths = []
        last_fall_capture_time = 0
        fall_capture_interval = 3  # seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            results = yolo_model(frame)
            detections = results.pandas().xyxy[0]
            fall_detected_in_frame = False

            for _, row in detections.iterrows():
                if row['name'] == 'person':
                    x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                    w, h = x2 - x1, y2 - y1
                    aspect_ratio = h / float(w)

                    if aspect_ratio < 0.75:
                        fall_detected_in_frame = True
                        color = (0, 0, 255)
                        label = "Fall Detected"
                    else:
                        color = (0, 255, 0)
                        label = "Standing"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if fall_detected_in_frame and (current_time - last_fall_capture_time > fall_capture_interval):
                total_falls_detected += 1
                last_fall_capture_time = current_time
                image_name = f"fall_{uuid.uuid4()}.jpg"
                image_path = os.path.join(settings.MEDIA_ROOT, 'fall_images', image_name)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                cv2.imwrite(image_path, frame)
                fall_image_paths.append(image_path)

            out.write(frame)

        cap.release()
        out.release()

        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output,
            '-vcodec', 'libx264', '-acodec', 'aac',
            output_path
        ], check=True)

        os.remove(temp_output)

        if fall_image_paths:
            send_fall_alert_email(fall_image_paths)

        return output_path, "Fall Detected" if total_falls_detected > 0 else "No Fall Detected"

    except Exception as e:
        print("Processing failed:", e)
        return None, f"Video processing error: {str(e)}"


def upload_and_result(request):
    video = None

    if request.method == 'POST':
        form = VideoURLForm(request.POST, request.FILES)
        if form.is_valid():
            video_url = form.cleaned_data.get('video_url')
            video_file = request.FILES.get('video_file')
            raw_path = None

            if video_url:
                raw_path = download_video_from_url(video_url)
                if not raw_path:
                    form.add_error(None, "Failed to download video from URL.")
                    return render(request, 'surveillance_app/upload.html', {'form': form, 'video': None})

            elif video_file:
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                filename = f"upload_{uuid.uuid4()}.mp4"
                raw_path = os.path.join(upload_dir, filename)

                with open(raw_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)

            processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
            os.makedirs(processed_dir, exist_ok=True)

            processed_filename = f"processed_{uuid.uuid4()}.mp4"
            processed_path = os.path.join(processed_dir, processed_filename)

            processed_video_path, result = fall_detection_with_overlay(raw_path, processed_path)

            if not processed_video_path:
                if os.path.exists(raw_path):
                    os.remove(raw_path)
                form.add_error(None, result or "Video processing failed.")
                return render(request, 'surveillance_app/upload.html', {'form': form, 'video': None})

            relative_path = os.path.relpath(processed_video_path, settings.MEDIA_ROOT)
            video = VideoUpload.objects.create(
                video_url=video_url or "Uploaded File",
                processed_video=relative_path,
                result=result
            )

            if os.path.exists(raw_path):
                os.remove(raw_path)

            return render(request, 'surveillance_app/result.html', {'video': video})

    else:
        form = VideoURLForm()

    return render(request, 'surveillance_app/upload.html', {'form': form, 'video': video})



import cv2, os, uuid, threading, time
from django.shortcuts import render
from django.conf import settings
from django.core.mail import EmailMessage
from django.http import StreamingHttpResponse
from .models import *
from ultralytics import YOLO

yolo_model = YOLO(os.path.join(settings.BASE_DIR, 'yolov8n.pt'))

fall_images_live = []
last_capture_time = 0
last_sent_time = time.time()

def generate_frames():
    global fall_images_live, last_capture_time, last_sent_time
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame)
        detections = results.pandas().xyxy[0]

        fall_detected = False
        for _, row in detections.iterrows():
            if row['name'] == 'person':
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                w, h = x2 - x1, y2 - y1
                aspect_ratio = h / float(w)

                if aspect_ratio < 0.75:
                    fall_detected = True
                    label = "Fall Detected"
                    color = (0, 0, 255)

                    if time.time() - last_capture_time >= 3:
                        image_name = f"live_fall_{uuid.uuid4()}.jpg"
                        image_path = os.path.join(settings.MEDIA_ROOT, 'fall_images', image_name)
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)

                        # Save frame with annotation
                        annotated = frame.copy()
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.imwrite(image_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        fall_images_live.append(image_path)
                        last_capture_time = time.time()
                else:
                    label = "Normal"
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Send email every 60 seconds
        if time.time() - last_sent_time >= 60 and fall_images_live:
            email = EmailMessage(
                subject='Fall Detected in Live Stream!',
                body="""Dear Safety Officer,

This is a system-generated alert from IVSS Fall Detector. We have detected fall incidents in the live video feed.

Please take the following action: Review the attached images and assess the situation.

If you need assistance, please contact support@ivss-monitoring.com.

Best regards,
IVSS Fall Detector
""",
                from_email='your_email@gmail.com',
                to=['aiimagegenerator1920@gmail.com'],
            )
            for img in fall_images_live:
                email.attach_file(img)
            email.send()
            fall_images_live.clear()
            last_sent_time = time.time()

        # Encode frame to stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def detect_fall_live(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_camera(request):
    return render(request, 'surveillance_app/live_camera.html')



def algorithms_view(request):
    return render(request, 'surveillance_app/algorithms.html')

import cv2
import math
import numpy as np
import os
import subprocess
from collections import deque, defaultdict
import torch
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Load YOLOv5
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.conf = 0.4

# ---------------- Object Tracker ----------------
class Track:
    def __init__(self, bbox, track_id, label):
        self.bbox = bbox
        self.track_id = track_id
        self.label = label
        self.centroids = deque(maxlen=20)
        self.centroids.append(self.get_center(bbox))
        self.prev_center = self.get_center(bbox)
        self.stopped_frames = 0

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bbox):
        self.bbox = bbox
        center = self.get_center(bbox)
        self.centroids.append(center)
        if math.dist(center, self.prev_center) < 1.5:
            self.stopped_frames += 1
        else:
            self.stopped_frames = 0
        self.prev_center = center

class SimpleTracker:
    def __init__(self, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold

    def iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return interArea / float(box1Area + box2Area - interArea)

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            best_iou = 0
            best_track = None
            for track in self.tracks:
                iou = self.iou(track.bbox, det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track = track

            if best_track:
                best_track.update(det['bbox'])
                best_track.label = det['label']
                updated_tracks.append(best_track)
                self.tracks.remove(best_track)
            else:
                new_track = Track(det['bbox'], self.next_id, det['label'])
                self.next_id += 1
                updated_tracks.append(new_track)

        self.tracks = updated_tracks
        return self.tracks

# ---------------- Fall Detection ----------------
def detect_fall(track):
    if track.label != 'person':
        return False
    x1, y1, x2, y2 = track.bbox
    w, h = x2 - x1, y2 - y1
    return w > h * 0.9 and track.stopped_frames > 10

# ---------------- Trajectory Intersection ----------------
def do_trajectories_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

# ---------------- Crash Detection View ----------------
def upload_crash(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage()
        input_path = fs.save('uploads/' + video.name, video)
        input_full = fs.path(input_path)

        output_path = input_full.replace('.mp4', '_processed.mp4')
        output_video, status = vehicle_crash_detection_with_overlay(input_full, output_path)

        return render(request, 'crash_result.html', {
            'video_url': fs.url(output_path),
            'status': status
        })

    return render(request, 'upload_crash.html')

# ---------------- Processing Function ----------------
def vehicle_crash_detection_with_overlay(input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None, "Failed to open video"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

        temp_output = output_path.replace('.mp4', '_raw.mp4')
        out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        tracker = SimpleTracker()
        trajectory_history = defaultdict(list)
        crash_detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            df = results.pandas().xyxy[0]

            detections = []
            for _, row in df.iterrows():
                label = row['name']
                if label not in ['car', 'bus', 'truck', 'motorcycle', 'person']:
                    continue
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                detections.append({'bbox': [x1, y1, x2, y2], 'label': label})

            tracks = tracker.update(detections)
            centers = {}
            crashed_ids = set()

            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centers[t.track_id] = (cx, cy)
                trajectory_history[t.track_id].append((cx, cy))
                if len(trajectory_history[t.track_id]) > 5:
                    trajectory_history[t.track_id].pop(0)

                if detect_fall(t):
                    crashed_ids.add(t.track_id)
                    crash_detected = True

            ids = list(centers.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id1, id2 = ids[i], ids[j]
                    if len(trajectory_history[id1]) < 2 or len(trajectory_history[id2]) < 2:
                        continue
                    if do_trajectories_intersect(
                        trajectory_history[id1][-2], trajectory_history[id1][-1],
                        trajectory_history[id2][-2], trajectory_history[id2][-1]
                    ):
                        crashed_ids.add(id1)
                        crashed_ids.add(id2)
                        crash_detected = True

            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                color = (0, 0, 255) if t.track_id in crashed_ids else (0, 255, 0)
                label = f"{t.label}-{t.track_id}"
                if t.track_id in crashed_ids:
                    label += " CRASH!" if t.label != 'person' else " FALL!"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)

        cap.release()
        out.release()

        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output,
            '-vcodec', 'libx264', '-acodec', 'aac', output_path
        ], check=True)

        os.remove(temp_output)
        return output_path, "Crash Detected" if crash_detected else "No Crash Detected"

    except Exception as e:
        print("Error:", e)
        return None, f"Error: {str(e)}"

 
def upload_crash(request):
    video = None

    if request.method == 'POST':
        form = VideoURLForm(request.POST, request.FILES)
        if form.is_valid():
            video_url = form.cleaned_data.get('video_url')
            video_file = request.FILES.get('video_file')
            raw_path = None

            if video_url:
                raw_path = download_video_from_url(video_url)
                if not raw_path:
                    form.add_error(None, "Failed to download video from URL.")
                    return render(request, 'surveillance_app/crash_upload.html', {'form': form, 'video': None})

            elif video_file:
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                filename = f"crash_upload_{uuid.uuid4()}.mp4"
                raw_path = os.path.join(upload_dir, filename)

                with open(raw_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)

            processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
            os.makedirs(processed_dir, exist_ok=True)

            processed_filename = f"processed_crash_{uuid.uuid4()}.mp4"
            processed_path = os.path.join(processed_dir, processed_filename)

            processed_video_path, result = vehicle_crash_detection_with_overlay(raw_path, processed_path)

            if not processed_video_path:
                if os.path.exists(raw_path):
                    os.remove(raw_path)
                form.add_error(None, result or "Video processing failed.")
                return render(request, 'surveillance_app/crash_upload.html', {'form': form, 'video': None})

            relative_path = os.path.relpath(processed_video_path, settings.MEDIA_ROOT)
            video = VideoUpload.objects.create(
                video_url=video_url or "Uploaded Crash File",
                processed_video=relative_path,
                result=result
            )

            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        form = VideoURLForm()

    return render(request, 'surveillance_app/crash_upload.html', {'form': form, 'video': video})

import os
import uuid
import cv2
import torch
import subprocess
import smtplib
import numpy as np
from email.message import EmailMessage
from django.conf import settings
from django.shortcuts import render
from .forms import VideoURLForm
from .models import VideoUpload
import yt_dlp
import time

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Email alert function
def send_speed_alert(frame, speed, label, vehicle_id):
    alert_dir = os.path.join(settings.MEDIA_ROOT, 'alerts')
    os.makedirs(alert_dir, exist_ok=True)
    alert_path = os.path.join(alert_dir, f"speed_alert_{vehicle_id}_{uuid.uuid4()}.jpg")
    cv2.imwrite(alert_path, frame)

    msg = EmailMessage()
    msg['Subject'] = f"Speed Alert: {label} at {speed:.2f} km/h"
    msg['From'] = settings.EMAIL_HOST_USER
    msg['To'] = 'aiimagegenerator1920@gmail.com'
    msg.set_content(f"A {label} (ID: {vehicle_id}) was detected moving at {speed:.2f} km/h, exceeding the threshold.")

    with open(alert_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename='alert.jpg')

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print("Email send error:", e)

# Download from URL
def download_video_from_url(url):
    filename = f"{uuid.uuid4()}.mp4"
    filepath = os.path.join(settings.MEDIA_ROOT, filename)
    ydl_opts = {'outtmpl': filepath, 'format': 'bestvideo+bestaudio/best', 'merge_output_format': 'mp4', 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Download error: {e}")
        return None
    return filepath if os.path.isfile(filepath) else None

# Object Tracking and Speed Estimation
def object_tracking_with_overlay(input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None, "Failed to read video"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_path.replace('.mp4', '_raw.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        next_id = 0
        tracks = {}
        prev_centers = {}
        last_sent_time = {}
        iou_threshold = 0.3
        speed_threshold = 60  # km/h

        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

        def estimate_speed(center1, center2):
            pixel_dist = np.linalg.norm(np.array(center2) - np.array(center1))
            scale = 0.05  # adjust based on camera
            m_per_frame = pixel_dist * scale
            speed_mps = m_per_frame * fps
            return speed_mps * 3.6  # m/s to km/h

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            detections = results.pandas().xyxy[0]

            objects = []
            for _, row in detections.iterrows():
                label = row['name']
                if label in ['car', 'truck', 'motorbike', 'bus']:
                    x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                    objects.append([x1, y1, x2, y2, label])

            updated_tracks = {}
            used = set()

            for track_id, track in tracks.items():
                best_iou, best_idx = 0, -1
                for i, obj in enumerate(objects):
                    if i in used: continue
                    iou = compute_iou(track['bbox'], obj[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_iou > iou_threshold and best_idx != -1:
                    updated_tracks[track_id] = {'bbox': objects[best_idx][:4], 'label': objects[best_idx][4]}
                    used.add(best_idx)

            for i, obj in enumerate(objects):
                if i not in used:
                    updated_tracks[next_id] = {'bbox': obj[:4], 'label': obj[4]}
                    next_id += 1

            for track_id, info in updated_tracks.items():
                x1, y1, x2, y2 = info['bbox']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                speed = 0
                if track_id in prev_centers:
                    speed = estimate_speed(prev_centers[track_id], center)
                    if speed > speed_threshold:
                        now = time.time()
                        last_time = last_sent_time.get(track_id, 0)
                        if now - last_time >= 3:
                            send_speed_alert(frame, speed, info['label'], track_id)
                            last_sent_time[track_id] = now

                prev_centers[track_id] = center

                # Draw bounding box and label
                color = (0, 255, 0) if speed < speed_threshold else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{info['label']} ID:{track_id} {speed:.1f} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            tracks = updated_tracks
            out.write(frame)

        cap.release()
        out.release()

        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output,
            '-vcodec', 'libx264', '-acodec', 'aac',
            output_path
        ], check=True)
        os.remove(temp_output)
        return output_path, "Tracking and speed estimation completed"
    except Exception as e:
        print("Tracking failed:", e)
        return None, f"Tracking error: {str(e)}"

# Django View
def object_tracking_view(request):
    video = None
    if request.method == 'POST':
        form = VideoURLForm(request.POST, request.FILES)
        if form.is_valid():
            video_url = form.cleaned_data.get('video_url')
            video_file = request.FILES.get('video_file')
            raw_path = None

            if video_url:
                raw_path = download_video_from_url(video_url)
                if not raw_path:
                    form.add_error(None, "Failed to download video from URL.")
                    return render(request, 'surveillance_app/object_tracking.html', {'form': form, 'video': None})

            elif video_file:
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                filename = f"upload_{uuid.uuid4()}.mp4"
                raw_path = os.path.join(upload_dir, filename)
                with open(raw_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)

            processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
            os.makedirs(processed_dir, exist_ok=True)
            processed_filename = f"tracked_{uuid.uuid4()}.mp4"
            processed_path = os.path.join(processed_dir, processed_filename)

            processed_video_path, result = object_tracking_with_overlay(raw_path, processed_path)

            if not processed_video_path:
                if os.path.exists(raw_path):
                    os.remove(raw_path)
                form.add_error(None, result or "Video processing failed.")
                return render(request, 'surveillance_app/object_tracking.html', {'form': form, 'video': None})

            relative_path = os.path.relpath(processed_video_path, settings.MEDIA_ROOT)
            video = VideoUpload.objects.create(
                video_url=video_url or "Uploaded File",
                processed_video=relative_path,
                result=result
            )

            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        form = VideoURLForm()

    return render(request, 'surveillance_app/object_tracking.html', {'form': form, 'video': video})
