import base64
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from django.core.files.base import ContentFile
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegisterForm, LoginForm
from .models import RecognitionResult
from .models import Feedback
import numpy as np
import cv2
import os
import json
from tensorflow.keras.models import load_model
from django.conf import settings
from django.http import JsonResponse


# Load model lazily so makemigrations/migrate work even if model file is missing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(settings.BASE_DIR, 'recognition_app', 'model', 'asl_model.h5')
LANDMARK_MODEL_PATH = os.path.join(settings.BASE_DIR, 'recognition_app', 'model', 'landmark_asl_model.h5')
labels_path = os.path.join(settings.BASE_DIR, 'recognition_app', 'model', 'labels.json')

_model = None
_landmark_model = None
_labels = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, compile=False)
    return _model

def get_landmark_model():
    global _landmark_model
    if _landmark_model is None:
        _landmark_model = load_model(LANDMARK_MODEL_PATH, compile=False)
    return _landmark_model

def get_labels():
    global _labels
    if _labels is None:
        with open(labels_path) as f:
            _labels = json.load(f)
    return _labels

def index(request):
    return render(request, "recognition_app/index.html")

def about_view(request):
    return render(request, "recognition_app/about.html")

def contact_view(request):
    feedbacks = Feedback.objects.all().order_by('-id')[:5]  # Get latest 5 feedbacks
    return render(request, "recognition_app/contact.html", {'feedbacks': feedbacks})

def predict(request):
    return JsonResponse({'prediction': 'OK'})

def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user) #auto lohin after register
            messages.success(request, 'Account created successfully')
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'recognition_app/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()   
            login(request, user)
            print("User logged in:", user.username) 
            messages.success(request, "Login Successful!")
            return redirect('/recognition/')
        else:
            messages.error(request, "Invalid credentials")
    else:
        form = LoginForm()
    return render(request, 'recognition_app/login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, "Logged out successfully")
    return redirect('index')

@login_required
def feedback_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        rating = request.POST.get("rating")
        message = request.POST.get("message")

        # validation
        if int(rating) < 1 or int(rating) > 10:
            messages.error(request, "Rating must be between 1 and 10")
            return redirect("home")

        Feedback.objects.create(
            user=request.user if request.user.is_authenticated else None,
            email=email,
            rating=rating,
            message=message
        )

        messages.success(request, "Feedback submitted successfully!")
        return redirect("home")

    return redirect("home")


# Initialize MediaPipe Hand Landmarker (Tasks API for mediapipe 0.10.x)
_HAND_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'recognition_app', 'model', 'hand_landmarker.task'
)

_hand_landmarker_options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=_HAND_MODEL_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=1,
)
_hand_landmarker = HandLandmarker.create_from_options(_hand_landmarker_options)

def extract_hand(img):
    """Extract the hand region from an image using MediaPipe Tasks API."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = _hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = img.shape
        hand = result.hand_landmarks[0]

        x_coords = [lm.x * w for lm in hand]
        y_coords = [lm.y * h for lm in hand]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        pad = 20
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

        return img[y_min:y_max, x_min:x_max]

    return None  # IMPORTANT

def preprocess_for_model(img_cv):
    img = extract_hand(img_cv)

    if img is None:
        return None  

    # Convert from BGR (OpenCV default) to RGB (Keras training expectation)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (128, 128))   # match training
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0) 

    return img

@login_required(login_url='/login/')
def recognition_view(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')

        if not image_data:
            return JsonResponse({"prediction": "No image received"})

        # Decode base64
        header, imgstr = image_data.split(';base64,')
        decoded = base64.b64decode(imgstr)

        # Convert to OpenCV image
        nparr = np.frombuffer(decoded, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # MediaPipe Detection
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = _hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return JsonResponse({"prediction": "No hand detected"})

        # Extract & Normalize landmarks
        hand_landmarks = result.hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])
        
        wrist = landmarks[0]
        translated = []
        for lm in landmarks:
            translated.append([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])
        
        flat_landmarks = np.array(translated).flatten()
        max_val = np.max(np.abs(flat_landmarks))
        if max_val > 0:
            flat_landmarks = flat_landmarks / max_val

        # Predict using Landmark Model
        input_data = np.expand_dims(flat_landmarks, axis=0)
        preds = get_landmark_model().predict(input_data)
        probs = preds[0]

        predicted_index = int(np.argmax(probs))
        predicted_class = get_labels()[predicted_index]
        confidence = float(np.max(probs))

        print("Landmark Prediction:", predicted_class, "Confidence:", confidence)

        # Filter out non-alphabet classes
        if predicted_class in ['del', 'nothing', 'space']:
            return JsonResponse({"prediction": "No sign detected", "confidence": 0.0})

        # Save result
        result_obj = RecognitionResult(user=request.user)
        result_obj.image.save('capture.png', ContentFile(decoded), save=False)
        result_obj.recognized_text = predicted_class
        result_obj.confidence = confidence
        result_obj.save()

        return JsonResponse({"prediction": predicted_class, "confidence": round(confidence, 2)})
    
    return render(request, 'recognition_app/recognition.html')


def base64_to_image(base64_string):
    format, imgstr = base64_string.split(';base64,') 
    decoded_img = base64.b64decode(imgstr)
    nparr = np.frombuffer(decoded_img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@login_required(login_url='/login/')
def history_view(request):
    results = RecognitionResult.objects.filter(user=request.user).order_by('-timestamp')
    
    return render(request, 'recognition_app/history.html', {
        'results': results
    })

@login_required(login_url='/login/')
def upload_api(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({"prediction": "No image file provided"}, status=400)
            
        file = request.FILES['image']
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_cv is None:
            return JsonResponse({"prediction": "Invalid image format"}, status=400)

        # MediaPipe Detection
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = _hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return JsonResponse({"prediction": "No hand detected"})

        # Extract & Normalize landmarks
        hand_landmarks = result.hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])
        
        wrist = landmarks[0]
        translated = []
        for lm in landmarks:
            translated.append([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])
        
        flat_landmarks = np.array(translated).flatten()
        max_val = np.max(np.abs(flat_landmarks))
        if max_val > 0:
            flat_landmarks = flat_landmarks / max_val

        try:
            # Predict using Landmark Model
            input_data = np.expand_dims(flat_landmarks, axis=0)
            preds = get_landmark_model().predict(input_data)
            probs = preds[0]

            predicted_index = int(np.argmax(probs))
            predicted_class = get_labels()[predicted_index]
            confidence = float(np.max(probs))

            print("API Landmark Prediction:", predicted_class, "Confidence:", confidence)

            # Filter out non-alphabet classes
            if predicted_class in ['del', 'nothing', 'space']:
                return JsonResponse({"prediction": "No sign detected", "confidence": 0.0})

            # Save result
            result_obj = RecognitionResult(user=request.user)
            result_obj.image.save(file.name, file, save=False)
            result_obj.recognized_text = predicted_class
            result_obj.confidence = confidence
            result_obj.save()

            return JsonResponse({"prediction": predicted_class, "confidence": round(confidence, 2)})
        except Exception as e:
            print(f"Model prediction error: {e}")
            return JsonResponse({"prediction": "Error processing image model format"}, status=500)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)
