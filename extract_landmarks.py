import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "asl_dataset", "train", "asl_alphabet_train")
OUTPUT_PATH = os.path.join(BASE_DIR, "landmarks_dataset.npz")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "recognition_app", "model", "hand_landmarker.task")

# Initialize MediaPipe Tasks HandLandmarker
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

# We have 29 classes
classes = sorted(os.listdir(TRAIN_DIR))
print("Found classes:", classes)

X_data = []
y_data = []

MAX_IMAGES_PER_CLASS = 200

print("[INFO] Starting landmark extraction...")
for class_idx, class_name in enumerate(classes):
    class_path = os.path.join(TRAIN_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)[:MAX_IMAGES_PER_CLASS]
    count = 0
    detected = 0
    
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Convert to RGB (MediaPipe expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        result = landmarker.detect(mp_image)
        
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for lm in hand_landmarks:
                landmarks.append([lm.x, lm.y, lm.z])
            
            # Translate relative to wrist (wrist is landmark index 0)
            wrist = landmarks[0]
            translated = []
            for lm in landmarks:
                translated.append([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])
            
            # Flatten
            flat_landmarks = np.array(translated).flatten()
            
            # Scale normalization to [-1, 1]
            max_val = np.max(np.abs(flat_landmarks))
            if max_val > 0:
                flat_landmarks = flat_landmarks / max_val
            
            X_data.append(flat_landmarks)
            y_data.append(class_idx)
            detected += 1
            
        count += 1
    
    print(f"Class '{class_name}' ({class_idx + 1}/29): Processed {count} images. Hands detected: {detected}/{count}")

# Convert to numpy arrays
X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.int32)

# Save to npz file
np.savez_compressed(OUTPUT_PATH, X=X_data, y=y_data, classes=classes)
print(f"[SUCCESS] Saved dataset to {OUTPUT_PATH}")
print(f"Total dataset shape: X={X_data.shape}, y={y_data.shape}")
