import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "landmarks_dataset.npz")
MODEL_DIR = os.path.join(BASE_DIR, "recognition_app", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "landmark_asl_model.h5")

if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] Dataset file not found at {DATASET_PATH}. Please run extract_landmarks.py first.")
    exit(1)

# Load dataset
data = np.load(DATASET_PATH)
X = data['X']
y = data['y']
classes = data['classes']

print(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")
print(f"Number of classes: {len(classes)}")

# Shuffle and split (80% Train, 20% Val) - Pure NumPy, no dependencies!
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split = int(X.shape[0] * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Train split: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Validation split: X_val = {X_val.shape}, y_val = {y_val.shape}")

# Build MLP model
model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"[SUCCESS] Validation Accuracy: {val_acc * 100:.2f}%")

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"[SUCCESS] Model saved successfully at:\n{MODEL_PATH}")
