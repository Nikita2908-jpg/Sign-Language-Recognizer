# setup_and_train.py


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# -------------------------------
# BASE PATH (SAFE & RECOMMENDED)
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "asl_dataset",
    "train",
    "asl_alphabet_train"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "asl_dataset",
    "test",
    "asl_alphabet_test"
)

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.h5")

# -------------------------------
# HYPERPARAMETERS
# -------------------------------

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 29 

# -------------------------------
# IMAGE GENERATORS
# -------------------------------

train_val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)

# test_gen = ImageDataGenerator(rescale=1./255)

# -------------------------------
# TRAIN DATA
# -------------------------------

train_data = train_val_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# -------------------------------
# VALIDATION DATA
# -------------------------------

val_data = train_val_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# -------------------------------
# TEST DATA
# -------------------------------

# test_data = test_gen.flow_from_directory(
#     TEST_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     shuffle=False
# )

# -------------------------------
# CNN MODEL
# -------------------------------

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
print(train_data.class_indices)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# TRAIN MODEL
# -------------------------------


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
print("🚀 Training started...")

try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stop],
        workers=1,
        use_multiprocessing=False,
        steps_per_epoch=300,
        validation_steps=100,
    )
except Exception as e:
    print("❌ TRAINING ERROR:", e)

# -------------------------------
# EVALUATE ON TEST DATA
# -------------------------------

# print("🧪 Evaluating on test data...")
# test_loss, test_acc = model.evaluate(test_data)
# print(f"✅ Test Accuracy: {test_acc:.4f}")

# -------------------------------
# SAVE MODEL
# -------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

print(f"🎉 Model saved successfully at:\n{MODEL_PATH}")
