import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 1. PATHS (Aapke structure ke hisaab se)
TRAIN_PATH = r"C:\Projects\Medical_Image_Classification\dataset\train"
VAL_PATH = r"C:\Projects\Medical_Image_Classification\dataset\val" # validation ke liye 'val' folder use karein
MODEL_DIR = "../model"
RESULTS_DIR = "../results"

# Directories check aur create karna
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2. IMAGE CONFIG
IMG_SIZE = 224
BATCH_SIZE = 32

# 3. DATA AUGMENTATION (Training ke liye variations, Validation ke liye sirf rescale)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 4. OPTIMIZED CNN MODEL
model = Sequential([
    # Layer 1
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(), # Training fast karne ke liye
    MaxPooling2D(2,2),

    # Layer 2
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Layer 3
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Flatten aur Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Overfitting rokne ke liye
    Dense(1, activation='sigmoid') # Binary classification ke liye best
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. ADVANCED CALLBACKS (Model ko smartly train karne ke liye)
callbacks = [
    # Best model save karega
    ModelCheckpoint(os.path.join(MODEL_DIR, 'medical_model_best.h5'), monitor='val_accuracy', save_best_only=True, verbose=1),
    
    # Agar 3 epochs tak loss nahi kam hua toh training rok do
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    
    # Accuracy rukne par learning rate kam karega
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
]

# 6. TRAINING
print("\n--- Model Training Shuru Ho Rahi Hai ---\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15, # EarlyStopping ise pehle hi rok dega agar model seekh chuka hoga
    callbacks=callbacks
)

# 7. SAVE ACCURACY PLOT
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_plot.png'))
print(f"\nModel aur Plots successfully save ho gaye hain '{MODEL_DIR}' aur '{RESULTS_DIR}' folders mein.")