import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ========== CONFIGURATION ==========
IMAGE_SIZE = [224, 224]
BATCH_SIZE = 32
EPOCHS = 15

DATA_DIR = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Computer Vision/Waste Classifier/Dataset/train"
MODEL_PATH = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Dry_Wet_classifier.h5"

# ========== CHECK DATA FOLDER ==========
print("📁 Checking folders and file count per class:")
for cls in os.listdir(DATA_DIR):
    cls_path = os.path.join(DATA_DIR, cls)
    if os.path.isdir(cls_path):
        print(f"  - {cls}: {len(os.listdir(cls_path))} images")

# ========== DATA AUGMENTATION ==========
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ========== TRAINING AND VALIDATION GENERATORS ==========
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ========== LOAD RESNET50 BASE MODEL ==========
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + [3])

# Freeze initial layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# ========== BUILD CUSTOM HEAD ==========
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ========== COMPILE MODEL ==========
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ========== TRAIN MODEL ==========
print("🚀 Starting Training...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
)

# ========== SAVE FINAL MODEL ==========
model.save(MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")
