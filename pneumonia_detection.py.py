import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. SETUP ---
# Create 'results' folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# YOUR PATH (We use the one you confirmed works)
dataset_path = dataset_path = r'C:\Users\mayan\Downloads\archive\chest_xray'

# Parameters
img_height, img_width = 150, 150
batch_size = 32

print(f"📂 Loading data from: {dataset_path}")

# --- 2. DATA LOADING ---
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("   Loading Training Set...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

print("   Loading Test Set...")
test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False # Crucial for correct Evaluation
)

# --- 3. BUILD MODEL (CNN) ---
print("\n🧠 Building Model...")
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    
    # Convolutional Block 1
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Convolutional Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Convolutional Block 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Classification Head
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Binary Output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. TRAIN ---
print("\n🏋️ Starting Training (5 Epochs)...")
history = model.fit(train_generator, epochs=5, validation_data=test_generator)

# Save the trained model
model.save('pneumonia_model.h5')
print("\n💾 Model saved as 'pneumonia_model.h5'")

# --- 5. EVALUATION & SAVING RESULTS ---
print("\n📊 Generating Results & Graphs...")

# Predict classes
predictions = model.predict(test_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)
true_classes = test_generator.classes

# A. Save Metric Text File
f1 = f1_score(true_classes, predicted_classes)
acc = accuracy_score(true_classes, predicted_classes)
report = classification_report(true_classes, predicted_classes, target_names=['Normal', 'Pneumonia'])

with open('results/metrics.txt', 'w') as f:
    f.write("PNEUMONIA DETECTION - FINAL RESULTS\n")
    f.write("===================================\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(report)

# B. Save Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig('results/accuracy_plot.png')
plt.close()

# C. Save Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_plot.png')
plt.close()

# D. Save Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pneumonia'], 
            yticklabels=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/confusion_matrix.png')
plt.close()

print("\n✅ DONE! All files saved in 'results/' folder.")
