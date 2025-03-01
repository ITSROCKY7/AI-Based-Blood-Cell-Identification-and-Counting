# Install dependencies
!pip install tensorflow numpy opencv-python matplotlib scikit-learn

# Import libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define dataset path
dataset_path = "BCCD_Dataset/images/"

# Load and preprocess images
def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 128))  # Resize to CNN input size
    image = image / 255.0  # Normalize pixel values
    return image

# Define labels
labels = {"RBC": 0, "WBC": 1, "Platelet": 2}
data = []
target = []

# Load dataset
for category in labels.keys():
    category_path = os.path.join(dataset_path, category)
    for img in os.listdir(category_path):
        img_path = os.path.join(category_path, img)
        image = load_image(img_path)
        data.append(image)
        target.append(labels[category])

# Convert lists to NumPy arrays
X = np.array(data)
y = np.array(target)

# Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 output classes (RBC, WBC, Platelet)
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# Save model
model.save("blood_cell_classifier.h5")

# Load trained model
model = tf.keras.models.load_model("blood_cell_classifier.h5")

# Load test image
test_image_path = "sample_blood.png"
test_image = load_image(test_image_path)
test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions for prediction

# Make prediction
prediction = model.predict(test_image)
predicted_class = np.argmax(prediction)

# Map result to label
cell_types = ["RBC", "WBC", "Platelet"]
predicted_label = cell_types[predicted_class]

print(f"Predicted Cell Type: {predicted_label}")

# Load test image for contour-based detection
image = cv2.imread(test_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to detect cells
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cell_count = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    roi = image[y:y+h, x:x+w]  # Extract cell region
    roi = cv2.resize(roi, (128, 128)) / 255.0  # Preprocess

    # Predict cell type
    roi = np.expand_dims(roi, axis=0)
    prediction = model.predict(roi)
    predicted_class = np.argmax(prediction)
    predicted_label = cell_types[predicted_class]

    # Draw bounding box & label
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cell_count += 1

# Display results
cv2.putText(image, f"Total Cells: {cell_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Blood Cell Identification & Counting", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
