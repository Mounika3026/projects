###    MOBILENETV2 PRETRAINED MODEL
import pandas as pd
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load annotations CSV
annotations_df = pd.read_csv('annotations.csv')

# Specify the root directory where images are located
image_root_directory = '/path/to/your/image/root/directory/'

# Load and preprocess images
images = []
labels = []

for index, row in annotations_df.iterrows():
    image_filename = row['filename']
    class_label = row['class']
    image_path = os.path.join(image_root_directory, class_label, image_filename)
    
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to match pre-trained model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    images.append(image)
    labels.append(class_label)

images = np.array(images)
labels = np.array(labels)

# Split data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load pre-trained model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
num_classes = len(set(labels))
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)
