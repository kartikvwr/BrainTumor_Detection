import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

#HYPERPARAMETERS & CONFIGURATION
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)    # Standard input size for ResNet
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4     # 0.0001 - Keeping it low for fine-tuning
CLASSES_OF_INTEREST = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

def load_and_filter_data(data_dir):
    """
    Scans the dataset directory and creates a DataFrame.
    Filters out the 'no_tumor' class as per Phase 1 requirements.
    """
    print(f"[STATUS] Scanning directory: {data_dir}...")
    
    filepaths = []
    labels = []

    # Walking through the directory structure
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract label from folder name
                folder_name = os.path.basename(root)
                
                # We only want the specific tumor types for this classifier
                if folder_name in CLASSES_OF_INTEREST:
                    filepaths.append(os.path.join(root, file))
                    labels.append(folder_name)

    # Sanity check
    if not filepaths:
        print("[ERROR] No images found. Please check the dataset path.")
        return None

    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })
    
    print(f"[SUCCESS] Data loaded. Total images: {len(df)}")
    print(f"Class Distribution:\n{df['label'].value_counts()}")
    return df

def build_model(num_classes):
    """
    Constructs the model using ResNet50 as the backbone.
    Freezes the convolutional base and adds a custom classification head.
    """
    print("[STATUS] Downloading/Loading ResNet50 backbone...")
    
    # Load ResNet50 pre-trained on ImageNet, without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model to prevent destroying learned features during initial training
    base_model.trainable = False
    
    # Adding our custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)        # Pooling to reduce parameters
    x = Dense(128, activation='relu')(x)   # Dense layer for feature learning
    x = Dropout(0.5)(x)                    # Dropout to reduce overfitting
    output = Dense(num_classes, activation='softmax')(x) 
    
    model = Model(inputs=base_model.input, outputs=output)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("[SUCCESS] Model compiled successfully.")
    return model

def main():
    # 1. Prepare Data
    df = load_and_filter_data(DATASET_PATH)
    if df is None: return

    # Stratified split to keep class ratios consistent
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df['label'])
    
    print(f"[INFO] Training set size: {len(train_df)}")
    print(f"[INFO] Validation set size: {len(val_df)}")

    # 2. Data Augmentation & Generators
    # Using ResNet50's native preprocessing function is crucial for performance
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    # 3. Model Training
    model = build_model(num_classes=3)
    
    print("[STATUS] Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )
    
    
    # 4. Save the Model
    save_name = "brain_tumor_resnet50_phase1.h5"
    model.save(save_name)
    print(f"[FINISHED] Model saved to {save_name}")

if __name__ == "__main__":
    main()