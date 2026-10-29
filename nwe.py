import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

# ===============================
# M2 PERFORMANCE SETUP
# ===============================

print("[INFO] TensorFlow version:", tf.__version__)

# Metal GPU Setup (Safest configuration)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Apple GPU enabled. Found {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)
else:
    print("[WARNING] No GPU found. Install tensorflow-metal.")

# ===============================
# HYPERPARAMETERS
# ===============================

DATASET_PATH  = "dataset"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 64      
EPOCHS        = 40      
LEARNING_RATE = 2e-5    

CLASSES_OF_INTEREST = [
    'glioma_tumor',
    'meningioma_tumor',
    'pituitary_tumor'
]
CLASS_NAMES = sorted(CLASSES_OF_INTEREST)

# ===============================
# LOAD DATASET
# ===============================

def load_and_filter_data(data_dir):
    print(f"[STATUS] Scanning: {data_dir}")
    filepaths, labels = [], []

    for root, _, files in os.walk(data_dir):
        folder_name = os.path.basename(root)
        if folder_name in CLASSES_OF_INTEREST:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(root, file))
                    labels.append(folder_name)

    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    print("[SUCCESS] Dataset loaded:", len(df), "images")
    return df

# ===============================
# BUILD HIGH-ACCURACY MODEL
# ===============================

def build_model(num_classes):
    print("[STATUS] Loading ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Deep Fine-Tuning: 100 layers unfrozen for maximum feature extraction
    for layer in base_model.layers[:-100]:
        layer.trainable = False
    for layer in base_model.layers[-100:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    attention = Dense(2048, activation='sigmoid')(x)
    x = Multiply()([x, attention])

    # L2 Regularizers and higher dropout to force robust learning
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

# ===============================
# PLOTTING FUNCTIONS
# ===============================

def plot_training(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'],     label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Validation')
    ax[0].set_title('Accuracy vs Epoch')
    ax[0].legend()

    ax[1].plot(history.history['loss'],     label='Train')
    ax[1].plot(history.history['val_loss'], label='Validation')
    ax[1].set_title('Loss vs Epoch')
    ax[1].legend()

    plt.tight_layout()
    fig.savefig("training_plot.png", dpi=150)
    print("[SAVED] training_plot.png")
    plt.show(block=True)
    plt.close()

def evaluate_model(model, val_generator):
    print("[STATUS] Running predictions (once for all plots)...")
    val_generator.reset()
    y_pred_proba = model.predict(val_generator, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    y_true_idx   = val_generator.classes

    cm = confusion_matrix(y_true_idx, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[SAVED] confusion_matrix.png")
    plt.show(block=True)
    plt.close()

    print("\nClassification Report:\n")
    print(classification_report(y_true_idx, y_pred, target_names=CLASS_NAMES))
    return y_pred_proba, y_pred, y_true_idx   

def gradcam(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer("conv5_block3_out").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_channel = preds[:, tf.argmax(preds[0])]

    grads       = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output  = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = np.float32(heatmap.numpy())

    orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    heatmap  = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap  = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay  = heatmap * 0.4 + orig_img

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(orig_img);            plt.title("Original"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(overlay.astype('uint8')); plt.title("Grad-CAM"); plt.axis('off')
    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=150)
    print("[SAVED] gradcam_output.png")
    plt.show(block=True)
    plt.close()

def plot_class_distribution(df):
    counts = df['label'].value_counts()
    colors = ['#4C72B0', '#DD8452', '#55A868']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Dataset', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    print("[SAVED] class_distribution.png")
    plt.show(block=True)
    plt.close()

def plot_roc_curves(y_true_idx, y_pred_proba):
    y_true_bin = label_binarize(y_true_idx, classes=list(range(len(CLASS_NAMES))))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    plt.figure(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'k--',lw=1.5)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.legend(loc='lower right', fontsize=11)
    plt.title('ROC Curves — Per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    print("[SAVED] roc_curves.png")
    plt.show(block=True)
    plt.close()

# ===============================
# MAIN
# ===============================

def main():
    df = load_and_filter_data(DATASET_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # Reverting to the safe CPU-based ImageDataGenerator to bypass Metal crashes
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label', target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='label', target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    MODEL_NAME = "brain_tumor_resnet50_v3.keras"
    
    if os.path.exists(MODEL_NAME):
        print(f"\n[STATUS] Found existing model '{MODEL_NAME}'. Skipping training!")
        model = tf.keras.models.load_model(MODEL_NAME, compile=False)
    else:
        print("\n[STATUS] No saved model found. Starting high-accuracy training...")
        model = build_model(num_classes=len(CLASS_NAMES))
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
            ]
        )
        model.save(MODEL_NAME)
        print(f"[SAVED] {MODEL_NAME}")
        plot_training(history)

    # Generate all plots using the loaded model
    y_pred_proba, y_pred, y_true_idx = evaluate_model(model, val_generator)
    
    plot_class_distribution(df)
    plot_roc_curves(y_true_idx, y_pred_proba)
    
    sample_path = val_df['filepath'].iloc[10]
    print(f"\n[STATUS] Grad-CAM on: {sample_path}")
    gradcam(model, sample_path)

    print("\n[DONE] All plots generated and saved!")

if __name__ == "__main__":
    main()