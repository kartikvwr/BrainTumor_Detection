import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
from tensorflow.keras import mixed_precision

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

# ===============================
# M2 PERFORMANCE SETUP
# ===============================

print("[INFO] TensorFlow version:", tf.__version__)

# 1. Mixed Precision — huge speedup on Apple Silicon
mixed_precision.set_global_policy('mixed_float16')
print("[INFO] Mixed precision: mixed_float16")

# 2. Metal GPU
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

# 3. Use ALL cores for CPU ops
num_cores = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
print(f"[INFO] Using {num_cores} CPU cores")

# ===============================
# HYPERPARAMETERS
# ===============================

DATASET_PATH  = "dataset"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 64      # Larger batch = better GPU utilization on M2
EPOCHS        = 30
LEARNING_RATE = 5e-5
AUTOTUNE      = tf.data.AUTOTUNE  # Let TF decide optimal prefetch buffer

CLASSES_OF_INTEREST = [
    'glioma_tumor',
    'meningioma_tumor',
    'pituitary_tumor'
]

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
    print(df['label'].value_counts())
    return df

# ===============================
# tf.data PIPELINE (REPLACES ImageDataGenerator)
# ===============================
# Why: ImageDataGenerator runs on CPU only.
# tf.data pipelines use prefetching + parallel loading
# so the GPU never has to wait for data. ~2-3x faster on M2.

CLASS_NAMES = sorted(CLASSES_OF_INTEREST)  # alphabetical, matches generator ordering

def parse_image(filepath, label):
    """Load, decode, resize and preprocess a single image."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

def augment(img, label):
    """Augmentation applied only during training."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    # Random rotation via map (lightweight)
    img = tf.keras.layers.RandomRotation(0.05)(img, training=True)
    img = tf.keras.layers.RandomZoom(0.1)(img, training=True)
    return img, label

def build_dataset(df, training=False):
    """
    Build a high-performance tf.data pipeline.
    - cache()       : stores dataset in RAM after first epoch (no re-reading disk)
    - prefetch()    : loads next batch while GPU processes current one
    - AUTOTUNE      : TF picks optimal parallelism automatically
    """
    label_indices = [CLASS_NAMES.index(l) for l in df['label']]
    labels_onehot = tf.keras.utils.to_categorical(label_indices, num_classes=len(CLASS_NAMES))

    ds = tf.data.Dataset.from_tensor_slices((df['filepath'].values, labels_onehot))
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)  # parallel image loading

    if training:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()          # cache in RAM — huge speedup from epoch 2 onwards
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ===============================
# BUILD MODEL
# ===============================

def build_model(num_classes):
    print("[STATUS] Loading ResNet50...")

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze early layers, fine-tune last 60
    for layer in base_model.layers[:-60]:
        layer.trainable = False
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    attention = Dense(2048, activation='sigmoid')(x)
    x = Multiply()([x, attention])

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    print("[SUCCESS] Model compiled")
    return model

# ===============================
# TRAINING PLOT
# ===============================

def plot_training(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history.history['accuracy'],     label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Validation')
    ax[0].set_title('Accuracy vs Epoch')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(history.history['loss'],     label='Train')
    ax[1].plot(history.history['val_loss'], label='Validation')
    ax[1].set_title('Loss vs Epoch')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    fig.savefig("training_plot.png", dpi=150)
    print("[SAVED] training_plot.png")

# ===============================
# EVALUATION — single predict call
# ===============================

def evaluate_model(model, val_ds, val_df):
    """
    One model.predict() call, results reused for ALL metrics.
    Previously each plot function called model.predict() separately — wasteful.
    """
    print("[STATUS] Running predictions (once for all plots)...")
    y_pred_proba = model.predict(val_ds, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    # Rebuild true labels in same order as dataset (no shuffle on val)
    y_true_idx = [CLASS_NAMES.index(l) for l in val_df['label']]
    y_true_idx = np.array(y_true_idx)

    # --- Confusion Matrix ---
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

    # --- Classification Report ---
    print("\nClassification Report:\n")
    print(classification_report(y_true_idx, y_pred, target_names=CLASS_NAMES))

    return y_pred_proba, y_pred, y_true_idx   # return for reuse

# ===============================
# GRAD-CAM
# ===============================

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

    if np.max(heatmap) == 0 or np.isnan(heatmap).all():
        print("[ERROR] Heatmap generation failed.")
        return

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

# ===============================
# PRESENTATION PLOTS
# (use pre-computed predictions — no extra model.predict calls)
# ===============================

def plot_class_distribution(df):
    counts = df['label'].value_counts()
    colors = ['#4C72B0', '#DD8452', '#55A868']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=12, fontweight='bold')

    plt.title('Class Distribution in Dataset', fontsize=15, fontweight='bold')
    plt.xlabel('Tumor Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    print("[SAVED] class_distribution.png")


def plot_sample_images(df):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Sample MRI Images per Tumor Class', fontsize=16, fontweight='bold')

    for row, label in enumerate(CLASSES_OF_INTEREST):
        samples = df[df['label'] == label].sample(5, random_state=42)
        for col, (_, row_data) in enumerate(samples.iterrows()):
            img = cv2.cvtColor(cv2.imread(row_data['filepath']), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(label.replace('_', '\n'),
                                          fontsize=11, fontweight='bold',
                                          rotation=0, labelpad=70, va='center')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    print("[SAVED] sample_images.png")


def plot_roc_curves(y_true_idx, y_pred_proba):
    """Accepts pre-computed predictions — no model call needed."""
    y_true_bin = label_binarize(y_true_idx, classes=list(range(len(CLASS_NAMES))))
    colors = ['#4C72B0', '#DD8452', '#55A868']

    plt.figure(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc(fpr,tpr):.2f})')

    plt.plot([0,1],[0,1],'k--',lw=1.5)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Per Class', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    print("[SAVED] roc_curves.png")


def plot_metrics_bar(y_true_idx, y_pred):
    """Accepts pre-computed predictions — no model call needed."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_idx, y_pred, average=None, labels=list(range(len(CLASS_NAMES)))
    )
    x = np.arange(len(CLASS_NAMES)); w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x-w, precision, w, label='Precision', color='#4C72B0', edgecolor='black')
    ax.bar(x,   recall,    w, label='Recall',    color='#DD8452', edgecolor='black')
    ax.bar(x+w, f1,        w, label='F1 Score',  color='#55A868', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall & F1 per Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=2)
    plt.tight_layout()
    plt.savefig('metrics_bar.png', dpi=150)
    print("[SAVED] metrics_bar.png")


def plot_prediction_confidence(model, val_df):
    samples = val_df.groupby('label').apply(lambda x: x.sample(1)).reset_index(drop=True)

    # Batch all sample predictions in one call
    imgs = []
    for _, row in samples.iterrows():
        img = tf.keras.preprocessing.image.load_img(row['filepath'], target_size=(224,224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        imgs.append(img_array)

    batch = tf.keras.applications.resnet50.preprocess_input(np.array(imgs))
    all_probs = model.predict(batch, verbose=0)   # single batched call

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    fig.suptitle('Model Predictions with Confidence', fontsize=15, fontweight='bold')

    for i, (_, row) in enumerate(samples.iterrows()):
        probs      = all_probs[i]
        pred_label = CLASS_NAMES[np.argmax(probs)]
        correct    = pred_label == row['label']

        axes[i,0].imshow(imgs[i].astype('uint8'))
        axes[i,0].axis('off')
        axes[i,0].set_title(f"True: {row['label']}\nPred: {pred_label}",
                             fontsize=11, fontweight='bold',
                             color='green' if correct else 'red')

        colors = ['#55A868' if c == pred_label else '#4C72B0' for c in CLASS_NAMES]
        axes[i,1].barh(CLASS_NAMES, probs, color=colors, edgecolor='black')
        axes[i,1].set_xlim(0, 1)
        axes[i,1].set_xlabel('Confidence', fontsize=11)
        axes[i,1].set_title('Class Probabilities', fontsize=11)
        for j, p in enumerate(probs):
            axes[i,1].text(p + 0.01, j, f'{p:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=150)
    print("[SAVED] prediction_confidence.png")

# ===============================
# MAIN
# ===============================

def main():
    # 1. Load data
    df = load_and_filter_data(DATASET_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # 2. Build fast tf.data pipelines
    train_ds = build_dataset(train_df, training=True)
    val_ds   = build_dataset(val_df,   training=False)

    # 3. Build & train model
    model = build_model(num_classes=len(CLASS_NAMES))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
        ]
    )

    # 4. Save model
    model.save("brain_tumor_resnet50_optimized.keras")
    print("[SAVED] brain_tumor_resnet50_optimized.keras")

    # 5. Training plot
    plot_training(history)
    best_val_acc = max(history.history['val_accuracy'])
    print(f"[BEST] Validation Accuracy: {best_val_acc*100:.2f}%")

    # 6. Evaluate ONCE — reuse results for all plots (no repeated model.predict)
    y_pred_proba, y_pred, y_true_idx = evaluate_model(model, val_ds, val_df)

    # 7. All presentation plots
    plot_class_distribution(df)
    plot_sample_images(df)
    plot_roc_curves(y_true_idx, y_pred_proba)
    plot_metrics_bar(y_true_idx, y_pred)
    plot_prediction_confidence(model, val_df)

    # 8. Grad-CAM on one sample
    sample_path = val_df['filepath'].iloc[10]
    print(f"[STATUS] Grad-CAM on: {sample_path}")
    gradcam(model, sample_path)

    print("\n[DONE] All plots saved.")


if __name__ == "__main__":
    main()





















import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
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
from tensorflow.keras import mixed_precision

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

# ===============================
# M2 PERFORMANCE SETUP
# ===============================

print("[INFO] TensorFlow version:", tf.__version__)

# 1. Mixed Precision — huge speedup on Apple Silicon
#mixed_precision.set_global_policy('mixed_float16')
#print("[INFO] Mixed precision: mixed_float16")

# 2. Metal GPU
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

# 3. Use ALL cores for CPU ops
num_cores = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
print(f"[INFO] Using {num_cores} CPU cores")

# ===============================
# HYPERPARAMETERS
# ===============================

DATASET_PATH  = "dataset"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 64      # Larger batch = better GPU utilization on M2
EPOCHS        = 30
LEARNING_RATE = 5e-5
AUTOTUNE      = tf.data.AUTOTUNE  # Let TF decide optimal prefetch buffer

CLASSES_OF_INTEREST = [
    'glioma_tumor',
    'meningioma_tumor',
    'pituitary_tumor'
]

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
    print(df['label'].value_counts())
    return df

# ===============================
# tf.data PIPELINE (REPLACES ImageDataGenerator)
# ===============================
# Why: ImageDataGenerator runs on CPU only.
# tf.data pipelines use prefetching + parallel loading
# so the GPU never has to wait for data. ~2-3x faster on M2.

CLASS_NAMES = sorted(CLASSES_OF_INTEREST)  # alphabetical, matches generator ordering

def parse_image(filepath, label):
    """Load, decode, resize and preprocess a single image."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

# Create the Keras layers OUTSIDE the function
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1)
])

def augment(img, label):
    """Augmentation applied only during training."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    
    # Apply the pre-built layers
    img = data_augmentation(img, training=True)
    return img, label

def build_dataset(df, training=False):
    """
    Build a high-performance tf.data pipeline.
    - cache()       : stores dataset in RAM after first epoch (no re-reading disk)
    - prefetch()    : loads next batch while GPU processes current one
    - AUTOTUNE      : TF picks optimal parallelism automatically
    """
    label_indices = [CLASS_NAMES.index(l) for l in df['label']]
    labels_onehot = tf.keras.utils.to_categorical(label_indices, num_classes=len(CLASS_NAMES))

    ds = tf.data.Dataset.from_tensor_slices((df['filepath'].values, labels_onehot))
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)  # parallel image loading

    if training:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()          # cache in RAM — huge speedup from epoch 2 onwards
    
    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()
    
    # ADD THIS TO PREVENT MISMATCHED LABELS IN EVALUATION:
    if not training:
        options = tf.data.Options()
        options.deterministic = True
        ds = ds.with_options(options)
        
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ===============================
# BUILD MODEL
# ===============================

def build_model(num_classes):
    print("[STATUS] Loading ResNet50...")

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze early layers, fine-tune last 60
    for layer in base_model.layers[:-60]:
        layer.trainable = False
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    attention = Dense(2048, activation='sigmoid')(x)
    x = Multiply()([x, attention])

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    print("[SUCCESS] Model compiled")
    return model

# ===============================
# TRAINING PLOT
# ===============================

def plot_training(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history.history['accuracy'],     label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Validation')
    ax[0].set_title('Accuracy vs Epoch')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(history.history['loss'],     label='Train')
    ax[1].plot(history.history['val_loss'], label='Validation')
    ax[1].set_title('Loss vs Epoch')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    fig.savefig("training_plot.png", dpi=150)
    print("[SAVED] training_plot.png")

# ===============================
# EVALUATION — single predict call
# ===============================

def evaluate_model(model, val_ds, val_df):
    """
    One model.predict() call, results reused for ALL metrics.
    Previously each plot function called model.predict() separately — wasteful.
    """
    print("[STATUS] Running predictions (once for all plots)...")
    y_pred_proba = model.predict(val_ds, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    # Rebuild true labels in same order as dataset (no shuffle on val)
    y_true_idx = [CLASS_NAMES.index(l) for l in val_df['label']]
    y_true_idx = np.array(y_true_idx)

    # --- Confusion Matrix ---
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

    # --- Classification Report ---
    print("\nClassification Report:\n")
    print(classification_report(y_true_idx, y_pred, target_names=CLASS_NAMES))

    return y_pred_proba, y_pred, y_true_idx   # return for reuse

# ===============================
# GRAD-CAM
# ===============================

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

    if np.max(heatmap) == 0 or np.isnan(heatmap).all():
        print("[ERROR] Heatmap generation failed.")
        return

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
    plt.show(block=True) # This makes the window pop up
    plt.close()          # This frees up your Mac's memory after you close the window

# ===============================
# PRESENTATION PLOTS
# (use pre-computed predictions — no extra model.predict calls)
# ===============================

def plot_class_distribution(df):
    counts = df['label'].value_counts()
    colors = ['#4C72B0', '#DD8452', '#55A868']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=12, fontweight='bold')

    plt.title('Class Distribution in Dataset', fontsize=15, fontweight='bold')
    plt.xlabel('Tumor Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    print("[SAVED] class_distribution.png")
    plt.show(block=True) # This makes the window pop up
    plt.close()          # This frees up your Mac's memory after you close the window


def plot_sample_images(df):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Sample MRI Images per Tumor Class', fontsize=16, fontweight='bold')

    for row, label in enumerate(CLASSES_OF_INTEREST):
        samples = df[df['label'] == label].sample(5, random_state=42)
        for col, (_, row_data) in enumerate(samples.iterrows()):
            img = cv2.cvtColor(cv2.imread(row_data['filepath']), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(label.replace('_', '\n'),
                                          fontsize=11, fontweight='bold',
                                          rotation=0, labelpad=70, va='center')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    print("[SAVED] sample_images.png")
    plt.show(block=True) # This makes the window pop up
    plt.close()          # This frees up your Mac's memory after you close the window


def plot_roc_curves(y_true_idx, y_pred_proba):
    """Accepts pre-computed predictions — no model call needed."""
    y_true_bin = label_binarize(y_true_idx, classes=list(range(len(CLASS_NAMES))))
    colors = ['#4C72B0', '#DD8452', '#55A868']

    plt.figure(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc(fpr,tpr):.2f})')

    plt.plot([0,1],[0,1],'k--',lw=1.5)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Per Class', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    print("[SAVED] roc_curves.png")
    plt.show(block=True) # This makes the window pop up
    plt.close()          # This frees up your Mac's memory after you close the window


def plot_metrics_bar(y_true_idx, y_pred):
    """Accepts pre-computed predictions — no model call needed."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_idx, y_pred, average=None, labels=list(range(len(CLASS_NAMES)))
    )
    x = np.arange(len(CLASS_NAMES)); w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x-w, precision, w, label='Precision', color='#4C72B0', edgecolor='black')
    ax.bar(x,   recall,    w, label='Recall',    color='#DD8452', edgecolor='black')
    ax.bar(x+w, f1,        w, label='F1 Score',  color='#55A868', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall & F1 per Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=2)
    plt.tight_layout()
    plt.savefig('metrics_bar.png', dpi=150)
    print("[SAVED] metrics_bar.png")
    plt.show(block=True) # This makes the window pop up
    plt.close()          # This frees up your Mac's memory after you close the window


def plot_prediction_confidence(model, val_df):
    samples = val_df.groupby('label').apply(lambda x: x.sample(1)).reset_index(drop=True)

    # Batch all sample predictions in one call
    imgs = []
    for _, row in samples.iterrows():
        img = tf.keras.preprocessing.image.load_img(row['filepath'], target_size=(224,224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        imgs.append(img_array)

    batch = tf.keras.applications.resnet50.preprocess_input(np.array(imgs))
    all_probs = model.predict(batch, verbose=0)   # single batched call

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    fig.suptitle('Model Predictions with Confidence', fontsize=15, fontweight='bold')

    for i, (_, row) in enumerate(samples.iterrows()):
        probs      = all_probs[i]
        pred_label = CLASS_NAMES[np.argmax(probs)]
        correct    = pred_label == row['label']

        axes[i,0].imshow(imgs[i].astype('uint8'))
        axes[i,0].axis('off')
        axes[i,0].set_title(f"True: {row['label']}\nPred: {pred_label}",
                             fontsize=11, fontweight='bold',
                             color='green' if correct else 'red')

        colors = ['#55A868' if c == pred_label else '#4C72B0' for c in CLASS_NAMES]
        axes[i,1].barh(CLASS_NAMES, probs, color=colors, edgecolor='black')
        axes[i,1].set_xlim(0, 1)
        axes[i,1].set_xlabel('Confidence', fontsize=11)
        axes[i,1].set_title('Class Probabilities', fontsize=11)
        for j, p in enumerate(probs):
            axes[i,1].text(p + 0.01, j, f'{p:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=150)
    print("[SAVED] prediction_confidence.png")
    plt.show(block=True) # This makes the window pop up
    plt.close()          # This frees up your Mac's memory after you close the window

# ===============================
# MAIN
# ===============================

def main():
    # 1. Load data
    df = load_and_filter_data(DATASET_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # 2. Build fast tf.data pipelines
    train_ds = build_dataset(train_df, training=True)
    val_ds   = build_dataset(val_df,   training=False)

    # 3. Build & train model
    model = build_model(num_classes=len(CLASS_NAMES))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
        ]
    )

    # 4. Save model
    model.save("brain_tumor_resnet50_optimized.keras")
    print("[SAVED] brain_tumor_resnet50_optimized.keras")

    # 5. Training plot
    plot_training(history)
    best_val_acc = max(history.history['val_accuracy'])
    print(f"[BEST] Validation Accuracy: {best_val_acc*100:.2f}%")

    # 6. Evaluate ONCE — reuse results for all plots (no repeated model.predict)
    y_pred_proba, y_pred, y_true_idx = evaluate_model(model, val_ds, val_df)

    # 7. All presentation plots
    plot_class_distribution(df)
    plot_sample_images(df)
    plot_roc_curves(y_true_idx, y_pred_proba)
    plot_metrics_bar(y_true_idx, y_pred)
    plot_prediction_confidence(model, val_df)

    # 8. Grad-CAM on one sample
    sample_path = val_df['filepath'].iloc[10]
    print(f"[STATUS] Grad-CAM on: {sample_path}")
    gradcam(model, sample_path)

    print("\n[DONE] All plots saved.")

    # 3. Build & train model OR Load existing
    # 3. Build & train model OR Load existing
    
    # CHANGE THIS LINE to match your actual saved file!
    MODEL_NAME = "brain_tumor_resnet50_phase2.keras"  
    
    if os.path.exists(MODEL_NAME):
        print(f"[STATUS] Found existing model '{MODEL_NAME}'. Skipping training!")
    
    if os.path.exists(MODEL_NAME):
        print(f"[STATUS] Found existing model '{MODEL_NAME}'. Skipping training!")
        model = tf.keras.models.load_model(MODEL_NAME)
    else:
        print("[STATUS] No saved model found. Starting training...")
        model = build_model(num_classes=len(CLASS_NAMES))
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
            ]
        )
        model.save(MODEL_NAME)
        print(f"[SAVED] {MODEL_NAME}")
        
        # Only plot training history if we actually trained!
        plot_training(history)


if __name__ == "__main__":
    main()



















import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
# matplotlib.use('Agg')  # Commented out so windows can pop up
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
from tensorflow.keras import mixed_precision

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

# ===============================
# M2 PERFORMANCE SETUP
# ===============================

print("[INFO] TensorFlow version:", tf.__version__)

# 1. Mixed Precision — DISABLED for M2 (causes crashes/slowdowns without Nvidia Tensor Cores)
# mixed_precision.set_global_policy('mixed_float16')
# print("[INFO] Mixed precision: mixed_float16")

# 2. Metal GPU
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

# 3. Use ALL cores for CPU ops
num_cores = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
print(f"[INFO] Using {num_cores} CPU cores")

# ===============================
# HYPERPARAMETERS
# ===============================

DATASET_PATH  = "dataset"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 64      
EPOCHS        = 30
LEARNING_RATE = 5e-5
AUTOTUNE      = tf.data.AUTOTUNE  

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
# tf.data PIPELINE (FIXED)
# ===============================

# CREATE LAYERS OUTSIDE THE FUNCTION SO IT DOESN'T CRASH
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1)
])

def parse_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = data_augmentation(img, training=True)
    return img, label

def build_dataset(df, training=False):
    label_indices = [CLASS_NAMES.index(l) for l in df['label']]
    labels_onehot = tf.keras.utils.to_categorical(label_indices, num_classes=len(CLASS_NAMES))

    ds = tf.data.Dataset.from_tensor_slices((df['filepath'].values, labels_onehot))
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)  

    if training:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()          
    
    # Ensure val data doesn't get shuffled accidentally
    if not training:
        options = tf.data.Options()
        options.deterministic = True
        ds = ds.with_options(options)
        
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ===============================
# BUILD MODEL
# ===============================

def build_model(num_classes):
    print("[STATUS] Loading ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers[:-60]:
        layer.trainable = False
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    attention = Dense(2048, activation='sigmoid')(x)
    x = Multiply()([x, attention])

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

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

def evaluate_model(model, val_ds, val_df):
    print("[STATUS] Running predictions (once for all plots)...")
    y_pred_proba = model.predict(val_ds, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    y_true_idx = [CLASS_NAMES.index(l) for l in val_df['label']]
    y_true_idx = np.array(y_true_idx)

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
    
    train_ds = build_dataset(train_df, training=True)
    val_ds   = build_dataset(val_df,   training=False)

    MODEL_NAME = "brain_tumor_resnet50_phase2.keras"
    
    if os.path.exists(MODEL_NAME):
        print(f"\n[STATUS] Found existing model '{MODEL_NAME}'. Skipping 30 epochs!")
        model = tf.keras.models.load_model(MODEL_NAME, compile=False)
    else:
        print("\n[STATUS] No saved model found. Starting training...")
        model = build_model(num_classes=len(CLASS_NAMES))
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
            ]
        )
        model.save(MODEL_NAME)
        print(f"[SAVED] {MODEL_NAME}")
        plot_training(history)

    # Generate all plots using the loaded model
    y_pred_proba, y_pred, y_true_idx = evaluate_model(model, val_ds, val_df)
    
    plot_class_distribution(df)
    plot_roc_curves(y_true_idx, y_pred_proba)
    
    sample_path = val_df['filepath'].iloc[10]
    print(f"\n[STATUS] Grad-CAM on: {sample_path}")
    gradcam(model, sample_path)

    print("\n[DONE] All plots generated and saved!")

if __name__ == "__main__":
    main()