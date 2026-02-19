# ============================================================
# CSCI218 – Airplane Model Recognition (Google Colab Full Code)
# - Mount Drive
# - Unzip dataset (archive.zip) to Colab local
# - Auto-find "crop" folder
# - Build dataframe (file_path, label)
# - Train/Val/Test split (70/21/9)
# - Generators with fixed class mapping (prevents shape mismatch)
# - CNN training + learning curves
# - Test evaluation + classification report + confusion matrix (top20)
# - Save outputs and model
# ============================================================

# -------------------------
# 0) Mount Drive
# -------------------------
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

OUT_DIR = "/content/output_figs"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# 1) Unzip dataset
# -------------------------
zip_path = "/content/drive/MyDrive/archive.zip"  # <-- change if your zip name differs
extract_path = "/content/dataset"

os.makedirs(extract_path, exist_ok=True)

print("\nUnzipping dataset...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_path)

print("Extracted to:", extract_path)
print("Top-level items:", os.listdir(extract_path)[:20])


# -------------------------
# 2) Auto-find "crop" folder
# -------------------------
main_dir = None
for root, dirs, files in os.walk(extract_path):
    if "crop" in dirs:
        main_dir = os.path.join(root, "crop")
        break

if main_dir is None:
    raise FileNotFoundError("❌ Could not find a folder named 'crop' in the extracted dataset.")

print("✅ Found crop folder at:", main_dir)


# -------------------------
# 3) Build dataframe
# -------------------------
img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

paths, labels = [], []
class_folders = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])

for cls in class_folders:
    cls_dir = os.path.join(main_dir, cls)
    for f in os.listdir(cls_dir):
        if f.lower().endswith(img_exts):
            paths.append(os.path.join(cls_dir, f))
            labels.append(cls)

df = pd.DataFrame({"file_path": paths, "label": labels})

print("\nDataset summary:")
print("Total images:", len(df))
print("Total classes:", df["label"].nunique())
print(df.head())

# Class distribution quick stats + plot top-20 (optional but useful for report)
counts = df["label"].value_counts()
print("\nClass count min/median/max:", int(counts.min()), int(counts.median()), int(counts.max()))

plt.figure(figsize=(10,5))
counts.head(20).plot(kind="bar")
plt.title("Top 20 Classes by Image Count")
plt.xlabel("Class")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dataset_top20_classes.png"), dpi=200)
plt.show()


# -------------------------
# 4) Split train/val/test (70/21/9)
# -------------------------
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=7, shuffle=True)
val_df, test_df  = train_test_split(tmp_df, test_size=0.30, random_state=7, shuffle=True)

print("\nSplit sizes:")
print("Train:", len(train_df), " Val:", len(val_df), " Test:", len(test_df))


# -------------------------
# 5) Generators (fixed class mapping prevents shape mismatch)
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

classes = sorted(df["label"].unique())  # fixed mapping across train/val/test

train_datagen = ImageDataGenerator(rescale=1./255)  # you can add augmentation later
valtest_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="file_path", y_col="label",
    target_size=IMG_SIZE, color_mode="rgb",
    class_mode="categorical", batch_size=BATCH_SIZE,
    classes=classes, shuffle=True
)

val_gen = valtest_datagen.flow_from_dataframe(
    val_df,
    x_col="file_path", y_col="label",
    target_size=IMG_SIZE, color_mode="rgb",
    class_mode="categorical", batch_size=BATCH_SIZE,
    classes=classes, shuffle=False
)

test_gen = valtest_datagen.flow_from_dataframe(
    test_df,
    x_col="file_path", y_col="label",
    target_size=IMG_SIZE, color_mode="rgb",
    class_mode="categorical", batch_size=BATCH_SIZE,
    classes=classes, shuffle=False
)

# ✅ FIX for your error: use class_indices instead of num_classes
print("\nGenerator classes:",
      len(train_gen.class_indices), len(val_gen.class_indices), len(test_gen.class_indices))

num_classes = len(train_gen.class_indices)
print("✅ num_classes =", num_classes)


# -------------------------
# 6) Build CNN model
# -------------------------
model = Sequential([
    Input(shape=(224,224,3)),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

LR = 0.001
model.compile(
    optimizer=Adamax(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Quick shape check to avoid mismatch
x_batch, y_batch = next(train_gen)
print("\nBatch label shape:", y_batch.shape)
print("Model output shape:", model.output_shape)
# Expect: y_batch (64, num_classes) and model output (None, num_classes)


# -------------------------
# 7) Train (with callbacks)
# -------------------------
EPOCHS = 10

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ModelCheckpoint("/content/best_model.keras", monitor="val_loss", save_best_only=True, verbose=1)
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=2
)


# -------------------------
# 8) Plot learning curves
# -------------------------
def plot_learning_curves(hist):
    tr_acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    tr_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]

    best_loss_epoch = int(np.argmin(val_loss)) + 1
    best_acc_epoch  = int(np.argmax(val_acc)) + 1
    epochs = np.arange(1, len(tr_acc) + 1)

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, tr_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.scatter(best_loss_epoch, val_loss[best_loss_epoch-1], s=120,
                label=f"Best val loss epoch={best_loss_epoch}")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, tr_acc, label="Train acc")
    plt.plot(epochs, val_acc, label="Val acc")
    plt.scatter(best_acc_epoch, val_acc[best_acc_epoch-1], s=120,
                label=f"Best val acc epoch={best_acc_epoch}")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_curves.png"), dpi=200)
    plt.show()

plot_learning_curves(history)


# -------------------------
# 9) Test evaluation + metrics
# -------------------------
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print("\n✅ TEST RESULTS")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Predict probabilities
test_gen.reset()
probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(probs, axis=1)
y_true = test_gen.classes

# Top-5 accuracy (useful for many classes)
top5 = top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(num_classes))
print(f"Top-5 accuracy: {top5:.4f}")

# Classification report (macro/weighted averages)
report_text = classification_report(y_true, y_pred, target_names=classes, digits=4)
print("\nClassification Report:")
print(report_text)

with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report_text)
print("Saved:", os.path.join(OUT_DIR, "classification_report.txt"))

# Confusion matrix (plot top-20 only)
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix shape:", cm.shape)

show_n = min(20, num_classes)
plt.figure(figsize=(8,7))
plt.imshow(cm[:show_n, :show_n])
plt.title(f"Confusion Matrix (Top {show_n} classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_top20.png"), dpi=200)
plt.show()


# -------------------------
# 10) Save model
# -------------------------
save_model_path = "/content/drive/MyDrive/aircraft_cnn_model.h5"
model.save(save_model_path)
print("\n✅ Model saved to:", save_model_path)

print("\n✅ Figures saved in:", OUT_DIR)
print("Files:", os.listdir(OUT_DIR))
