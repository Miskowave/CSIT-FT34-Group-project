#Airplane Recognition Comparison


from google.colab import drive
import os, zipfile, random, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score, confusion_matrix, classification_report

drive.mount('/content/drive')

# DATASET SETUP
ZIP_PATH = "/content/drive/MyDrive/Aircraft Image Dataset.zip"
EXTRACT_PATH = "/content/dataset"
os.makedirs(EXTRACT_PATH, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_PATH)

# Auto-detect class folders
def find_class_root(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            cand = os.path.join(root, d)
            subs = [x for x in os.listdir(cand) if os.path.isdir(os.path.join(cand,x))]
            if len(subs) >= 2: return cand
    return None

MAIN_DIR = find_class_root(EXTRACT_PATH)
classes = sorted([d for d in os.listdir(MAIN_DIR) if os.path.isdir(os.path.join(MAIN_DIR,d))])
num_classes = len(classes)

# STRATIFIED SPLIT (70/20/10)
paths, labels = [], []
for cls in classes:
    cls_dir = os.path.join(MAIN_DIR, cls)
    for f in os.listdir(cls_dir):
        if f.lower().endswith((".jpg",".png",".jpeg")):
            paths.append(os.path.join(cls_dir, f))
            labels.append(cls)

df = pd.DataFrame({"file_path": paths, "label": labels})
label_to_idx = {c:i for i,c in enumerate(classes)}

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=7, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=7, stratify=temp_df["label"])

# TF.DATA PIPELINE
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def make_ds(dataframe, training=False):
    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        return img, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((dataframe["file_path"].values, dataframe["label"].map(label_to_idx).values))
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])
        ds = ds.map(lambda x,y: (aug(x, training=True), y))
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_df, training=True)
val_ds = make_ds(val_df)
test_ds = make_ds(test_df)

# MODEL BUILDERS
def build_simple_cnn():
    model = tf.keras.Sequential([
        layers.Input(shape=(224,224,3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def build_efficientnet():
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base.trainable = False
    inputs = layers.Input(shape=(224,224,3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs=x), base

# GRAD-CAM
def get_gradcam(model, img_array, class_idx):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer("efficientnetb0").output, model.output])
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()
    return heatmap

# TRAINING & EVALUATION HELPER
def train_and_eval(model, name, fine_tune_base=None):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

    if fine_tune_base:
        fine_tune_base.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

    probs = model.predict(test_ds)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])

    print(f"\n{name} Results:")
    print(classification_report(y_true, y_pred, target_names=classes))
    return y_true, y_pred

# EXECUTION
cnn_true, cnn_pred = train_and_eval(build_simple_cnn(), "Simple CNN")
eff_model, eff_base = build_efficientnet()
eff_true, eff_pred = train_and_eval(eff_model, "EfficientNetB0", fine_tune_base=eff_base)

# Confusion Matrix for Final Model
sns.heatmap(confusion_matrix(eff_true, eff_pred), annot=True, xticklabels=classes, yticklabels=classes)
plt.show()
