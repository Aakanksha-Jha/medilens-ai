"""
MediLens AI — Step 3: Computer Vision (CNN + Transfer Learning)

DATA NOTE: The real target dataset here is "Chest X-Ray Images (Pneumonia)"
(Kaggle / NIH). This sandbox cannot reach Kaggle or general file hosts
(network is locked to package registries only), so this script:
  1) Defines the exact architecture you'd use in production.
  2) Trains it end-to-end on a small SYNTHETIC image set (simple procedurally
     generated "lung-like" blob textures with two classes) purely so every
     line below actually executes and produces real numbers.
  3) Is 100% ready to point at the real dataset — swap `load_synthetic_xrays()`
     for `image_dataset_from_directory("data/chest_xray/train", ...)` and
     nothing else changes.

Two models are built:
  A) A small CNN trained from scratch.
  B) A transfer-learning model on top of frozen MobileNetV2 (ImageNet weights).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = 96
OUT = "/home/claude/MediLens_AI/reports"

# ---------------------------------------------------------------------------
# Synthetic "chest x-ray" style data generator (stand-in for real dataset)
# Class 0 = "normal": smooth radial gradient blob
# Class 1 = "pneumonia": blob + scattered high-frequency opacity patches
# ---------------------------------------------------------------------------
def make_image(label, rng):
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx, cy = IMG_SIZE / 2 + rng.uniform(-5, 5), IMG_SIZE / 2 + rng.uniform(-5, 5)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    base = np.clip(1 - r / (IMG_SIZE * 0.6), 0, 1)
    img = base * rng.uniform(0.6, 1.0)
    if label == 1:
        n_patches = rng.integers(4, 10)
        for _ in range(n_patches):
            px, py = rng.integers(10, IMG_SIZE - 10, size=2)
            patch_r = rng.integers(3, 9)
            yy2, xx2 = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
            patch_mask = (xx2 - px) ** 2 + (yy2 - py) ** 2 < patch_r ** 2
            img[patch_mask] += rng.uniform(0.3, 0.6)
    img += rng.normal(0, 0.05, size=img.shape)
    img = np.clip(img, 0, 1)
    return np.stack([img] * 3, axis=-1)  # fake 3-channel to feed MobileNet

def make_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in [0, 1]:
        for _ in range(n_per_class):
            X.append(make_image(label, rng))
            y.append(label)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

X_train, y_train = make_dataset(150, seed=1)
X_val, y_val = make_dataset(30, seed=2)
X_test, y_test = make_dataset(40, seed=3)

# ---------------------------------------------------------------------------
# Model A: Small CNN from scratch
# ---------------------------------------------------------------------------
def build_small_cnn():
    m = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

cnn = build_small_cnn()
hist_cnn = cnn.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=8, batch_size=16, verbose=0)
cnn_test_loss, cnn_test_acc = cnn.evaluate(X_test, y_test, verbose=0)

# ---------------------------------------------------------------------------
# Model B: Transfer Learning — MobileNetV2 backbone (frozen) + custom head
# ---------------------------------------------------------------------------

# NOTE: This sandbox's network is locked to package registries only (pip/npm/GitHub),
# so it cannot reach storage.googleapis.com to download ImageNet pretrained weights.
# In a normal environment (Colab/Kaggle/your own machine with internet),
# weights="imagenet" downloads them automatically and this becomes true transfer
# learning. We fall back to weights=None here ONLY so this script still runs
# end-to-end in this sandbox; the architecture, freezing pattern, and preprocessing
# pipeline are identical to production transfer learning.
try:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    USED_PRETRAINED = True
except Exception as e:
    print(f"[warning] Could not fetch ImageNet weights ({e}); falling back to random init.")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None
    )
    USED_PRETRAINED = False
base.trainable = False  # freeze backbone (this is what makes it "transfer learning" once pretrained weights are loaded)

tl_model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
tl_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
hist_tl = tl_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=8, batch_size=16, verbose=0)
tl_test_loss, tl_test_acc = tl_model.evaluate(X_test, y_test, verbose=0)

# Save training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_cnn.history["accuracy"], label="CNN train")
plt.plot(hist_cnn.history["val_accuracy"], label="CNN val")
plt.plot(hist_tl.history["accuracy"], label="MobileNetV2 train")
plt.plot(hist_tl.history["val_accuracy"], label="MobileNetV2 val")
plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(hist_cnn.history["loss"], label="CNN train")
plt.plot(hist_cnn.history["val_loss"], label="CNN val")
plt.plot(hist_tl.history["loss"], label="MobileNetV2 train")
plt.plot(hist_tl.history["val_loss"], label="MobileNetV2 val")
plt.title("Loss"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/cnn_training_curves.png", dpi=140)
plt.close()

cnn_results = {
    "small_cnn_scratch": {"test_loss": float(cnn_test_loss), "test_accuracy": float(cnn_test_acc)},
    "mobilenetv2_transfer": {"test_loss": float(tl_test_loss), "test_accuracy": float(tl_test_acc)},
    "note": "Trained on synthetic procedurally-generated images as a stand-in for the real Chest X-Ray Pneumonia dataset (not downloadable in this offline sandbox, which only allows package-registry network egress). Architecture is production-ready for the real dataset; with normal internet access weights='imagenet' loads automatically for genuine transfer learning."
}
with open(f"{OUT}/cnn_metrics.json", "w") as f:
    json.dump(cnn_results, f, indent=2)

print(json.dumps(cnn_results, indent=2))

# Save model architectures as text for the report
with open(f"{OUT}/cnn_architecture_summary.txt", "w") as f:
    cnn.summary(print_fn=lambda x: f.write(x + "\n"))
    f.write("\n\n--- MobileNetV2 Transfer Model ---\n\n")
    tl_model.summary(print_fn=lambda x: f.write(x + "\n"))

print("Done.")
