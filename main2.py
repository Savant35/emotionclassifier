
#!/usr/bin/env python3
import os, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

# 1) Mixed-precision for Apple M4
mixed_precision.set_global_policy('mixed_float16')

# 2) Paths & hyperparams
train_dir  = 'data/train'
val_dir    = 'data/test'
batch_size = 128
img_size   = (224, 224)

# 3) Count images
num_train = sum(len(files) for _,_,files in os.walk(train_dir))
num_val   = sum(len(files) for _,_,files in os.walk(val_dir))

# 4) Build tf.data datasets with integer labels
raw_train_ds = image_dataset_from_directory(
    train_dir,
    label_mode='int',     # integer labels
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)
raw_val_ds = image_dataset_from_directory(
    val_dir,
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
classes = raw_train_ds.class_names

print("── DATASET ─────────────────────────")
print(f" Train images: {num_train},  Val images: {num_val}")
print(f" Batch size:   {batch_size},  Image size: {img_size}")
print(f" Classes:      {classes}\n")

# 5) Gather all labels (batch-wise) and compute class weights
all_labels = []
for _imgs, labels in raw_train_ds:
    all_labels.append(labels.numpy())
all_labels = np.concatenate(all_labels, axis=0)

print(" Class distribution:")
for idx, cls in enumerate(classes):
    print(f"  {cls:10s}: {np.sum(all_labels == idx)}")

ids = np.unique(all_labels)
cw  = compute_class_weight('balanced', classes=ids, y=all_labels)
class_weight = {int(k): float(v) for k, v in zip(ids, cw)}
print("\n Class weights:", class_weight, "\n")

# 6) Preprocess & augment pipeline
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_batch(images, labels):
    return preprocess_input(images), labels

def augment(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, 0.2)
    images = tf.image.random_contrast(images, 0.8, 1.2)
    return images, labels

train_ds = (
    raw_train_ds
    .map(preprocess_batch, num_parallel_calls=AUTOTUNE)
    .cache()                    # ~2–3 GB in RAM
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    raw_val_ds
    .map(preprocess_batch, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# 7) Build EfficientNetB0 head
base = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=img_size + (3,)
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(len(classes), activation='softmax', dtype='float32')(x)
model = Model(base.input, out)

print("── MODEL ARCH ───────────────────────")
model.summary()
print()

# 8) Phase 1: train head only
for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # sparse, no smoothing
    metrics=['accuracy']
)

ckpt_head = ModelCheckpoint(
    'best_head.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
callbacks = [
    ckpt_head,
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# 9) Phase 2: fine-tune entire backbone
for layer in base.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

ckpt_ft = ModelCheckpoint(
    'best_finetuned.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
# reuse LR & earlystop from before
callbacks_ft = [ckpt_ft] + callbacks[1:]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks_ft,
    verbose=1
)

# 10) Plot metrics
acc      = history1.history['accuracy']  + history2.history['accuracy']
val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss     = history1.history['loss']      + history2.history['loss']
val_loss = history1.history['val_loss']   + history2.history['val_loss']

plt.figure()
plt.plot(acc,     label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid()

plt.figure()
plt.plot(loss,     label='Train Loss')
plt.plot(val_loss,  label='Val Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid()

plt.show()

# 11) Final evaluation & save
loss, acc = model.evaluate(val_ds, verbose=1)
print(f"\nFinal val loss: {loss:.4f},  val acc: {acc:.4f}")

model.save('emotion_model4.keras')
print("→ Saved final model to emotion_model4.keras")

