"""
NOTES (CNN on CIFAR-10 with Keras)

DATA
- 60k color images, 32x32 pixels, 3 channels (RGB), 10 classes (airplane, automobile, bird, cat, deer,
  dog, frog, horse, ship, truck); 50k train / 10k test. Chance level = 10%.
- Each image is 32*32*3 = 3072 numbers, 0..255. Harder than MNIST: real objects, color, clutter.
- Labels come as shape (N, 1), not (N,), so we .ravel() them for sklearn's metrics.
- Scaled to 0..1 with / 255.0 (same reason as MNIST: stable gradients). .astype("float32") first because
  / 255.0 on integers gives float64, which is 2x the memory and slower on CPU.

MODEL
- Conv2D(32, 3, strides=2, padding="same"): 32 kernels of 3x3. strides=2 moves the kernel 2 pixels at a
  time, halving height/width (32x32 -> 16x16) and making it much cheaper. padding="same" keeps the size
  otherwise. MaxPooling2D() keeps the max of each 2x2 block (halves height/width again).
- Dropout(0.3) randomly zeroes 30% of values during training only (regularization against overfitting).
- Dense(10, softmax) = one probability per class. Dense(256, relu) = a plain hidden layer.
- Deliberately small so it runs on CPU; a bigger model (two convs per block) gets more accuracy.

READING THE OUTPUT
- Expect roughly 60-65% test accuracy. Watch accuracy vs val_accuracy in the epoch log: if val_accuracy
  stalls while accuracy keeps rising, it's overfitting (more dropout / data augmentation / fewer epochs).
- Confusion matrix: rows = true class, columns = predicted. Bright diagonal = correct. Typical confusions:
  cat/dog, automobile/truck, deer/horse. Class order is 0-9 as listed above.
- See the notes in mnist-scikit-only.py for scaling vs regularization and how to read the plots.
"""

import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train = y_train.ravel()  # (N, 1) -> (N,)
y_test = y_test.ravel()

model = keras.Sequential([
    keras.layers.Input((32, 32, 3)),  # height, width, channels (3 = RGB)
    keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=6, batch_size=256, validation_split=0.1)

y_pred = model.predict(X_test).argmax(axis=1)
print("Test accuracy: %f" % accuracy_score(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
