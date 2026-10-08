"""
NOTES (CNN on MNIST with Keras)

- Same dot-product idea as the MLP, but each Conv2D kernel is small (3x3 = 9 weights), slides over
  the image, and reuses the same weights at every position (weight sharing). Far fewer parameters,
  and a pattern is detected anywhere in the image. The 28x28 grid is kept (no flattening until the end).
- Conv2D(32, 3): 32 kernels of size 3x3. MaxPooling2D(): keeps the max of each 2x2 block (halves the size).
  Flatten(): turns the final feature maps into one vector for the Dense layers.
- Dense(64, relu) = a plain hidden layer of 64 units (same as the MLP). Dense(10, softmax) = the output:
  10 probabilities, one per digit.
- Scaling X to 0..1 matters here too (see mnist-scikit-only.py).
- Reading the output: the confusion matrix reads the same as in the sklearn script (rows = true digit,
  columns = predicted, bright diagonal = good). The epoch log shows accuracy (training) and val_accuracy
  (held-out 10%); if val_accuracy stalls or falls while accuracy keeps rising, that's overfitting.
- Scaling (input preprocessing) and regularization (model-side, fights overfitting) are different
  things; see the notes in mnist-scikit-only.py.
- Uses the standard 60k train / 10k test split, so accuracy is not directly comparable to the sklearn
  script (30% train / 70% test).
"""

import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# "(a, b), (c, d) = ..." unpacks nested tuples: ((train images, train labels), (test images, test labels))
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# [..., None] adds a channel axis: (N, 28, 28) -> (N, 28, 28, 1), since Conv2D expects channels (1 = grayscale)
X_train = X_train[..., None] / 255.0
X_test = X_test[..., None] / 255.0

model = keras.Sequential([  # Sequential = layers stacked one after another
    keras.layers.Input((28, 28, 1)),  # input shape: height, width, channels
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
# adam = optimizer (smarter SGD); the loss expects integer labels 0-9 (not one-hot)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# epochs = passes over the data; batch_size = images per weight update; validation_split = 10% held out
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# predict() returns 10 probabilities per image; argmax(axis=1) picks the most likely digit per row
y_pred = model.predict(X_test).argmax(axis=1)
print("Test accuracy: %f" % accuracy_score(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
