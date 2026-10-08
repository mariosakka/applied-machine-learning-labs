"""
NOTES (MLP on MNIST with scikit-learn)

DATA
- Each MNIST image is 28x28 pixels = 784 numbers, each a grayscale value 0 (black) to 255 (white).
  "mnist_784" in fetch_openml refers to this. X has shape (70000, 784): one flattened image per row.
- Flattening throws away the 2D layout: an MLP doesn't know which pixels are neighbours.
  A CNN keeps the 28x28 grid and uses that structure.

WHY X / 255.0 (scaling) MATTERS
- With raw pixels (0..255) a hidden unit's z = w.x + b is ~255x bigger, softmax saturates, the loss is
  huge, and gradients are ~255x bigger too. With learning_rate_init=0.2, SGD overshoots and diverges.
- Result without scaling: the network outputs the same class for every image (here all "1", the most
  common digit, ~11%), so accuracy ~11% and the confusion matrix is one column. Weight tiles look like
  noise (pareidolia can make one look like a 4).
- Scaled to 0..1, z stays ~1, softmax stays soft, and lr=0.2 takes sensible steps.

HIDDEN LAYERS / HIDDEN UNITS
- Network = chain of layers: input (784 pixels) -> hidden layer(s) -> output (10 digit scores).
- A hidden unit (neuron) is one node in a hidden layer: a = ReLU(w.x + b). "Hidden" because you
  don't directly see or set its value. A hidden layer is a group of them.
- hidden_layer_sizes=(128, 64): the tuple's length = number of hidden layers (2), each entry = number
  of neurons in that layer (128 then 64). (10,) = one layer of 10. () = no hidden layer.
- No hidden layer = logistic regression (~90-92%). One hidden layer ~97-98%. More layers add little on
  MNIST. The first hidden layer is the one that matters most.
- Weight shapes: coefs_[0] is (784, 128) = 784 inputs x 128 units; each column is one unit's 784
  weights. Total weights here: 784*128 + 128*64 + 64*10 (plus biases). .T flips it so we can loop
  over units.
- Each unit learns a template over the WHOLE image; its output says how well the image matches it.
  Later layers take weighted sums of those match scores, and the highest of the 10 outputs wins.
  Units are not assigned a digit; training makes them useful building blocks.
- Only the first layer's weights connect to pixels, so only they can be shown as images. Deeper
  layers combine other units, so they can't.
- Tiles look like digit-ish patterns because w.x is large when the image resembles w, so training
  pushes w toward shapes of the digits it helps detect. Bright = positive weight, dark = negative.
  Tiles are NOT the model redrawing digits (MLPClassifier is a classifier, not generative).

MLP UNIT vs CNN KERNEL
- Same core operation (multiply weights by inputs and sum). An MLP unit is like a kernel as big as
  the whole image that never slides, with different weights per pixel position.
- A CNN kernel is small (e.g. 3x3), slides over the image, and reuses the same weights at every
  position (weight sharing): far fewer parameters, detects a pattern anywhere in the image.

SCALING / NORMALIZATION vs REGULARIZATION (different things!)
- Scaling/normalization = preprocessing of the INPUT data. Min-max scaling squeezes values into a fixed
  range (X / 255.0 -> 0..1). Standardization (StandardScaler) subtracts the mean and divides by the std
  so each feature has mean 0, std 1. "Normalization" is used loosely for either, so check which one.
- Why: gradient descent is sensitive to input size (see above), features on different scales compete
  unfairly, and training is faster and more stable with inputs near 0..1.
- If you use a fitted scaler (StandardScaler), fit it on the TRAIN data only, then apply it to the test
  data, otherwise test information leaks into training. X / 255.0 has no fitted parameters, so no issue.
- Regularization = acts on the MODEL to fight overfitting (memorizing training noise, doing worse on new
  data). L2 / "weight decay" adds a penalty on large weights, pushing toward smaller weights and smoother
  functions. alpha=1e-4 here is that penalty strength: bigger = more regularization, too much =
  underfitting, 0 = off. Other forms: dropout, early stopping, data augmentation, a smaller model.
- In this script: the 11% accuracy was a SCALING problem (training broke). Regularization would matter
  if train accuracy were far above test accuracy.

HOW TO READ THE OUTPUTS
- Console: loss per iteration (verbose=10) should go down; flat or nan = training broken.
  Train score vs test score: both high and close = good fit; train >> test = overfitting (regularize);
  both low = underfitting (bigger model / more training); both ~0.11 = predicts one class for everything.
- Figure 1, confusion matrix: rows = true digit, columns = predicted digit, each cell = count of test
  images. The diagonal = correct predictions (bright diagonal = good). Off-diagonal = errors: a bright
  cell at row 4, column 9 means many 4s were predicted as 9s (typical: 4/9, 3/5/8, 7/1). A single bright
  column = model predicts one class for everything. Accuracy = diagonal sum / total; the matrix also
  shows WHICH digits fail.
- Figure 2, weight tiles (first hidden layer): each tile = one unit's 784 weights as 28x28. White =
  positive weight (ink there raises activation), dark = negative (ink there lowers it), mid-gray ~ 0
  (ignored). Healthy: strokes/loops/blobs on a gray background. Pure noise = undertrained or diverged.
  Flat gray = small weights vs the shared color scale. Empty boxes with 0..1 ticks = unused grid slots,
  meaningless. Tiles are feature detectors, not the model's picture of a digit.

PLOT GRID
- plt.subplots(4, 4) is only the plot grid (4 rows x 4 cols = 16 axes), NOT a model setting.
  zip() stops at the shorter list: with 10 hidden units, 6 axes stay empty (blank boxes with 0..1
  ticks); with 128 units only the first 16 are shown. Use e.g. plt.subplots(8, 16) to show all 128.
- All tiles share one color scale (vmin/vmax), so units with small weights look flat gray.
"""

import warnings  # lets us silence the ConvergenceWarning below

import matplotlib.pyplot as plt  # plotting; "as plt" is just a short alias

from sklearn.datasets import fetch_openml  # downloads MNIST from openml.org
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # the plain (fully connected) neural network

# return_X_y=True -> returns (X, y) instead of one object; as_frame=False -> plain numpy arrays
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # scale pixels from 0..255 to 0..1

# test_size=0.7 -> 70% test / 30% train; random_state=0 -> same split every run
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # 2 hidden layers: 128 neurons, then 64
    max_iter=8,  # max passes (epochs) over the training data
    alpha=1e-4,  # L2 regularization strength (1e-4 = 0.0001)
    solver="sgd",  # stochastic gradient descent
    verbose=10,  # print training progress (loss per iteration)
    random_state=1,  # fixes the random initial weights
    learning_rate_init=0.2,  # step size of each weight update
)

# 8 iterations is too few to converge; catch the warning and ignore it
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

# "%f" is old-style string formatting: inserts the float after the % operator
print("Training set score: %f" % mlp.score(X_train, y_train))  # score() = accuracy
print("Test set score: %f" % mlp.score(X_test, y_test))

y_pred = mlp.predict(X_test)  # predicted digit for every test image
print("Test accuracy: %f" % accuracy_score(y_test, y_pred))
# Confusion matrix: rows = true digit, columns = predicted digit; diagonal = correct predictions
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

fig, axes = plt.subplots(4, 4)  # figure with a 4x4 grid of axes
# "a, b = x, y" unpacks two values; min/max over all first-layer weights give a shared color scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# mlp.coefs_[0].T: one row per hidden unit; axes.ravel() flattens the 4x4 grid into a 1D list;
# zip() pairs them up and stops at the shorter one
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    # reshape(28, 28): 784 weights back into an image; cmap=gray: dark = negative, bright = positive
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())  # () = no ticks (empty tuple)
    ax.set_yticks(())

plt.show()  # display all figures
