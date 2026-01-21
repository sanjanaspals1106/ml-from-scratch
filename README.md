# ML From Scratch

Core Machine Learning algorithms implemented **entirely from scratch** using NumPy,
with a strong focus on **intuition, geometry, and training dynamics** rather than
library abstractions.

This repository treats machine learning as a process of:
**derivation → implementation → optimization → evaluation**.

No high-level ML libraries are used.

---

## Structure & Progress

### Book 1 — Linear Regression (Geometry of Prediction)
- Linear model intuition: y = Xw + b
- Predictions as projections
- Residuals as geometric distances
- Underfitting vs good fit
- ML view: fitting a line/plane to data

---

## Book 2 — Linear Regression (Learning via Optimization)
- Loss function: Mean Squared Error (MSE)
- Error as a function of parameters
- Loss surface and convexity
- Gradient as direction of steepest increase
- Gradient descent as iterative learning
- Learning rate and convergence
- ML view: learning parameters by minimizing error

---

## Book 3 — Linear Regression (Normal Equation)
- Least squares formulation in matrix form
- Objective: minimize squared error  
  \[
  \min_w \|Xw - y\|^2
  \]
- Closed-form solution:
  \[
  w = (X^\top X)^{-1} X^\top y
  \]
- Geometric interpretation: projection of \( y \) onto the column space of \( X \)
- Conditions for invertibility of \( X^\top X \)
- Failure cases: singular matrices and multicollinearity
- Computational cost of matrix inversion
- Why the normal equation does not scale
- ML view: exact solution vs iterative learning

---

## Book 4 — Logistic Regression (Sigmoid)

- Binary classification setup:  
  \[
  y \in \{0, 1\}
  \]
- Linear model produces a score:
  \[
  z = w^\top x + b
  \]
- Problem: linear scores are unbounded and not probabilities
- Sigmoid function:
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
- Mapping:
  \[
  (-\infty, +\infty) \rightarrow (0, 1)
  \]
- Probabilistic interpretation:
  \[
  \sigma(z) = P(y = 1 \mid x)
  \]
- Decision boundary:
  \[
  z = 0 \;\Rightarrow\; \sigma(z) = 0.5
  \]

---

## Book 5 — Log Loss & Gradients

- Classification error as probability error, not value error
- Failure of MSE for classification (flat gradients, weak penalties)
- Log loss as confidence-based penalty
- Heavy punishment for confident wrong predictions
- Log loss formula for binary classification
- Numerical stability via probability clipping
- Gradient intuition: correction signal from prediction error

---

## Book 6 — Logistic Regression (Training Loop)

- Binary classification using gradient descent
- Linear score computation:
  \[
  z = w^\top x + b
  \]
- Sigmoid maps score to confidence:
  \[
  \hat{y} = \sigma(z)
  \]
- Log loss penalizes confident wrong predictions
- Gradient signal for parameters:
  \[
  \nabla_w \mathcal{L} = (\hat{y} - y)x
  \]
- Training updates a linear decision boundary
- Loss reflects confidence improvement; accuracy may plateau
- Learning rate controls convergence behavior

---

## Book 7 — Model Evaluation (Judging Performance)

- Training vs testing performance
- Why training accuracy is misleading
- Generalization and unseen data
- Manual train–test split
- Confusion matrix as the foundation of evaluation
- Error types: true positives, false positives, false negatives, true negatives
- Accuracy and its failure on imbalanced data
- Precision as control over false positives
- Recall as control over false negatives
- Precision–recall trade-off via decision threshold
- F1 score as balanced performance measure
- Evaluation as failure analysis, not score maximization
- ML view: understanding *how* and *why* a model fails
