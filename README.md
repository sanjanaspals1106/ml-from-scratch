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
- Least squares in matrix form
- Closed-form solution: \( w = (X^T X)^{-1} X^T y \)
- Geometry: projection onto column space
- Invertibility and feature independence
- Failure cases: singularity and multicollinearity
- Why it doesn’t scale (matrix inversion)
- ML view: exact solution vs iterative learning

---

## Book 4 — Logistic Regression (Sigmoid)

- Binary classification setup: \( y \in \{0, 1\} \)
- Linear model produces a confidence score: \( z = w^T x + b \)
- Problem: linear outputs are unbounded and not probabilities
- Sigmoid function: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
- Role of sigmoid: map \( (-\infty, +\infty) \rightarrow (0, 1) \)
- Interpretation: \( \sigma(z) = P(y = 1 \mid x) \)
- Decision boundary at \( z = 0 \Rightarrow \sigma(z) = 0.5 \)

---

## Book 5 — Log Loss & Gradients

- Classification error as probability error, not value error
- Failure of MSE for classification (flat gradients, weak penalties)
- Log loss as confidence-based penalty
- Heavy punishment for confident wrong predictions
- Log loss formula for binary classification
- Numerical stability via probability clipping
- Gradient intuition: correction signal from prediction error


