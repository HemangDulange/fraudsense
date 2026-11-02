"""Supervised Logistic Regression baseline implemented from scratch with NumPy.

Everyday Analogy:
-----------------
Imagine an experienced fraud risk analyst working with a scorecard.
For each transaction, they review 30 questions (Time, Amount, and the 28 PCA features).
Each question has a specific weight: for instance, an unusually large purchase at 4:00 AM
might add +3.5 risk points, while a transaction consistent with typical weekday behavior
might subtract -2.0 points.
The analyst adds up all the weighted points plus a starting base score.
Finally, to convert this raw score into a calibrated probability percentage between 0% and 100%,
they pass it through an S-shaped curve (the sigmoid function).

Algorithm Description:
----------------------
1. Forward Pass:
   - Linear score: z = X @ w + b
   - Sigmoid activation: p = 1 / (1 + exp(-clip(z, -25, 25)))
   where p represents the model's estimated probability that Class = 1 (Fraud).

2. Class Weighting for Severe Imbalance:
   Because fraud cases represent only ~0.17% of transactions, standard unweighted training
   can get stuck predicting all 0s. We apply balanced class weighting:
       w_0 = N / (2 * N_0),   w_1 = N / (2 * N_1)
   so misclassifying a single rare fraud penalizes the loss function much more heavily.

3. Loss and Gradient Optimization:
   Weighted Binary Cross-Entropy with optional L2 regularization:
       L = - (1/N) * sum_i s_i * [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ] + 0.5 * l2 * ||w||_2^2
   Gradients:
       g_i = s_i * (p_i - y_i)
       grad_w = (1/N) * X^T @ g + l2 * w
       grad_b = (1/N) * sum(g)
   Updated via batch gradient descent over n_epochs.

4. Interface:
   - anomaly_score(X) returns the predicted probability P(Class=1 | X).
"""

from typing import Dict, Any, Optional
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z_clipped = np.clip(z, -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


class LogisticRegressionFromScratch:
    """Binary Logistic Regression with class weighting and L2 regularization.

    Parameters
    ----------
    lr : float, default=0.05
        Learning rate for gradient descent.
    n_epochs : int, default=500
        Number of gradient descent iterations.
    l2_reg : float, default=0.01
        L2 regularization penalty (weight decay). Set to 0.0 to disable.
    use_class_weights : bool, default=True
        Whether to adjust sample weights inversely proportional to class frequencies.
    """

    def __init__(
        self,
        lr: float = 0.05,
        n_epochs: int = 500,
        l2_reg: float = 0.01,
        use_class_weights: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}")
        if l2_reg < 0:
            raise ValueError(f"l2_reg cannot be negative, got {l2_reg}")

        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.l2_reg = float(l2_reg)
        self.use_class_weights = bool(use_class_weights)

        self.weights_: Optional[np.ndarray] = None  # shape: (n_features,)
        self.bias_: float = 0.0
        self.loss_history_: list = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionFromScratch":
        """Fit weights using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Binary labels (0 or 1), shape (n_samples,).

        Returns
        -------
        self : LogisticRegressionFromScratch
            Fitted classifier.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D features array, got shape {X_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Mismatch: len(X)={len(X_arr)} while len(y)={len(y_arr)}"
            )

        n_samples, n_features = X_arr.shape

        # Initialize weights to zeros (or small values)
        self.weights_ = np.zeros(n_features, dtype=np.float32)
        self.bias_ = 0.0
        self.loss_history_ = []

        # Compute sample weights if class weighting is enabled
        if self.use_class_weights:
            n_pos = np.sum(y_arr == 1.0)
            n_neg = np.sum(y_arr == 0.0)
            if n_pos > 0 and n_neg > 0:
                w_pos = n_samples / (2.0 * n_pos)
                w_neg = n_samples / (2.0 * n_neg)
                sample_weights = np.where(y_arr == 1.0, w_pos, w_neg).astype(np.float32)
            else:
                sample_weights = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weights = np.ones(n_samples, dtype=np.float32)

        # Batch Gradient Descent Loop
        for epoch in range(self.n_epochs):
            # 1. Forward linear score: z = X @ w + b
            z = X_arr @ self.weights_ + self.bias_
            p = sigmoid(z)

            # 2. Compute error weighted by class weights
            # g_i = s_i * (p_i - y_i)
            error = sample_weights * (p - y_arr)

            # 3. Gradients with L2 regularization
            grad_w = (X_arr.T @ error) / n_samples + self.l2_reg * self.weights_
            grad_b = float(np.sum(error) / n_samples)

            # 4. Parameter update step
            self.weights_ -= self.lr * grad_w.astype(np.float32)
            self.bias_ -= self.lr * grad_b

            # Track loss every 50 epochs (or for records)
            if epoch % 50 == 0 or epoch == self.n_epochs - 1:
                eps = 1e-12
                bce = -np.mean(
                    sample_weights * (y_arr * np.log(p + eps) + (1.0 - y_arr) * np.log(1.0 - p + eps))
                )
                l2_penalty = 0.5 * self.l2_reg * np.sum(self.weights_**2)
                self.loss_history_.append(float(bce + l2_penalty))

        self.is_fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score as predicted probability of fraud.

        Higher score = higher probability of fraud = more suspicious.
        Outputs range in [0, 1].

        Parameters
        ----------
        X : np.ndarray
            Features to score, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1D float32 array of predicted probabilities, shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        z = X_arr @ self.weights_ + self.bias_
        probabilities = sigmoid(z)
        return probabilities.astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classify transactions as 0 (Normal) or 1 (Fraud) at a chosen threshold."""
        scores = self.anomaly_score(X)
        return (scores >= threshold).astype(np.int8)

    def to_state(self) -> Dict[str, Any]:
        """Serialize model state to a dictionary without pickle."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted LogisticRegressionFromScratch.")

        return {
            "model_type": "LogisticRegressionFromScratch",
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "l2_reg": self.l2_reg,
            "use_class_weights": self.use_class_weights,
            "weights": self.weights_.tolist(),
            "bias": float(self.bias_),
            "loss_history": self.loss_history_,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "LogisticRegressionFromScratch":
        """Reconstruct LogisticRegression from serialized state dictionary."""
        instance = cls(
            lr=state.get("lr", 0.05),
            n_epochs=state.get("n_epochs", 500),
            l2_reg=state.get("l2_reg", 0.01),
            use_class_weights=state.get("use_class_weights", True),
        )
        instance.weights_ = np.array(state["weights"], dtype=np.float32)
        instance.bias_ = float(state["bias"])
        instance.loss_history_ = state.get("loss_history", [])
        instance.is_fitted = bool(state.get("is_fitted", True))
        return instance
