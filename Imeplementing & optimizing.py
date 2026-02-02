import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt

# -----------------------------
# Pinball (Quantile) Loss
# -----------------------------
def pinball_loss(y, y_pred, q):
    return np.mean(np.maximum(q * (y - y_pred), (q - 1) * (y - y_pred)))


# -----------------------------
# Quantile Regression Tree
# -----------------------------
class QuantileTree:
    def __init__(self, max_depth=5, min_samples_leaf=10, quantile=0.5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.quantile = quantile
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(y) <= self.min_samples_leaf
        ):
            return {"value": np.quantile(y, self.quantile)}

        best_feature, best_threshold, best_loss = None, None, np.inf

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                left_pred = np.quantile(y[left_mask], self.quantile)
                right_pred = np.quantile(y[right_mask], self.quantile)

                loss = (
                    pinball_loss(y[left_mask], left_pred, self.quantile)
                    + pinball_loss(y[right_mask], right_pred, self.quantile)
                )

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return {"value": np.quantile(y, self.quantile)}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _predict_row(self, row, node):
        if "value" in node:
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        else:
            return self._predict_row(row, node["right"])

    def predict(self, X):
        return np.array([self._predict_row(x, self.tree) for x in X])


# -----------------------------
# Quantile Regression Forest
# -----------------------------
class QuantileRegressionForest:
    def __init__(self, n_estimators=30, max_depth=5, min_samples_leaf=10, quantile=0.5):
        self.trees = [
            QuantileTree(max_depth, min_samples_leaf, quantile)
            for _ in range(n_estimators)
        ]

    def fit(self, X, y):
        n = len(X)
        for tree in self.trees:
            idx = np.random.choice(n, n, replace=True)
            tree.fit(X[idx], y[idx])

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return preds.mean(axis=0)


# -----------------------------
# Heteroscedastic Data Generator
# -----------------------------
def generate_data(n=1000):
    X = np.random.uniform(0, 10, size=(n, 1))
    noise = np.random.normal(0, 0.3 + 0.3 * X.squeeze())
    y = 2 * X.squeeze() + noise
    return X, y


# -----------------------------
# Experiment
# -----------------------------
np.random.seed(42)

X, y = generate_data()
X_test, y_test = generate_data(300)

quantiles = [0.1, 0.5, 0.9]
qrf_models = {}

for q in quantiles:
    qrf = QuantileRegressionForest(
        n_estimators=40,
        max_depth=6,
        min_samples_leaf=15,
        quantile=q
    )
    qrf.fit(X, y)
    qrf_models[q] = qrf

# Linear Regression Baseline
lr = LinearRegression()
lr.fit(X, y)
lr_preds = lr.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("Pinball Loss Comparison:\n")

for q in quantiles:
    qrf_preds = qrf_models[q].predict(X_test)
    loss = mean_pinball_loss(y_test, qrf_preds, alpha=q)
    print(f"QRF Quantile {q}: {loss:.4f}")

median_loss_lr = mean_pinball_loss(y_test, lr_preds, alpha=0.5)
print(f"\nLinear Regression (Median): {median_loss_lr:.4f}")

# -----------------------------
# Visualization
# -----------------------------
x_sorted = np.argsort(X_test.squeeze())
plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, s=10, alpha=0.3, label="Data")

for q in quantiles:
    plt.plot(
        X_test[x_sorted],
        qrf_models[q].predict(X_test)[x_sorted],
        label=f"QRF q={q}"
    )

plt.plot(
    X_test[x_sorted],
    lr_preds[x_sorted],
    color="black",
    linestyle="--",
    label="Linear Regression"
)

plt.legend()
plt.title("Quantile Regression Forest vs Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
