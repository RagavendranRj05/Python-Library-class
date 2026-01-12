import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split   # only for splitting

# -------------------------------
# Gini Impurity
# -------------------------------
def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

# -------------------------------
# Split Dataset
# -------------------------------
def split_dataset(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# -------------------------------
# Best Split
# -------------------------------
def best_split(X, y):
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, None

    parent_impurity = gini_impurity(y)
    best_gain = 0
    best_feature, best_threshold = None, None

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Weighted impurity
            child_impurity = (len(y_left) / n_samples) * gini_impurity(y_left) + \
                             (len(y_right) / n_samples) * gini_impurity(y_right)

            gain = parent_impurity - child_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature, best_threshold = feature, threshold

    return best_feature, best_threshold

# -------------------------------
# Decision Tree Node
# -------------------------------
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# -------------------------------
# Build Tree
# -------------------------------
def build_tree(X, y, depth=0, max_depth=5):
    if len(np.unique(y)) == 1:   # pure node
        return DecisionTreeNode(value=np.unique(y)[0])

    if depth >= max_depth:
        values, counts = np.unique(y, return_counts=True)
        return DecisionTreeNode(value=values[np.argmax(counts)])

    feature, threshold = best_split(X, y)
    if feature is None:
        values, counts = np.unique(y, return_counts=True)
        return DecisionTreeNode(value=values[np.argmax(counts)])

    X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
    left_node = build_tree(X_left, y_left, depth + 1, max_depth)
    right_node = build_tree(X_right, y_right, depth + 1, max_depth)

    return DecisionTreeNode(feature, threshold, left_node, right_node)

# -------------------------------
# Prediction
# -------------------------------
def predict_one(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(tree, X):
    return [predict_one(tree, x) for x in X]

# -------------------------------
# Main - Heart Attack Prediction
# -------------------------------
if __name__ == "__main__":
    # Load dataset (UCI Heart Disease or Kaggle heart.csv)
    data = pd.read_csv("heart.csv")   # make sure file exists

    # Encode categorical columns
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].map({"M": 1, "F": 0})
    if "ExerciseAngina" in data.columns:
        data["ExerciseAngina"] = data["ExerciseAngina"].map({"Y": 1, "N": 0})

    categorical_cols = ["ChestPainType", "RestingECG", "ST_Slope"]
    for col in categorical_cols:
        if col in data.columns:
            data = pd.get_dummies(data, columns=[col])

    # Features and Target
    X = data.drop("target", axis=1).values
    y = data["target"].values

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree
    tree = build_tree(X_train, y_train, max_depth=5)

    # Predictions
    preds = predict(tree, X_test)

    # Accuracy
    accuracy = np.mean(preds == y_test)
    print("Test Accuracy:", accuracy)