from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
import graphviz

def train_models(X_train, y_train, save_dir="models/"):
    """Train Decision Tree, KNN, and SVM, save as .pkl files."""
    os.makedirs(save_dir, exist_ok=True)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=7),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))
    return models

def load_models(save_dir="models/"):
    """Load trained models from .pkl files."""
    models = {
        "Decision Tree": joblib.load(os.path.join(save_dir, "Decision Tree.pkl")),
        "KNN": joblib.load(os.path.join(save_dir, "KNN.pkl")),
        "SVM": joblib.load(os.path.join(save_dir, "SVM.pkl"))
    }
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate and return accuracy, precision, and F1 scores for each model."""
    scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        scores[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='binary'),
            "f1_score": f1_score(y_test, y_pred, average='binary')
        }
    return scores

def save_decision_tree_graph(model, feature_names, filepath="tree.png"):
    """Save Decision Tree graph as PNG."""
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(filepath.replace(".png", ""), format="png", cleanup=True)

def save_knn_graph(X_train, y_train, filepath="knn_plot.png"):
    """Save KNN PCA graph as PNG."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df["target"] = y_train
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="target", palette="Set1")
    plt.title("KNN - PCA Projection")
    plt.savefig(filepath)
    plt.close()
    

def save_nb_distribution(X_train, y_train, filepath="nb_dist.png"):
    """Save Naive Bayes distribution of first feature."""
    df = pd.DataFrame(X_train)
    df["target"] = y_train
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x=0, hue="target", bins=30, element="step", kde=True)
    plt.title("Naive Bayes - Distribution of First Feature")
    plt.xlabel("Feature 1")
    plt.savefig(filepath)
    plt.close()

def plot_multiple_comparisons(scores, filename_prefix="plots/comparison"):
    """Save Comparison Plots for Accuracy, Precision, F1 Score."""
    labels = list(scores.keys())

    # Accuracy Bar Plot
    fig, ax = plt.subplots(figsize=(8,5))
    acc_values = [scores[m]["accuracy"] for m in labels]
    ax.bar(labels, acc_values, color=["steelblue", "green", "red"])
    ax.set_title("Model Comparison - Accuracy")
    ax.set_ylabel("Accuracy")
    for i, v in enumerate(acc_values): 
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    fig.savefig(f"{filename_prefix}_accuracy.png"); plt.close(fig)

    # Precision Line Plot
    fig, ax = plt.subplots(figsize=(8,5))
    precision_values = [scores[m]["precision"] for m in labels]
    ax.plot(labels, precision_values, marker='o', linestyle='-', color="purple")
    ax.set_title("Model Comparison - Precision")
    ax.set_ylabel("Precision")
    for i, v in enumerate(precision_values): 
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    fig.savefig(f"{filename_prefix}_precision.png"); plt.close(fig)

    # F1 Score Pie Plot
    fig, ax = plt.subplots(figsize=(5,5))
    f1_values = [scores[m]["f1_score"] for m in labels]
    ax.pie(f1_values, labels=labels, autopct='%1.1f%%', colors=["steelblue", "green", "red"])
    ax.set_title("F1 Score Comparison (Pie Chart)")
    fig.savefig(f"{filename_prefix}_f1_score.png"); plt.close(fig)

def plot_model_correctness(y_test, y_pred, title, filename):
    """Save Pie chart for Correct vs Incorrect Predictions."""
    total_samples = len(y_test)
    correct = sum(y_test == y_pred)
    incorrect = total_samples - correct
    labels = ["Correct", "Incorrect"]
    sizes = [correct, incorrect]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=["green", "red"])
    ax.set_title(title)
    fig.savefig(filename)
    plt.close(fig)
