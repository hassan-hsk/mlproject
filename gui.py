import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import os

from preprocessing import load_and_preprocess_data
from models import (
    train_models,
    load_models,
    evaluate_models,
    save_decision_tree_graph,
    save_knn_graph,
    save_nb_distribution,
    plot_multiple_comparisons
)

# ---- Load and Preprocess ----
X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data("heart.csv")

# ---- Model Loading ----
if os.path.exists("models/Decision Tree.pkl"):
    models = load_models()
else:
    models = train_models(X_train, y_train)

scores = evaluate_models(models, X_test, y_test)

# ---- Save Graphs ----
save_decision_tree_graph(models["Decision Tree"], feature_names, "tree.png")
save_knn_graph(X_train, y_train, "knn_plot.png")
save_nb_distribution(X_train, y_train, "nb_dist.png")
os.makedirs("plots", exist_ok=True)
plot_multiple_comparisons(scores, "plots/comparison")

# ---- App ----
root = tk.Tk()
root.geometry("1000x700")
root.title("Heart Disease Prediction System")
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

def display_image(frame, filepath, title, subtitles=None):
    """Display an image if available, with title and subtitles."""
    title_label = tk.Label(frame, text=title, font=("Arial", 14, "bold"))
    title_label.pack(pady=5)

    if filepath:
        try:
            img = Image.open(filepath)
            img = img.resize((600, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            panel = tk.Label(frame, image=photo)
            panel.image = photo
            panel.pack(pady=5)
        except Exception as e:
            tk.Label(frame, text=f"Error loading image ({e})").pack()
    else:
        tk.Label(frame, text="(No image available for this model.)").pack()

    if subtitles:
        for line in subtitles:
            tk.Label(frame, text=line).pack()

# ---- Comparison Page ----
comp_frame = ttk.Frame(notebook)
notebook.add(comp_frame, text="Comparative Analysis")
title_label = tk.Label(comp_frame, text="Comparative Model Analysis", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

canvas = tk.Canvas(comp_frame, width=900, height=600)
scrollbar = ttk.Scrollbar(comp_frame, orient="vertical", command=canvas.yview)
scroll_frame = ttk.Frame(canvas)

scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

display_image(scroll_frame, "plots/comparison_accuracy.png", "Accuracy Comparison (Bar Graph)")
display_image(scroll_frame, "plots/comparison_precision.png", "Precision Comparison (Line Graph)")
display_image(scroll_frame, "plots/comparison_f1_score.png", "F1 Score Comparison (Pie Chart)")

# ---- Model Pages ----
def make_model_page(model_name, description, image_file):
    """Create a page for the given model."""
    page = ttk.Frame(notebook)
    notebook.add(page, text=model_name)

    label = tk.Label(page, text=f"{model_name} Model", font=("Arial", 16, "bold"))
    label.pack(pady=10)

    text = tk.Label(page, text=description, wraplength=600, justify="left")
    text.pack(pady=10)

    acc = scores[model_name]["accuracy"]
    label_acc = tk.Label(page, text=f"Accuracy: {acc:.2f}", font=("Arial", 12))
    label_acc.pack(pady=10)

    display_image(page, image_file, f"{model_name} Plot")
    return page

make_model_page("Decision Tree", "A Decision Tree makes decisions by splitting data into branches based on feature values.", "tree.png")
make_model_page("KNN", "K-Nearest Neighbors classifies by finding the closest points in the dataset.", "knn_plot.png")
make_model_page("SVM", "Support Vector Machine finds an optimal hyperplane for classification.", None)

# ---- Prediction Page ----
predict_frame = ttk.Frame(notebook)
notebook.add(predict_frame, text="Predict")
predict_label = tk.Label(predict_frame, text="Enter Patient Details", font=("Arial", 16, "bold"))
predict_label.pack(pady=10)

form_frame = ttk.Frame(predict_frame)
form_frame.pack(pady=10)

entries = {}
for i, feature in enumerate(feature_names):
    lbl = tk.Label(form_frame, text=f"{feature}: ")
    lbl.grid(row=i, column=0, sticky="e", padx=5, pady=3)
    entry = tk.Entry(form_frame, width=30)
    entry.grid(row=i, column=1, padx=5, pady=3)
    entries[feature] = entry

btn_frame = ttk.Frame(predict_frame)
btn_frame.pack(pady=20)

def fill_sample_values():
    """Better example yielding disease across Decision Tree, KNN, and SVM."""
    sample_values = {
        "age": 67,
        "sex": 1,
        "cp": 3,
        "trestbps": 160,
        "chol": 315,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 1,
        "oldpeak": 3.5,
        "slope": 3,
        "ca": 3,
        "thal": 3
    }
    for feature, entry in entries.items():
        entry.delete(0, tk.END)
        entry.insert(0, sample_values.get(feature, ""))

def clear_form():
    """Clear all fields."""
    for entry in entries.values():
        entry.delete(0, tk.END)

def save_prediction(result):
    """Save prediction result."""
    with open("last_prediction.txt", "w") as f:
        f.write(result)

def predict():
    """Perform prediction based on user inputs."""
    try:
        input_data_array = np.array([[float(entries[feature].get()) for feature in feature_names]])
        input_scaled = scaler.transform(input_data_array)

        results = []
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
            results.append(f"{name}: {'Disease' if pred == 1 else 'No Disease'}"
                           + (f" (Confidence: {prob:.2%})" if prob is not None else ""))

        result = "\n".join(results)

        save_prediction(result)
        messagebox.showinfo("Prediction Result", result)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values in all fields.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Buttons
predict_btn = tk.Button(btn_frame, text="Predict Disease", command=predict, bg="green", fg="white", font=("Arial", 12, "bold"))
predict_btn.grid(row=0, column=0, padx=10)

fill_btn = tk.Button(btn_frame, text="Fill Sample Values", command=fill_sample_values, bg="blue", fg="white", font=("Arial", 12, "bold"))
fill_btn.grid(row=0, column=1, padx=10)

clear_btn = tk.Button(btn_frame, text="Clear Form", command=clear_form, bg="red", fg="white", font=("Arial", 12, "bold"))
clear_btn.grid(row=0, column=2, padx=10)

root.mainloop()
