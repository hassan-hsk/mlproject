
# ❤️ Heart Disease Prediction System

This project implements a machine learning-based system to predict heart disease risk using patient data. It includes a graphical user interface (GUI) built with Tkinter, multiple trained models (Decision Tree, KNN, SVM), and various visualizations to analyze model performance.

---

## Project Overview

The Heart Disease Prediction System helps users input patient details and predicts the likelihood of heart disease using three different ML models. It also provides comparative visualizations to evaluate model accuracy, precision, and F1 scores.

---

## Project Structure

```


├── models/                     # Saved ML model files (.pkl)
├── plots/                      # Generated graphs and plots
├── heart.csv                   # Dataset file containing patient features and labels
├── main.py                    # Main GUI application script (Tkinter app)
├── preprocessing.py           # Data loading and preprocessing functions
├── models.py                  # Model training, evaluation, and plotting functions
├── last_prediction.txt        # File to save last prediction results

````

---

## Features

- Trains and uses Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) models
- GUI for easy input of patient details and prediction display
- Saves and displays multiple visualizations for model comparison:
  - Decision Tree structure
  - KNN PCA projection
  - Model accuracy, precision, and F1 score comparisons
- Saves last prediction result to a file
- Includes sample data filling and form clearing options in the GUI

---

## Installation


1. Clone the repository:

   ```bash
   git clone https://github.com/hassan-hsk/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: The requirements should include packages like scikit-learn, pandas, numpy, matplotlib, seaborn, pillow, graphviz, and joblib.*

4. Make sure Graphviz is installed on your system (for decision tree visualization):

   * Windows: Download from [https://graphviz.org/download/](https://graphviz.org/download/)
   * macOS: `brew install graphviz`
   * Linux: Use your distro package manager

---

## Running the Application

Run the main Tkinter app:

```bash
python main.py
```

The GUI window will open allowing you to:

* Input patient data
* Predict heart disease using multiple models
* View visual model comparisons and plots

---

## Dataset

The dataset used (`heart.csv`) contains patient information including:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Fasting blood sugar
* Resting electrocardiographic results
* Maximum heart rate achieved
* Exercise-induced angina
* ST depression induced by exercise
* Slope of peak exercise ST segment
* Number of major vessels colored by fluoroscopy
* Thalassemia status
* Target label indicating presence (1) or absence (0) of heart disease

---

## Model Details

* **Decision Tree:** Classifies based on feature splits, max depth set to 7 for balance between bias and variance.
* **K-Nearest Neighbors (KNN):** Classifies based on the closest 5 neighbors.
* **Support Vector Machine (SVM):** Uses a probabilistic kernel to find optimal separating hyperplane.

Models are trained on the training dataset and saved as `.pkl` files in the `models/` directory.

---

## Visualizations

* **Decision Tree Plot:** Shows the decision nodes and branches.
* **KNN PCA Projection:** 2D visualization of data points projected by Principal Component Analysis.
* **Comparative Analysis:** Bar, line, and pie charts comparing accuracy, precision, and F1 scores of models.

Plots are saved in the `plots/` directory and displayed within the app.

---

## How to Update Models

To retrain models with updated data:

1. Update the dataset `heart.csv`.
2. Run the training script (or the part of your code that calls `train_models`).
3. New model files and plots will be generated automatically.

---

## Author

**Hassan Sarfraz**

---
