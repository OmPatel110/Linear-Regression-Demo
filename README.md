# Linear vs Polynomial Regression — Interactive Streamlit Demo

An interactive **Streamlit web app** that visually demonstrates the difference between **Simple Linear Regression** and **Polynomial Regression**, using dynamic plots, model comparison, and real-time metrics.
Built for students, educators, and ML beginners to intuitively understand **underfitting**, **overfitting**, **bias–variance tradeoff**, and **model complexity**.

---

## 🌟 Key Features

### 📊 **1. Visual Model Comparison**

* Compare simple linear regression with higher-degree polynomial fits.
* Add up to **3 polynomial degrees** for side-by-side visualization.
* Real-time updates as you adjust degrees.

---

### ⚙️ **2. Interactive Data Generation**

* Choose the underlying function: **Linear**, **Polynomial**, or **Custom (expression)**.
* Control:

  * Number of points
  * Noise level
  * X-range
  * Train/Test split

---

### 🧠 **3. Two Model Fitting Methods**

You can choose how each polynomial model is trained:

#### ✔ **Analytic (sklearn)** — closed-form solution

#### ✔ **Gradient Descent (demo)** — step-by-step numerical optimization

* Adjustable: Learning rate, epochs, batch size
* Loss curve visualization
* Every selected degree is trained using GD when enabled
* Metrics, predictions, and plots all use GD results

---

### 🎨 **4. Visualization Controls**

* **Show full curve** → reveals the entire polynomial (even extreme areas).
* **Focus mode** → clips extreme values and zooms into the data region.
* Adjustable **Y-axis zoom** slider for refined inspection.

---

### 📈 **5. Model Metrics & Evaluation**

For each selected degree, the app shows:

* **R² (coefficient of determination)**
* **RMSE (root mean squared error)**
* **MAE (mean absolute error)**
* **MSE (mean squared error)**
* Metrics displayed for both **Train** and **Test** sets
* Color-coded quick-metrics cards for fast comparison

---

### 🔮 **6. Predict & Compare**

Enter any X-value and instantly get:

* Predictions for **all selected degrees**
* Clean green success cards
* Pairwise differences (when 2+ degrees selected)

---

### 🧭 **7. Bias–Variance Sweep (U-Curve Visualization)**

A dedicated mode to illustrate the famous ML concept:

* Computes Train/Test MSE for **degrees 1–12**
* Highlights:

  * **Underfitting region** (high bias)
  * **Overfitting region** (high variance)
  * **Sweet spot** (minimum test error)
  * **Current model degree** selected by the user

---

### 💾 **8. Export Options**

* **Download dataset** as CSV

---

## 🚀 Installation & Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

Made by **Om Patel**
🔗 **LinkedIn:** [https://www.linkedin.com/in/om-patel-tech/](https://www.linkedin.com/in/om-patel-tech/)
