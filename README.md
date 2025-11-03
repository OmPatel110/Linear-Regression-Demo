# 🎓 Linear vs Polynomial Regression — Interactive Streamlit Demo

An interactive **Streamlit web app** that visually demonstrates the difference between **Simple Linear Regression** and **Polynomial Regression** using dynamic plots, interactivity, and real-time metrics.  
Designed for learning and teaching key machine learning concepts like **underfitting**, **overfitting**, and **model complexity**.

---

## 🌟 Features

✅ **Visual model comparison**
- Compare simple linear vs higher-degree polynomial fits.
- Observe how the regression curve changes with degree.

✅ **Interactive data generation**
- Add random **noise** to simulate real-world data.
- Adjust **train/test split** size.
- Optionally clip extreme values for better visualization.

✅ **Model control panel**
- Choose polynomial **degree(s)** dynamically (1 to 10+).
- Add up to 3 degrees for **side-by-side visual comparison**.
- Switch between **regularization methods** (`None`, `Ridge`, `Lasso`).
- Tune regularization **alpha** value.

✅ **Visualization controls**
- Toggle **Show full curve** or **Focus on data region**.
- Adjust **Y-axis zoom** dynamically.
- Click a regression line in the chart to **highlight** it — other lines fade out.

✅ **Metrics & evaluation**
- View **R²**, **RMSE**, **MAE**, and **MSE** for training and testing sets.
- Compare performance across selected polynomial degrees.

✅ **Predict & compare**
- Enter any X-value and instantly view predictions for each selected polynomial degree.
- See prediction differences across models in green-highlighted cards.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/yourusername/polynomial-regression-demo.git
cd polynomial-regression-demo
