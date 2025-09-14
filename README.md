# Wildfire Smoke Spread Prediction  

This project is part of my AI/ML coursework and focuses on predicting **wildfire smoke spread** using weather and fire-related data. The goal is to build a model that helps understand how environmental conditions influence wildfire smoke intensity and spread.  

---

## 📌 Project Lifecycle (Current Progress)
✔️ Problem Definition  
✔️ Data Collection (CSV dataset with weather + fire parameters)  
✔️ Data Preprocessing (cleaning, feature selection, train-test split, scaling)  
⏳ Model Training (Next step)  
⏳ Evaluation & Deployment  

---

## 📂 Dataset
The dataset contains weather and fire attributes, including:  

- `lat`, `lon` → Location  
- `fire_weather_index` → Fire risk indicator  
- `pressure_mean` → Atmospheric pressure  
- `wind_speed_max`, `wind_direction_mean` → Wind conditions  
- `temp_mean`, `humidity_min`, `dewpoint_mean` → Temperature & humidity  
- `cloud_cover_mean`, `solar_radiation_mean` → Atmospheric effects  
- `occured` → Whether fire occurred (Classification target)  
- `frp` → Fire Radiative Power (Regression target, fire intensity)  

---

## ⚙️ Preprocessing Steps Done
- Loaded dataset with Pandas  
- Checked for missing values  
- Scaled numerical features  
- Train-test split for both classification (`occured`) and regression (`frp`)  

---

## 🚀 Next Steps
- Train ML models (Logistic Regression, Random Forest, Gradient Boosting, etc.)  
- Evaluate performance (Accuracy, RMSE, R² score)  
- Add visualizations for feature importance & predictions  
- Deploy or report findings  

---

## 🛠️ Tech Stack
- Python 3.13  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib / Seaborn (for visualization)  

---

## 📜 How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/wildfire-smoke-prediction.git
   cd wildfire-smoke-prediction
2. Create a virtual environment and install dependencies
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows
    pip install -r requirements.txt
3. Open wildfire.ipynb in Jupyter Notebook or VS Code and run the cells.
