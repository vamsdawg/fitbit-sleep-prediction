# 🌙 Fitbit Sleep Quality Prediction - Team Setup Guide

**DSBA 6211 Final Project**

ML project predicting sleep quality from Fitbit activity data. Includes data processing notebooks, trained models (Logistic Regression, Random Forest, XGBoost), and an interactive Streamlit web app.

---

## � Quick Setup (5 Minutes)

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd "<repository-folder-name>"
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate           # Mac/Linux
# OR
.venv\Scripts\activate              # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**✅ Setup Complete!** You're ready to run everything.

---

## � Running the Notebooks

### Option A: Using Jupyter Notebook
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook
```

Then open either:
1. **`DataCleaning-EDA-FeatureEngineering.ipynb`** - Data processing & feature engineering
2. **`ModelTraining.ipynb`** - Model training & evaluation

### Option B: Using VS Code
1. Open the `.ipynb` file in VS Code
2. Select kernel: Choose `.venv` (Python 3.9.x)
3. Run cells with `Shift + Enter`

**Note:** All notebooks are already executed. You can review outputs or re-run cells as needed.

---

## 🎮 Running the Streamlit App

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Run the app
streamlit run app.py
```

The app will open automatically at **`http://localhost:8501`**

### What You Can Do:
- **Page 1 - Quick Predict:** Enter daily activity → Get sleep quality prediction + personalized recommendations
- **Page 2 - Batch Analysis:** Upload CSV with multiple days → Analyze trends
- **Page 3 - Learn More:** View feature importance & model details
- **Page 4 - About:** Model comparison & project info

---

## 📁 Project Files

```
├── DataCleaning-EDA-FeatureEngineering.ipynb   # Notebook 1: Data processing
├── ModelTraining.ipynb                          # Notebook 2: Model training
├── app.py                                       # Streamlit web application
├── requirements.txt                             # Python dependencies
│
├── Data Files (8 CSV files)
│   ├── merged_activity_data.csv                 # Daily activity
│   ├── merged_calories_data.csv                 # Hourly calories
│   ├── merged_heartrate_data.csv                # Heart rate
│   ├── merged_intensity_data.csv                # Activity intensity
│   ├── merged_steps_data.csv                    # Hourly steps
│   ├── merged_weightLogInfo_data.csv            # Weight/BMI
│   ├── minuteSleep_merged.csv                   # Sleep data
│   └── ml_ready_sleep_prediction_FINAL.csv      # Final ML dataset
│
└── Model Files (6 .pkl files)
    ├── xgboost_model.pkl                        # Best model (85% accuracy)
    ├── random_forest_model.pkl                  # Random Forest
    ├── logistic_regression_model.pkl            # Logistic Regression
    ├── logistic_regression_scaler.pkl           # Feature scaler
    ├── feature_names.pkl                        # Feature names
    └── model_metadata.pkl                       # Model metadata
```

---

## 🔧 Troubleshooting

### ❌ "ModuleNotFoundError"
**Fix:** Activate virtual environment and install dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### ❌ "Port 8501 already in use"
**Fix:** Use a different port
```bash
streamlit run app.py --server.port 8502
```

### ❌ Jupyter kernel not found
**Fix:** Install ipykernel
```bash
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name="Python (.venv)"
```

### ❌ "File not found" errors
**Fix:** Make sure you're in the project root directory
```bash
pwd  # Should show path ending in "Final Project"
ls   # Should see app.py, *.ipynb, *.csv files
```

### ❌ Import errors in notebooks
**Fix:** Select the correct kernel (`.venv`) in Jupyter/VS Code

---

## � Workflow Overview

### If You Want to Review the Full Pipeline:

**Step 1:** Run `DataCleaning-EDA-FeatureEngineering.ipynb`
- Loads raw CSV files
- Performs exploratory data analysis
- Engineers features
- Exports `ml_ready_sleep_prediction_FINAL.csv`

**Step 2:** Run `ModelTraining.ipynb`
- Trains 3 ML models (Logistic Regression, Random Forest, XGBoost)
- Evaluates performance
- Creates SHAP explanations
- Saves models as `.pkl` files

**Step 3:** Run the Streamlit app
- Uses the saved models
- Provides interactive predictions
- Visualizes results

### If You Just Want to Test the App:
Just run `streamlit run app.py` - all models are pre-trained!

---

## 📊 Model Performance (Quick Reference)

| Model               | Accuracy | 
|---------------------|----------|
| Logistic Regression | 78.2%    |
| Random Forest       | 82.4%    |
| **XGBoost** ⭐      | **85.1%**|

*XGBoost is used in the Streamlit app*

---

## 💡 Key Features Used for Prediction

1. **VeryActiveMinutes** - High-intensity activity time
2. **FairlyActiveMinutes** - Moderate activity time
3. **LightlyActiveMinutes** - Light activity time
4. **SedentaryMinutes** - Sitting/inactive time
5. **TotalSteps_hourly** - Daily step count
6. **TotalCalories_hourly** - Daily calories burned
7. **ActiveRatio** - Active vs. sedentary ratio
8. **IsWeekend** - Weekend indicator (0 or 1)

**Note:** Despite "_hourly" in the names, TotalSteps_hourly and TotalCalories_hourly contain **daily totals**, not hourly averages.

---

## 🆘 Need Help?

1. **Check the troubleshooting section** above
2. **Verify your Python version:** `python3 --version` (should be 3.9+)
3. **Make sure virtual environment is activated** (you should see `(.venv)` in terminal)
4. **Try reinstalling dependencies:** `pip install -r requirements.txt --force-reinstall`

---

## ✅ Quick Verification

After setup, verify everything works:

```bash
# 1. Check Python version
python3 --version  # Should be 3.9 or higher

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Verify packages installed
pip list | grep -E "streamlit|pandas|xgboost|sklearn|shap"

# 4. Test Streamlit app
streamlit run app.py
```

If the app opens and you can make predictions, **you're all set!** 🎉

---

**Happy coding! 🚀**
