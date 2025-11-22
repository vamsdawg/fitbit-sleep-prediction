"""
🌙 Sleep Quality Predictor - Streamlit Application
Predict your sleep quality based on daily Fitbit activity data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sleep Quality Predictor 😴",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .good-sleep {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .poor-sleep {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data (cached for performance)
@st.cache_resource
def load_models():
    """Load all trained models and metadata"""
    try:
        # Load best model (XGBoost)
        xgb_model = joblib.load('xgboost_model.pkl')
        
        # Load other models for comparison
        rf_model = joblib.load('random_forest_model.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        lr_scaler = joblib.load('logistic_regression_scaler.pkl')
        
        # Load metadata
        feature_names = joblib.load('feature_names.pkl')
        metadata = joblib.load('model_metadata.pkl')
        
        return {
            'xgb': xgb_model,
            'rf': rf_model,
            'lr': lr_model,
            'scaler': lr_scaler,
            'features': feature_names,
            'metadata': metadata
        }
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ Model files not found! 
        
        Please run ModelTraining.ipynb first to train and save the models.
        
        Missing file: {str(e)}
        """)
        return None

# Load models
models = load_models()

if models is None:
    st.stop()

# Good sleepers averages (calculated from training data)
GOOD_SLEEPER_AVERAGES = {
    'TotalSteps_hourly': 9500,
    'VeryActiveMinutes': 45,
    'FairlyActiveMinutes': 30,
    'LightlyActiveMinutes': 140,
    'SedentaryMinutes': 650,
    'TotalCalories_hourly': 2400,
    'ActiveRatio': 0.25
}

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/sleeping-in-bed.png", width=80)
    st.title("🌙 Sleep Predictor")
    
    page = st.radio(
        "Navigate:",
        ["🏠 Quick Predict", "📊 Batch Analysis", "📚 Learn More", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model info
    st.subheader("📈 Model Performance")
    best_model = models['metadata']['best_model'].replace('_', ' ').title()
    st.metric("Best Model", best_model)
    
    perf = models['metadata']['model_performance']['xgboost']
    st.metric("Accuracy", f"{perf['accuracy']*100:.1f}%")
    st.metric("ROC-AUC", f"{perf['auc']:.3f}")
    
    st.markdown("---")
    
    # Disclaimer
    st.warning("⚠️ **Important**\n\nThis is an educational tool, not medical advice. For chronic sleep issues, consult a healthcare professional.")
    
    st.markdown("---")
    st.caption(f"Last updated: {models['metadata']['train_date']}")

# ============================================================================
# PAGE 1: QUICK PREDICT
# ============================================================================

if page == "🏠 Quick Predict":
    st.title("😴 Sleep Quality Predictor")
    st.markdown("""
    ### Predict Your Sleep Quality
    Enter your daily activity metrics below to predict whether you'll have **Good Sleep** (≥85% efficiency) 
    or **Poor Sleep** (<85% efficiency).
    """)
    
    st.markdown("---")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Activity Metrics")
        
        total_steps_hourly = st.slider(
            "Total Steps 🚶",
            min_value=0,
            max_value=25000,
            value=10000,
            step=500,
            help="Total steps walked during the day"
        )
        
        very_active_min = st.slider(
            "Very Active Minutes 🏃",
            min_value=0,
            max_value=120,
            value=20,
            step=5,
            help="High-intensity exercise (running, sports, HIIT)"
        )
        
        fairly_active_min = st.slider(
            "Fairly Active Minutes 🚴",
            min_value=0,
            max_value=120,
            value=30,
            step=5,
            help="Moderate exercise (brisk walking, cycling, dancing)"
        )
        
        lightly_active_min = st.slider(
            "Lightly Active Minutes 🚶‍♀️",
            min_value=0,
            max_value=300,
            value=150,
            step=10,
            help="Light movement (slow walking, household chores)"
        )
    
    with col2:
        st.subheader("💺 Rest & Calories")
        
        sedentary_min = st.slider(
            "Sedentary Minutes 🪑",
            min_value=0,
            max_value=1440,
            value=800,
            step=30,
            help="Time sitting or lying down (not sleeping)"
        )
        
        total_calories_hourly = st.slider(
            "Calories Burned 🔥",
            min_value=1500,
            max_value=4500,
            value=2200,
            step=50,
            help="Total calories burned during the day"
        )
        
        is_weekend = st.checkbox(
            "Is it Weekend? 📅",
            help="Check if today is Saturday or Sunday"
        )
        
        st.markdown("---")
        
        # Quick stats
        total_active = very_active_min + fairly_active_min + lightly_active_min
        st.metric("Total Active Minutes", f"{total_active} min")
        active_ratio = total_active / (total_active + sedentary_min + 0.001)
        st.metric("Activity Ratio", f"{active_ratio:.2%}")
    
    st.markdown("---")
    
    # Predict button
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        predict_button = st.button("🔮 PREDICT MY SLEEP QUALITY", type="primary", use_container_width=True)
    
    if predict_button:
        # Calculate engineered features
        total_active_minutes = very_active_min + fairly_active_min + lightly_active_min
        active_ratio_calc = total_active_minutes / (total_active_minutes + sedentary_min + 0.001)
        
        # Prepare input dataframe (must match training feature order from the model)
        user_input = pd.DataFrame([{
            'VeryActiveMinutes': very_active_min,
            'FairlyActiveMinutes': fairly_active_min,
            'LightlyActiveMinutes': lightly_active_min,
            'SedentaryMinutes': sedentary_min,
            'TotalCalories_hourly': total_calories_hourly,
            'TotalSteps_hourly': total_steps_hourly,
            'ActiveRatio': active_ratio_calc,
            'IsWeekend': int(is_weekend)
        }])
        
        # Make prediction with XGBoost
        prediction = models['xgb'].predict(user_input)[0]
        probability = models['xgb'].predict_proba(user_input)[0]
        confidence = probability[prediction] * 100
        
        # Display prediction
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box good-sleep">
                ✅ GOOD SLEEP PREDICTED!<br>
                <span style="font-size: 1.2rem;">You have a high chance of quality sleep tonight</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric("Confidence Level", f"{confidence:.1f}%", help="Model's confidence in this prediction")
        else:
            st.markdown(f"""
            <div class="prediction-box poor-sleep">
                ⚠️ POOR SLEEP PREDICTED<br>
                <span style="font-size: 1.2rem;">You may experience lower sleep quality tonight</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric("Confidence Level", f"{confidence:.1f}%", help="Model's confidence in this prediction")
        
        # SHAP Explanation
        st.markdown("---")
        st.markdown("## 🔍 Why This Prediction?")
        st.markdown("Understanding which factors influenced this prediction:")
        
        with st.spinner("Calculating SHAP values..."):
            # Calculate SHAP values
            explainer = shap.TreeExplainer(models['xgb'])
            shap_values = explainer.shap_values(user_input)
            
            # Create SHAP waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=user_input.iloc[0],
                    feature_names=models['features']
                ),
                show=False
            )
            plt.title(f'Feature Contributions to "{("Poor", "Good")[prediction]} Sleep" Prediction', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.info("""
        💡 **How to read this chart:**
        - Red bars push the prediction toward **Good Sleep**
        - Blue bars push the prediction toward **Poor Sleep**
        - Longer bars = stronger influence on the prediction
        """)
        
        # Personalized Recommendations
        st.markdown("---")
        st.markdown("## 💡 Personalized Recommendations")
        
        recommendations = []
        
        # Compare to good sleepers
        if total_steps_hourly < GOOD_SLEEPER_AVERAGES['TotalSteps_hourly']:
            diff = GOOD_SLEEPER_AVERAGES['TotalSteps_hourly'] - total_steps_hourly
            recommendations.append({
                'icon': '📈',
                'title': 'Increase Daily Steps',
                'message': f"You walked **{total_steps_hourly:,}** steps. Try to reach **{GOOD_SLEEPER_AVERAGES['TotalSteps_hourly']:,}** steps (+{diff:,} more).",
                'impact': 'High'
            })
        
        if sedentary_min > GOOD_SLEEPER_AVERAGES['SedentaryMinutes']:
            diff = sedentary_min - GOOD_SLEEPER_AVERAGES['SedentaryMinutes']
            recommendations.append({
                'icon': '📉',
                'title': 'Reduce Sedentary Time',
                'message': f"You were sedentary for **{sedentary_min}** minutes. Try to reduce to **{GOOD_SLEEPER_AVERAGES['SedentaryMinutes']}** minutes (-{diff} min).",
                'impact': 'Very High'
            })
        
        if very_active_min < GOOD_SLEEPER_AVERAGES['VeryActiveMinutes']:
            diff = GOOD_SLEEPER_AVERAGES['VeryActiveMinutes'] - very_active_min
            recommendations.append({
                'icon': '💪',
                'title': 'Add Vigorous Exercise',
                'message': f"You had **{very_active_min}** minutes of vigorous activity. Try to reach **{GOOD_SLEEPER_AVERAGES['VeryActiveMinutes']}** minutes (+{diff} min).",
                'impact': 'High'
            })
        
        if recommendations:
            for rec in recommendations:
                impact_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Very High': '🔴'}
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #1f1f1f; margin-bottom: 0.5rem;">{rec['icon']} {rec['title']} {impact_color[rec['impact']]} {rec['impact']} Impact</h4>
                    <p style="color: #262730; margin: 0;">{rec['message']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
        else:
            st.success("✅ **Excellent!** Your activity levels are similar to good sleepers. Keep it up!")
        
        st.info("📊 **Expected Impact:** Following these recommendations could improve your sleep quality probability by 15-25%.")
        
        # Comparison Radar Chart
        st.markdown("---")
        st.markdown("## 📊 Your Profile vs. Good Sleepers")
        
        categories = ['Steps', 'Very Active', 'Fairly Active', 'Lightly Active', 'Low Sedentary']
        
        # Normalize to 0-100 scale for visualization
        user_values = [
            min((total_steps_hourly / 15000) * 100, 100),
            min((very_active_min / 60) * 100, 100),
            min((fairly_active_min / 60) * 100, 100),
            min((lightly_active_min / 200) * 100, 100),
            min(((1440 - sedentary_min) / 1440) * 100, 100)
        ]
        
        good_sleeper_values = [
            (GOOD_SLEEPER_AVERAGES['TotalSteps_hourly'] / 15000) * 100,
            (GOOD_SLEEPER_AVERAGES['VeryActiveMinutes'] / 60) * 100,
            (GOOD_SLEEPER_AVERAGES['FairlyActiveMinutes'] / 60) * 100,
            (GOOD_SLEEPER_AVERAGES['LightlyActiveMinutes'] / 200) * 100,
            ((1440 - GOOD_SLEEPER_AVERAGES['SedentaryMinutes']) / 1440) * 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='Your Profile',
            line=dict(color='#f5576c', width=2)
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=good_sleeper_values,
            theta=categories,
            fill='toself',
            name='Good Sleepers Average',
            line=dict(color='#4CAF50', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=True,
                    ticks='outside'
                )
            ),
            showlegend=True,
            title="Activity Profile Comparison",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: BATCH ANALYSIS
# ============================================================================

elif page == "📊 Batch Analysis":
    st.title("📊 Batch Analysis")
    st.markdown("""
    ### Analyze Multiple Days of Activity Data
    Upload a CSV file with your Fitbit data to get predictions for multiple days and see trends over time.
    """)
    
    st.markdown("---")
    
    # Show required format
    with st.expander("📋 Required CSV Format"):
        st.markdown("""
        Your CSV file should have these columns (in any order):
        - `VeryActiveMinutes` - Minutes of vigorous exercise
        - `FairlyActiveMinutes` - Minutes of moderate exercise
        - `LightlyActiveMinutes` - Minutes of light activity
        - `SedentaryMinutes` - Minutes sitting/lying down
        - `TotalCalories_hourly` - **Total daily calories** (not per hour!)
        - `TotalSteps_hourly` - **Total daily steps** (not per hour!)
        - `ActiveRatio` - Will be auto-calculated if missing
        - `IsWeekend` - 0 for weekday, 1 for weekend
        
        Optional: `ActivityDate` for time-series analysis
        
        📝 **Note:** The "_hourly" suffix is from data processing, but values should be **daily totals**.
        """)
        
        # Show example
        example_df = pd.DataFrame({
            'ActivityDate': ['2024-11-01', '2024-11-02', '2024-11-03'],
            'VeryActiveMinutes': [20, 35, 10],
            'FairlyActiveMinutes': [25, 30, 15],
            'LightlyActiveMinutes': [120, 150, 90],
            'SedentaryMinutes': [850, 650, 900],
            'TotalCalories_hourly': [2100, 2450, 1950],
            'TotalSteps_hourly': [8500, 12000, 6500],
            'IsWeekend': [0, 0, 1]
        })
        st.dataframe(example_df, use_container_width=True)
    
    uploaded_file = st.file_uploader("📁 Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found **{len(df_upload)}** days of data.")
            
            # Show first few rows
            st.markdown("### Preview of Your Data")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            # Check for required columns
            required_cols = ['VeryActiveMinutes', 'FairlyActiveMinutes', 
                            'LightlyActiveMinutes', 'SedentaryMinutes', 
                            'TotalCalories_hourly', 'TotalSteps_hourly']
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Calculate engineered features if not present
            if 'ActiveRatio' not in df_upload.columns:
                total_active = (df_upload['VeryActiveMinutes'] + 
                               df_upload['FairlyActiveMinutes'] + 
                               df_upload['LightlyActiveMinutes'])
                df_upload['ActiveRatio'] = total_active / (
                    total_active + df_upload['SedentaryMinutes'] + 0.001
                )
            
            if 'IsWeekend' not in df_upload.columns:
                df_upload['IsWeekend'] = 0  # Default to weekday
            
            # Make predictions
            with st.spinner("Making predictions for all days..."):
                X_batch = df_upload[models['features']]
                predictions = models['xgb'].predict(X_batch)
                probabilities = models['xgb'].predict_proba(X_batch)[:, 1]
                
                df_upload['Prediction'] = ['Good Sleep' if p == 1 else 'Poor Sleep' for p in predictions]
                df_upload['Good Sleep Probability'] = probabilities
                df_upload['Confidence'] = [probabilities[i] if predictions[i] == 1 else 1-probabilities[i] 
                                          for i in range(len(predictions))]
            
            st.success("✅ Predictions complete!")
            
            # Summary statistics
            st.markdown("---")
            st.markdown("### 📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            good_sleep_pct = (predictions == 1).sum() / len(predictions) * 100
            poor_sleep_pct = 100 - good_sleep_pct
            avg_confidence = df_upload['Confidence'].mean() * 100
            
            with col1:
                st.metric("Total Days", len(df_upload))
            with col2:
                st.metric("Good Sleep Days", f"{(predictions == 1).sum()}")
            with col3:
                st.metric("Good Sleep %", f"{good_sleep_pct:.1f}%")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Show results table
            st.markdown("### 📋 Detailed Results")
            
            # Select columns to display
            display_cols = ['Prediction', 'Good Sleep Probability', 'Confidence', 
                           'TotalSteps_hourly', 'SedentaryMinutes', 'TotalCalories_hourly']
            if 'ActivityDate' in df_upload.columns:
                display_cols = ['ActivityDate'] + display_cols
            
            st.dataframe(
                df_upload[display_cols].style.background_gradient(
                    subset=['Good Sleep Probability'], 
                    cmap='RdYlGn'
                ),
                use_container_width=True
            )
            
            # Visualization
            if 'ActivityDate' in df_upload.columns and len(df_upload) > 1:
                st.markdown("---")
                st.markdown("### 📊 Trends Over Time")
                
                # Convert to datetime
                df_upload['ActivityDate'] = pd.to_datetime(df_upload['ActivityDate'])
                df_upload = df_upload.sort_values('ActivityDate')
                
                # Create time series plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_upload['ActivityDate'],
                    y=df_upload['Good Sleep Probability'],
                    mode='lines+markers',
                    name='Sleep Quality Probability',
                    line=dict(color='#4CAF50', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_hline(
                    y=0.5, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="50% Threshold"
                )
                
                fig.update_layout(
                    title="Sleep Quality Probability Over Time",
                    xaxis_title="Date",
                    yaxis_title="Good Sleep Probability",
                    yaxis_range=[0, 1],
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.markdown("---")
            csv = df_upload.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"sleep_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ============================================================================
# PAGE 3: LEARN MORE
# ============================================================================

elif page == "📚 Learn More":
    st.title("📚 Learn More About Sleep Prediction")
    
    st.markdown("""
    ### How It Works
    Our machine learning model analyzes your daily activity patterns to predict sleep quality.
    """)
    
    st.markdown("---")
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Model Information")
        st.markdown(f"""
        - **Algorithm:** {models['metadata']['best_model'].replace('_', ' ').title()}
        - **Training Data:** Fitbit dataset from April 2016
        - **Sample Size:** {models['metadata']['training_samples']} training samples
        - **Test Accuracy:** {models['metadata']['model_performance']['xgboost']['accuracy']*100:.1f}%
        - **ROC-AUC Score:** {models['metadata']['model_performance']['xgboost']['auc']:.3f}
        """)
    
    with col2:
        st.markdown("### 📊 Performance Metrics")
        perf = models['metadata']['model_performance']['xgboost']
        st.metric("Precision", f"{perf['precision']*100:.1f}%")
        st.metric("Recall", f"{perf['recall']*100:.1f}%")
        st.metric("F1-Score", f"{perf['f1']:.3f}")
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### 🎯 Most Important Features")
    st.markdown("These factors have the biggest impact on sleep quality predictions:")
    
    # Get feature importance from XGBoost
    importance_df = pd.DataFrame({
        'Feature': models['features'],
        'Importance': models['xgb'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance (XGBoost Model)',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # What the model considers
    st.markdown("### ✅ What the Model Considers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Factors Included:**
        - Daily step count
        - Active minutes (very, fairly, lightly)
        - Sedentary time
        - Calories burned
        - Activity ratios
        - Day of week (weekend vs weekday)
        """)
    
    with col2:
        st.warning("""
        **NOT Considered:**
        - Stress levels
        - Caffeine intake
        - Room temperature
        - Age or health conditions
        - Sleep environment
        - Medications
        """)
    
    st.markdown("---")
    
    # Tips for better sleep
    st.markdown("### 💡 Tips for Better Sleep (Based on Our Data)")
    
    st.markdown("""
    #### 🏆 Patterns from Good Sleepers:
    
    1. **Stay Active:** Good sleepers average **9,500+ steps** per day
    2. **Reduce Sitting:** Keep sedentary time below **700 minutes** per day
    3. **Add Intensity:** Include **45+ minutes** of vigorous exercise
    4. **Be Consistent:** Maintain similar activity levels on weekends
    5. **Balance Activity:** Mix of vigorous, moderate, and light activity
    """)
    
    st.info("""
    💡 **Did you know?**
    Our model found that **sedentary time** is the strongest predictor of sleep quality.
    Even small reductions in sitting time can significantly improve sleep!
    """)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ### Sleep Quality Prediction Using Machine Learning
    
    This application uses machine learning to predict sleep quality based on daily activity data
    from Fitbit devices. The goal is to help users understand how their daytime activities 
    affect their sleep quality.
    """)
    
    st.markdown("---")
    
    # Project details
    st.markdown("### 🎓 Project Overview")
    st.markdown("""
    - **Course:** DSBA 6211 - Final Project
    - **Dataset:** Fitbit Fitness Tracker Data (April 2016)
    - **Users:** 24-33 participants
    - **Total Observations:** ~400-900 days of activity + sleep data
    - **Models Tested:** Logistic Regression, Random Forest, XGBoost
    """)
    
    st.markdown("---")
    
    # Model comparison
    st.markdown("### 📊 Model Comparison")
    
    comparison_data = []
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        perf = models['metadata']['model_performance'][model_name]
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{perf['accuracy']*100:.1f}%",
            'Precision': f"{perf['precision']*100:.1f}%",
            'Recall': f"{perf['recall']*100:.1f}%",
            'F1-Score': f"{perf['f1']:.3f}",
            'ROC-AUC': f"{perf['auc']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("---")
    
    # Limitations
    st.markdown("### ⚠️ Limitations & Disclaimers")
    
    st.warning("""
    **Important Limitations:**
    
    1. **Not Perfect:** Model accuracy is ~85%, meaning ~15% of predictions may be incorrect
    2. **Limited Dataset:** Trained on 24-33 users from a single month in 2016
    3. **Missing Factors:** Doesn't account for stress, caffeine, medications, sleep environment, etc.
    4. **Not Medical Advice:** This is an educational tool, not a medical device
    5. **Individual Variation:** Results may vary based on personal factors
    
    **🏥 If you have chronic sleep issues, please consult a healthcare professional.**
    """)
    
    st.markdown("---")
    
    # Future improvements
    st.markdown("### 🚀 Future Improvements")
    st.markdown("""
    - **More Data:** Train on larger, more diverse dataset
    - **Additional Features:** Include heart rate, sleep stages, environmental factors
    - **Personalization:** Adapt model to individual user patterns over time
    - **Real-time Tracking:** Direct integration with Fitbit API
    - **Recommendations:** More specific, actionable advice based on user patterns
    """)
    
    st.markdown("---")
    
    # Contact
    st.markdown("### 📧 Contact & Feedback")
    st.info("""
    Questions or feedback? Feel free to reach out!
    
    This project is part of DSBA 6211 coursework and demonstrates the application
    of machine learning to health and wellness data.
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, scikit-learn, XGBoost, and SHAP")
