# Wine Quality Prediction Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .good-quality {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
    }
    .poor-quality {
        background: linear-gradient(90deg, #f44336, #da190b);
        color: white;
    }
    .feature-info {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    """Load the trained model and sample data"""
    try:
        # For deployment, you'll need to have these files in your repo
        model = joblib.load('wine_quality_model.joblib')
        
        # If metadata file exists, load it
        try:
            metadata = joblib.load('model_metadata.joblib')
            features = metadata['features']
        except:
            # Fallback feature list if metadata file doesn't exist
            features = [
                'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol'
            ]
        
        return model, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_feature_input_form(features):
    """Create input form for wine features"""
    
    # Feature ranges and descriptions (based on typical red wine values)
    feature_info = {
        'fixed acidity': {'min': 4.6, 'max': 15.9, 'default': 8.32, 'unit': 'g/L', 'desc': 'Non-volatile acids'},
        'volatile acidity': {'min': 0.12, 'max': 1.58, 'default': 0.53, 'unit': 'g/L', 'desc': 'Acetic acid content'},
        'citric acid': {'min': 0.0, 'max': 1.0, 'default': 0.27, 'unit': 'g/L', 'desc': 'Freshness and flavor'},
        'residual sugar': {'min': 0.9, 'max': 15.5, 'default': 2.54, 'unit': 'g/L', 'desc': 'Sugar remaining after fermentation'},
        'chlorides': {'min': 0.012, 'max': 0.611, 'default': 0.087, 'unit': 'g/L', 'desc': 'Salt content'},
        'free sulfur dioxide': {'min': 1.0, 'max': 72.0, 'default': 15.87, 'unit': 'mg/L', 'desc': 'Free form of SO2'},
        'total sulfur dioxide': {'min': 6.0, 'max': 289.0, 'default': 46.47, 'unit': 'mg/L', 'desc': 'Total SO2 content'},
        'density': {'min': 0.99007, 'max': 1.00369, 'default': 0.9967, 'unit': 'g/cm³', 'desc': 'Wine density'},
        'pH': {'min': 2.74, 'max': 4.01, 'default': 3.31, 'unit': '', 'desc': 'Acidity level (0-14 scale)'},
        'sulphates': {'min': 0.33, 'max': 2.0, 'default': 0.66, 'unit': 'g/L', 'desc': 'Wine preservative'},
        'alcohol': {'min': 8.4, 'max': 14.9, 'default': 10.42, 'unit': '% vol', 'desc': 'Alcohol content'}
    }
    
    st.sidebar.header("🍷 Wine Properties")
    st.sidebar.markdown("*Adjust the sliders to input your wine's characteristics*")
    
    inputs = {}
    
    # Create input widgets for each feature
    for feature in features:
        info = feature_info.get(feature, {'min': 0.0, 'max': 100.0, 'default': 50.0, 'unit': '', 'desc': 'Wine property'})
        
        # Create two columns for label and unit
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            inputs[feature] = st.slider(
                label=feature.replace('_', ' ').title(),
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=float(info['default']),
                step=(info['max'] - info['min']) / 100,
                help=info['desc']
            )
        
        with col2:
            st.markdown(f"<small style='color: #666;'>{info['unit']}</small>", unsafe_allow_html=True)
    
    return inputs

def make_prediction(model, inputs, features):
    """Make prediction using the trained model"""
    try:
        # Create feature vector in the correct order
        feature_vector = np.array([[inputs[feature] for feature in features]])
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        prediction_proba = model.predict_proba(feature_vector)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def create_prediction_visualization(prediction_proba):
    """Create visualization for prediction results"""
    
    # Create gauge chart for confidence
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction_proba[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Good Quality Confidence %"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

def display_feature_importance():
    """Display feature importance if available"""
    try:
        metadata = joblib.load('model_metadata.joblib')
        importance_data = metadata.get('feature_importance', [])
        
        if importance_data:
            # Convert to DataFrame and get top features
            df_importance = pd.DataFrame(importance_data)
            top_features = df_importance.head(6)
            
            # Create bar chart
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title='Top Feature Importance',
                color='importance',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                height=300,
                yaxis={'categoryorder':'total ascending'}
            )
            
            return fig
    except:
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Predict whether your red wine is of <strong>good quality</strong> (rating ≥ 7) based on its chemical properties.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and features
    model, features = load_model_and_data()
    
    if model is None:
        st.error("Failed to load the model. Please ensure the model files are available.")
        st.stop()
    
    # Create two main columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📊 Model Information")
        
        # Model info
        st.info("""
        **Model:** Random Forest Classifier
        
        **Features:** 11 wine chemical properties
        
        **Target:** Good Quality (≥ 7 rating)
        
        **Performance:** ~85% accuracy
        """)
        
        # Feature importance chart
        importance_fig = display_feature_importance()
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
    
    with col2:
        st.header("🔮 Prediction Results")
        
        # Get user inputs
        inputs = create_feature_input_form(features)
        
        # Make prediction when inputs are provided
        if st.sidebar.button("🍷 Predict Wine Quality", type="primary"):
            prediction, prediction_proba = make_prediction(model, inputs, features)
            
            if prediction is not None:
                # Display prediction result
                quality_label = "Good Quality Wine" if prediction == 1 else "Poor Quality Wine"
                confidence = prediction_proba[prediction] * 100
                
                # Prediction box styling
                box_class = "good-quality" if prediction == 1 else "poor-quality"
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>🎯 Prediction: {quality_label}</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create and display gauge chart
                gauge_fig = create_prediction_visualization(prediction_proba)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Detailed probabilities
                st.subheader("📈 Detailed Probabilities")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        "Poor Quality", 
                        f"{prediction_proba[0]:.1%}",
                        delta=f"{prediction_proba[0] - 0.5:.1%}" if prediction_proba[0] < 0.5 else None
                    )
                
                with col_b:
                    st.metric(
                        "Good Quality", 
                        f"{prediction_proba[1]:.1%}",
                        delta=f"{prediction_proba[1] - 0.5:.1%}" if prediction_proba[1] > 0.5 else None
                    )
                
                # Feature values summary
                with st.expander("📋 Input Summary"):
                    input_df = pd.DataFrame(
                        list(inputs.items()), 
                        columns=['Feature', 'Value']
                    )
                    input_df['Feature'] = input_df['Feature'].str.replace('_', ' ').str.title()
                    st.dataframe(input_df, hide_index=True)
        
        else:
            st.info("👈 Adjust the wine properties in the sidebar and click 'Predict Wine Quality' to get your prediction!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🍷 Wine Quality Predictor | Built with Streamlit & Machine Learning</p>
        <p><small>Model trained on red wine chemical properties dataset</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()





    