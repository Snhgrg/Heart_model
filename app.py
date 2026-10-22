import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- PAGE CONFIG & CUSTOM DARK PURPLE-BLUE THEME ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #121026;
        color: #f0f2f6;
    }
    section[data-testid="stSidebar"] {
        background-color: #1b1638;
    }
    label, .stMarkdown, p {
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- LOAD DATA, TRAIN, & COMPUTE ACCURACY ---
@st.cache_resource
def load_processed_data():
    df = pd.read_csv("new1.csv")
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    continuous_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    
    # Fit scaler for user inputs
    scaler = StandardScaler()
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
    
    # Split data to calculate a realistic model accuracy score (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    # Calculate accuracy on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, list(X.columns), continuous_cols, accuracy

model, scaler, expected_columns, continuous_cols, model_accuracy = load_processed_data()

# --- STREAMLIT UI ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter your health details below to evaluate risk using your trained pipeline:")

# Display the model accuracy nicely in the app
st.metric(label="Model Accuracy (Test Set)", value=f"{model_accuracy * 100:.2f}%")

st.markdown("Enter your health details below to evaluate risk using your trained pipeline:")
st.markdown("---")

age = st.slider("Age", 20, 100, 50)

sex_str = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_str == "Male" else 0

cp_str = st.selectbox("Chest Pain Type", ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "TA (Typical Angina)", "ASY (Asymptomatic)"])
cp_mapping = {
    "ATA (Atypical Angina)": 0, 
    "NAP (Non-Anginal Pain)": 1, 
    "TA (Typical Angina)": 2, 
    "ASY (Asymptomatic)": 3
}
chest_pain = cp_mapping[cp_str]

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

fasting_bs_str = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
fasting_bs = 1 if fasting_bs_str == "Yes" else 0

ecg_str = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
ecg_mapping = {"Normal": 0, "ST": 1, "LVH": 2}
resting_ecg = ecg_mapping[ecg_str]

max_hr = st.slider("Max Heart Rate", 60, 220, 150)

angina_str = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
exercise_angina = 1 if angina_str == "Yes" else 0

oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

slope_str = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
slope_mapping = {"Up": 0, "Flat": 1, "Down": 2}
st_slope = slope_mapping[slope_str]

# --- PREDICTION LOGIC ---
if st.button("Predict Risk", type="primary"):
    raw_input = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([raw_input])
    input_df = input_df[expected_columns]
    
    # Scale user input continuous values using the live-fitted scaler
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Warning: High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")