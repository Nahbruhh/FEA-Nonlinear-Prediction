import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os

# #deploying debug
# print("Current Working Directory:", os.getcwd())
# print("Files in the current directory:", os.listdir())
# print("Loading model from:", os.path.abspath('../models/best_rf_model.joblib'))
# print("Loading scaler_X from:", os.path.abspath('../models/scaler_X_rf.joblib'))
# print("Loading scaler_y from:", os.path.abspath('../models/scaler_y_rf.joblib'))
# model_path = '../models/best_rf_model.joblib'
# if os.access(model_path, os.R_OK):
#     print(f"File {model_path} is readable.")
# else:
#     print(f"File {model_path} is not readable.")


base_dir = os.path.dirname(os.path.abspath(__file__))


print("Current dir:", os.getcwd())
print("App dir:", base_dir)
print("Files in app dir:", os.listdir(base_dir))

rf_model = joblib.load(os.path.join(base_dir,'best_rf_model.joblib'))
rf_scaler_X = joblib.load(os.path.join(base_dir,'scaler_X_rf.joblib'))
rf_scaler_y = joblib.load(os.path.join(base_dir,'scaler_y_rf.joblib'))

xgb_model = joblib.load(os.path.join(base_dir,'best_xgb_model.joblib'))
xgb_scaler_X = joblib.load(os.path.join(base_dir,'scaler_X_xgb.joblib'))
xgb_scaler_y = joblib.load(os.path.join(base_dir,'scaler_y_xgb.joblib'))

st.set_page_config(page_title="Nonlinear FEA Prediction App", page_icon="🧬", layout="wide")
st.title('Nonlinear FEA Prediction App')





# 340.1523409	376.1708725	0.001700776
if 'history_log' not in st.session_state:
    st.session_state.history_log = []


feature_names = ['Sigma_linear_VM', 'Sigma_linear_MaxP', 'Epsilon_linear_Equiv']

col1,_, col2 = st.columns([1,0.5,3])
with col1:
    model_choice = st.selectbox("Select Model:",("Random Forest","XGBoost"))
    sigma_linear_vm = st.number_input('Linear von Mises Stress (MPa)', value = 340.1523409, format="%.6f")
    sigma_linear_maxp = st.number_input('Linear Maximum Principal Stress (MPa)', value = 376.1708725, format="%.6f")
    epsilon_linear_equiv = st.number_input('Linear Equivalent Strain', value = 0.001700776, format="%.6f")


with col2:
    if st.button('Predict',icon="🚀"):
        input_data = input_data = pd.DataFrame([[sigma_linear_vm, sigma_linear_maxp, epsilon_linear_equiv]],
                                                columns=feature_names
                                                )

        if model_choice == 'Random Forest':
            input_data_scaled = rf_scaler_X.transform(input_data)
            prediction_scaled = rf_model.predict(input_data_scaled)
            prediction = rf_scaler_y.inverse_transform(prediction_scaled)
        elif model_choice == 'XGBoost':
            input_data_scaled = xgb_scaler_X.transform(input_data)
            prediction_scaled = xgb_model.predict(input_data_scaled)
            prediction = xgb_scaler_y.inverse_transform(prediction_scaled)
        else:
            st.warning("Please select a model.")
            prediction = None
    
        if prediction is not None:
            st.subheader('Predicted Nonlinear Results:')
            st.write(f"Nonlinear VM Stress = {prediction[0, 0]:.6f} MPa")
            st.write(f"Nonlinear MaxP Stress = {prediction[0, 1]:.6f} MPa")
            st.write(f"Nonlinear Total Strain = {prediction[0, 2]:.6f} mm/mm")
            st.write(f"Nonlinear Plastic Strain = {prediction[0, 3]:.6f} mm/mm")
            st.write(f"Nonlinear Elastic Strain = {prediction[0, 4]:.6f} mm/mm")

            # Add to history log with timestamp (keep max 10 entries)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = {'timestamp': timestamp,
                        'model': model_choice,
                        'inputs': [sigma_linear_vm, sigma_linear_maxp, epsilon_linear_equiv],
                        'outputs': prediction.flatten().tolist()
                        }
            st.session_state.history_log.append(log_entry)
            if len(st.session_state.history_log) > 10:  # Keep only the most recent 10 entries
                st.session_state.history_log.pop(0)





st.subheader('History Log (Recent 10):')
col1,_, col2 = st.columns([1,0.5,3])
with col1:
    if st.button(f'Clear History',
                icon="🗑️",
                use_container_width=True,
                help="Click to erase all the history log."):
        st.session_state.history_log.clear()
        st.write("History log cleared.")

    st.download_button(
        label="Download History Log",
        icon="⬇️",
        use_container_width=True,
        help="Click to download the history log as a CSV file (up to 10 recent entries).",
        data=pd.DataFrame([
            {
                'Timestamp': entry['timestamp'],
                'Model': entry['model'],
                'Input_VM_Stress': entry['inputs'][0],
                'Input_MaxP_Stress': entry['inputs'][1],
                'Input_Equiv_Strain': entry['inputs'][2],
                'Output_Nonlinear_VM_Stress': entry['outputs'][0],
                'Output_Nonlinear_MaxP_Stress': entry['outputs'][1],
                'Output_Nonlinear_Total_Strain': entry['outputs'][2],
                'Output_Nonlinear_Plastic_Strain': entry['outputs'][3],
                'Output_Nonlinear_Elastic_Strain': entry['outputs'][4]
            } for entry in st.session_state.history_log
        ]).to_csv(index=False),
        file_name='history_log.csv',
        mime='text/csv'
    )

if st.session_state.history_log:
    for i, entry in enumerate(st.session_state.history_log[::-1],1):
        st.write(f"\n{i}. Time: {entry['timestamp']}")
        st.write(f"Model: {entry['model']}")
        st.write(f"Inputs: VM Stress={entry['inputs'][0]:.6f}, MaxP Stress={entry['inputs'][1]:.6f}, Equiv Strain={entry['inputs'][2]:.6f}")
        st.write(f"Outputs:")
        st.write(f"Nonlinear VM Stress = {entry['outputs'][0]:.6f} MPa")
        st.write(f"Nonlinear MaxP Stress = {entry['outputs'][1]:.6f} MPa")
        st.write(f"Nonlinear Total Strain = {entry['outputs'][2]:.6f} mm/mm")
        st.write(f"Nonlinear Plastic Strain = {entry['outputs'][3]:.6f} mm/mm")
        st.write(f"Nonlinear Elastic Strain = {entry['outputs'][4]:.6f} mm/mm")

else:
    st.write("No entries in the history log.")

