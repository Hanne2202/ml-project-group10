import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import os
import sys
import time

# Đảm bảo có thể import các module tự định nghĩa trong project
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from modules.deep_learning import MLP
    from modules.mappings import reverse_mapping
except ImportError:
    st.error("Cannot find modules. Please make sure to run this app at the root directory of the project.")

# ---------------------------------------------------------
# 1. KHỞI TẠO CÁC GIÁ TRỊ MẶC ĐỊNH
# ---------------------------------------------------------
# Dữ liệu Categorical
CATEGORIES = {
    'workclass': ['Private', 'Local Government', 'Self-Employed (Not Incorporated)', 'Federal Government', 'State Government', 'Self-Employed (Incorporated)', 'Without Pay'],
    'marital-status': ['Never Married', 'Married (Civilian Spouse)', 'Widowed', 'Separated', 'Divorced', 'Married (Spouse Absent)', 'Married (Armed Forces Spouse)'],
    'occupation': ['Machine Operator / Inspector', 'Farming / Fishing', 'Protective Services', 'Other Service', 'Professional / Specialty', 'Craft / Repair', 'Admin / Clerical', 'Executive / Managerial', 'Technical Support', 'Sales', 'Private House Service', 'Transport / Moving', 'Handlers / Cleaners', 'Armed Forces'],
    'relationship': ['Own Child', 'Husband', 'Not in Family', 'Unmarried', 'Wife', 'Other Relative'],
    'race': ['Black / African American', 'White', 'Other', 'American Indian / Eskimo', 'Asian / Pacific Islander'],
    'gender': ['Male', 'Female'],
    'native-country': ['United States', 'Peru', 'Guatemala', 'Mexico', 'Dominican Republic', 'Ireland', 'Germany', 'Philippines', 'Thailand', 'Haiti', 'El Salvador', 'Puerto Rico', 'Vietnam', 'South Korea', 'Colombia', 'Japan', 'India', 'Cambodia', 'Poland', 'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Nicaragua', 'Honduras', 'Iran', 'Scotland', 'Jamaica', 'Ecuador', 'Yugoslavia', 'Hungary', 'Hong Kong', 'Greece', 'Trinidad & Tobago', 'Outlying US (Guam, USVI, etc.)', 'France', 'Holland / Netherlands']
}

# ---------------------------------------------------------
# 2. GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Adult Income Prediction", layout="wide")
st.title("💸 Adult Income Prediction App")
st.markdown("Fill in or select the information below to let the deep learning model infer whether the income exceeds **$50K/year**.")

# Tạo form nhập liệu
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal & Demographic Information")
        age = st.slider("Age", min_value=17, max_value=90, value=39) # Default: mean rounded
        gender = st.selectbox("Gender", CATEGORIES['gender'], index=CATEGORIES['gender'].index('Male')) # Default: Male
        race = st.selectbox("Race", CATEGORIES['race'], index=CATEGORIES['race'].index('White'))
        native_country = st.selectbox("Native Country", CATEGORIES['native-country'], index=CATEGORIES['native-country'].index('United States'))
        marital_status = st.selectbox("Marital Status", CATEGORIES['marital-status'], index=CATEGORIES['marital-status'].index('Married (Civilian Spouse)'))
        relationship = st.selectbox("Relationship", CATEGORIES['relationship'], index=CATEGORIES['relationship'].index('Husband'))
        
    with col2:
        st.subheader("Work & Education Information")
        workclass = st.selectbox("Work Class", CATEGORIES['workclass'], index=CATEGORIES['workclass'].index('Private'))
        educational_num = st.number_input("Educational Num", min_value=1, max_value=16, value=10)
        occupation = st.selectbox("Occupation", CATEGORIES['occupation'], index=CATEGORIES['occupation'].index('Professional / Specialty'))
        
        st.subheader("Financial Information")
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=1101)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=4356, value=89)
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99, value=41)
        
    submit_button = st.form_submit_button(label="PREDICT INCOME", type="primary")

# ---------------------------------------------------------
# 3. XỬ LÝ VÀ DỰ ĐOÁN
# ---------------------------------------------------------
if submit_button:
    # Gom dữ liệu thành 1 dòng (DataFrame)
    input_data = pd.DataFrame([{
        'age': reverse_mapping.get(age, age),
        'workclass': reverse_mapping.get(workclass, workclass),
        'educational-num': educational_num,
        'marital-status': reverse_mapping.get(marital_status, marital_status),
        'occupation': reverse_mapping.get(occupation, occupation),
        'relationship': reverse_mapping.get(relationship, relationship),
        'race': reverse_mapping.get(race, race),
        'gender': reverse_mapping.get(gender, gender),
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': reverse_mapping.get(native_country, native_country)
    }])
    
    # st.write("### Input Data:")
    # st.dataframe(input_data)
    
    # Đọc model tốt nhất từ thư mục cấu hình
    model_dir = 'models'
    type_file_path = os.path.join(model_dir, 'best_model_type.txt')
    
    if not os.path.exists(type_file_path):
        st.error(f"Configuration file {type_file_path} not found! Please train the model in the 04_deep_learning notebook first.")
        st.stop()
        
    with open(type_file_path, 'r') as f:
        best_model_type = f.read().strip()
        
    # st.info(f"Initializing inference predictor using structure: **{best_model_type}** (Best model from previous training).")
    
    try:
        # Preprocessing căn bản dựa vào loại model
        if best_model_type == 'MLP':
            # 1. Load objects
            scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
            ohe_encoder = joblib.load(os.path.join(model_dir, 'ohe_encoder.joblib'))
                
            # Tách biến
            num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
            
            # scale num_cols
            X_num_df = input_data[num_cols].reset_index(drop=True)
            X_num_scaled = scaler.transform(X_num_df)
            X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=num_cols)
            
            # encode cat_cols
            X_cat_ohe = ohe_encoder.transform(input_data[cat_cols])
            cat_ohe_cols = ohe_encoder.get_feature_names_out(cat_cols)
            X_cat_df = pd.DataFrame(X_cat_ohe, columns=cat_ohe_cols)
            
            # combine và chuyển sang tensor
            X_combined = pd.concat([X_num_scaled_df, X_cat_df], axis=1)
            X_tensor = torch.tensor(X_combined.values, dtype=torch.float32)
            
            # 2. Định nghĩa cấu trúc và Load Weights 
            # Giả định cấu trúc [256, 128, 64] -> Nếu Optuna đổi config thì bạn hãy update thông số hidden layers tương thích ở đây
            # hoặc lưu Hyperparams ra YAML rồi apply. Ở đây ta khởi tạo MLP theo params mặc định tốt nhất 
            # *LƯU Ý*: input_dim phải khớp với X_tensor.shape[1]
            input_dim = X_tensor.shape[1]
            
            # Khởi tạo mô hình
            model = MLP(input_dim=input_dim, hidden_dims=[320, 448, 32], dropout=0.35) 
            
            # Load weights
            model.load_state_dict(torch.load(os.path.join(model_dir, 'best_dl_model.pth'), map_location=torch.device('cpu')))
            model.eval()
            
            # Predict
            with torch.no_grad():
                logits = model(X_tensor)
                proba = torch.sigmoid(logits).item()
                prediction = int(proba >= 0.5)
                
        elif best_model_type == 'TabNet':
            from pytorch_tabnet.tab_model import TabNetClassifier
            
            # 1. Load prep obj
            prep_tabnet = joblib.load(os.path.join(model_dir, 'prep_tabnet.joblib'))
            
            # Cần map các label từ cat_maps của prep_tabnet 
            cat_maps = prep_tabnet.get('cat_maps', {})
            numerical_cols = prep_tabnet.get('num_features', ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'])
            categorical_cols = prep_tabnet.get('cat_features', ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'])
            ordered_features = prep_tabnet.get('ordered_features', numerical_cols + categorical_cols)
            
            # Áp dụng binning nếu model TabNet được train có dùng binning
            binning_info = prep_tabnet.get('binning_info', {})
            for col, bins in binning_info.items():
                if col in input_data.columns:
                    new_col = f'{col}_bin'
                    input_data[new_col] = pd.cut(
                        input_data[col], bins=bins, include_lowest=True
                    ).astype(str)
                    # TabNet preprocess mặc định drop cột gốc nếu có binning
                    input_data.drop(columns=[col], inplace=True)
            
            # Encode categorical theo cat_maps
            for col in categorical_cols:
                if col in cat_maps:
                    mapping = cat_maps[col]
                    unk_idx = len(mapping)
                    input_val = input_data[col].values[0]
                    if input_val in mapping:
                        input_data.loc[0, col] = mapping[input_val]
                    else:
                        # Tắt cảnh báo trên UI, chỉ gán thành unknown index ngầm
                        input_data.loc[0, col] = unk_idx
                        
            # Sắp xếp đúng theo thứ tự X_train của tabnet
            X_tabnet = input_data[ordered_features].values.astype(np.float32)
            
            # 2. Load model & Predict
            clf = TabNetClassifier()
            clf.load_model(os.path.join(model_dir, 'best_tabnet_model.zip'))
            
            prediction = clf.predict(X_tabnet)[0]
            proba = clf.predict_proba(X_tabnet)[0][1]

        # In kết quả
        # st.write("---")
        # Định nghĩa hàm dialog (pop-up)
        @st.dialog("PREDICTION RESULT")
        def show_results(prediction, proba):
            st.markdown("<h1 style='text-align: center;'>Your Predicted Income Per Year Is</h1>", unsafe_allow_html=True)
            
            if prediction == 1:
                    st.success("#### 🌟 High Income Bracket (>= $50K)")
                    # st.metric(label="Likelihood of >= $50K", value=f"{proba:.2%}", delta="High Earning Potential")
                    # st.markdown("> *Based on the provided demographic and financial data, the model confidently places this profile in the higher compensation tier.*")
                    st.balloons()
            else:
                    st.warning("#### 📊 Standard Income Bracket (< $50K)")
                    # st.metric(label="Likelihood of >= $50K", value=f"{proba:.2%}", delta="Standard Earning Potential", delta_color="off")
                    # st.markdown("> *Based on the provided data, the model places this profile within the standard income group (< $50K/year).*")
                    st.snow()
                
        with st.spinner("Running inference..."):
            time.sleep(1)
        show_results(prediction, proba)
        
            
    except Exception as e:
        st.error(f"An error occurred during inference. Error: {e}")
