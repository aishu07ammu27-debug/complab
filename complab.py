import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(
    page_title="AI Predictive Maintenance - Lab Risk Analytics",
    page_icon="🖥️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .status-high { color: #DC2626; font-weight: bold; }
    .status-med { color: #D97706; font-weight: bold; }
    .status-low { color: #059669; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA LOADING & CACHING PIPELINE
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data(file_source):
    # Reads either an uploaded file buffer or specified local path
    df = pd.read_excel(file_source)
    
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Map computer IDs into 4 distinct department lab units
    pc_nums = df['Computer_ID'].str.extract(r'(\d+)')[0].astype(int)
    df['Lab_Name'] = pc_nums.apply(lambda x: f"Lab {((x-1)//25)+1} (CS Dept)")
    
    # Standardize target binary outcome
    df['Target'] = df['Previous_Failure'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
    return df

@st.cache_resource
def train_models(df):
    feature_cols = [
        'Computer_Age', 'Daily_Usage_Hours', 'CPU_Temperature', 'RAM_Usage', 
        'Disk_Usage', 'Previous_Repair_Count', 'Days_Since_Maintenance', 
        'Shutdown_Count', 'Error_Count', 'Performance_Score'
    ]
    
    X = pd.get_dummies(df[feature_cols + ['Fan_Status']], columns=['Fan_Status'], drop_first=False)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    metrics = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        metrics[name] = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall': recall_score(y_test, preds, zero_division=0),
            'F1 Score': f1_score(y_test, preds, zero_division=0)
        }
        
    best_model = models['Random Forest']
    return best_model, models, metrics, X.columns.tolist(), scaler

def compute_dataset_risk(df, model, model_features):
    feature_cols = [
        'Computer_Age', 'Daily_Usage_Hours', 'CPU_Temperature', 'RAM_Usage', 
        'Disk_Usage', 'Previous_Repair_Count', 'Days_Since_Maintenance', 
        'Shutdown_Count', 'Error_Count', 'Performance_Score'
    ]
    X_full = pd.get_dummies(df[feature_cols + ['Fan_Status']], columns=['Fan_Status'], drop_first=False)
    X_full = X_full.reindex(columns=model_features, fill_value=0)
    
    probs = model.predict_proba(X_full)[:, 1]
    df['Failure_Probability'] = np.round(probs * 100, 1)
    
    def classify_risk(prob):
        if prob >= 70:
            return 'High Risk'
        elif prob >= 40:
            return 'Medium Risk'
        else:
            return 'Healthy'
            
    df['Risk_Level'] = df['Failure_Probability'].apply(classify_risk)
    return df

# ---------------------------------------------------------
# PATH CONFIGURATION & FILE HANDLING
# ---------------------------------------------------------
st.sidebar.title("🖥️ Navigation & Data Source")

# Primary OneDrive Document path formatted as a raw string to prevent unicode escape errors
PRIMARY_PATH = r"C:\Users\acer\Documents\collegelab.xlsx"
FALLBACK_PATH = "collegelab.xlsx"

uploaded_file = st.sidebar.file_uploader("Upload Dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    st.sidebar.success("Loaded from Uploaded File!")
elif os.path.exists(PRIMARY_PATH):
    df = load_and_clean_data(PRIMARY_PATH)
    st.sidebar.success("Loaded from OneDrive Documents")
elif os.path.exists(FALLBACK_PATH):
    df = load_and_clean_data(FALLBACK_PATH)
    st.sidebar.success("Loaded from Local Script Folder")
else:
    st.error(f"⚠️ Dataset `collegelab.xlsx` not found at:\n`{PRIMARY_PATH}`\n\nPlease upload the file in the sidebar.")
    st.stop()

# Initialize ML Pipeline
best_model, all_models, model_metrics, model_features, scaler = train_models(df)
df = compute_dataset_risk(df, best_model, model_features)

page = st.sidebar.radio(
    "Select Dashboard Module:",
    ["🏠 Dashboard", "📊 Analytics", "🤖 Prediction", "⚠️ Risk Analysis", "💡 Recommendations"]
)

# ---------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🖥️ College Computer Lab Predictive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time health monitoring and failure risk overview across computer labs.</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Computers", len(df))
    c2.metric("Healthy Systems", len(df[df['Risk_Level'] == 'Healthy']), delta="Normal Operation")
    c3.metric("Medium Risk", len(df[df['Risk_Level'] == 'Medium Risk']), delta="Attention Needed", delta_color="normal")
    c4.metric("High Risk", len(df[df['Risk_Level'] == 'High Risk']), delta="Critical Action Required", delta_color="inverse")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("System Risk Level Distribution")
        fig_pie = px.pie(
            df, names='Risk_Level', color='Risk_Level',
            color_discrete_map={'Healthy': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.subheader("Lab-wise Risk Level Breakdown")
        lab_risk = df.groupby(['Lab_Name', 'Risk_Level']).size().reset_index(name='Count')
        fig_bar = px.bar(
            lab_risk, x='Lab_Name', y='Count', color='Risk_Level',
            color_discrete_map={'Healthy': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'},
            barmode='stack'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Computer Age vs Failure Risk")
        fig_scatter1 = px.scatter(
            df, x='Computer_Age', y='Failure_Probability', color='Risk_Level',
            size='Daily_Usage_Hours', hover_data=['Computer_ID'],
            color_discrete_map={'Healthy': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'}
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
        
    with c6:
        st.subheader("Daily Usage Hours vs Performance Score")
        fig_scatter2 = px.scatter(
            df, x='Daily_Usage_Hours', y='Performance_Score', color='Risk_Level',
            hover_data=['Computer_ID'],
            color_discrete_map={'Healthy': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'}
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: ANALYTICS
# ---------------------------------------------------------
elif page == "📊 Analytics":
    st.markdown('<div class="main-header">📊 Exploratory Data & Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-depth statistical insights into system workload, thermal stress, and ML performance.</div>', unsafe_allow_html=True)
    
    st.subheader("🤖 Machine Learning Model Evaluation")
    metrics_df = pd.DataFrame(model_metrics).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)
    
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🔥 Thermal & Workload Stress", "🛠️ Repair & Maintenance History", "📉 Metric Correlation Matrix"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("CPU Temperature vs Historical Failure")
            fig = px.box(df, x='Previous_Failure', y='CPU_Temperature', color='Previous_Failure')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("RAM vs Disk Usage Distribution")
            fig = px.scatter(df, x='RAM_Usage', y='Disk_Usage', color='Risk_Level', size='Error_Count',
                             color_discrete_map={'Healthy': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'})
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Days Since Maintenance vs Error Count")
            fig = px.scatter(df, x='Days_Since_Maintenance', y='Error_Count', color='Risk_Level',
                             color_discrete_map={'Healthy': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Repair Count Distribution")
            fig = px.histogram(df, x='Previous_Repair_Count', color='Previous_Failure', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        st.subheader("Numerical Metric Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: PREDICTION
# ---------------------------------------------------------
elif page == "🤖 Prediction":
    st.markdown('<div class="main-header">🤖 Single System Failure Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input operational parameters for a computer unit to compute real-time risk.</div>', unsafe_allow_html=True)
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        comp_age = st.number_input("Computer Age (Years)", min_value=1, max_value=10, value=4)
        daily_hours = st.slider("Daily Usage Hours", 1.0, 16.0, 10.0, step=0.5)
        cpu_temp = st.slider("CPU Temperature (°C)", 40.0, 95.0, 78.0, step=0.5)
        ram_usage = st.slider("RAM Usage (%)", 20.0, 99.0, 86.0, step=1.0)
    with col_in2:
        disk_usage = st.slider("Disk Usage (%)", 20.0, 99.0, 75.0, step=1.0)
        prev_repairs = st.number_input("Previous Repairs Count", min_value=0, max_value=10, value=3)
        days_maint = st.number_input("Days Since Last Maintenance", min_value=1, max_value=365, value=120)
        shutdown_count = st.number_input("Unexpected Shutdowns", min_value=0, max_value=30, value=8)
    with col_in3:
        error_count = st.number_input("System Error Logs Count", min_value=0, max_value=50, value=15)
        fan_status = st.selectbox("Cooling Fan Status", ["Normal", "Fair", "Needs Service"])
        perf_score = st.slider("Performance Score", 10.0, 100.0, 42.0, step=1.0)

    if st.button("🚀 Calculate Risk & Predict Maintenance Status", use_container_width=True):
        input_data = pd.DataFrame([{
            'Computer_Age': comp_age,
            'Daily_Usage_Hours': daily_hours,
            'CPU_Temperature': cpu_temp,
            'RAM_Usage': ram_usage,
            'Disk_Usage': disk_usage,
            'Previous_Repair_Count': prev_repairs,
            'Days_Since_Maintenance': days_maint,
            'Shutdown_Count': shutdown_count,
            'Error_Count': error_count,
            'Performance_Score': perf_score,
            'Fan_Status_Fair': 1 if fan_status == 'Fair' else 0,
            'Fan_Status_Needs Service': 1 if fan_status == 'Needs Service' else 0,
            'Fan_Status_Normal': 1 if fan_status == 'Normal' else 0,
        }])
        
        input_data = input_data.reindex(columns=model_features, fill_value=0)
        prob = best_model.predict_proba(input_data)[0][1] * 100
        
        st.markdown("---")
        st.subheader("━━━━━━━━━━━━━━━━━━━━━━ FAILURE PREDICTION RESULT ━━━━━━━━━━━━━━━━━━━━━━")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Failure Probability", f"{prob:.1f}%")
        with res_col2:
            if prob >= 70:
                st.markdown("Risk Level: <span class='status-high'>HIGH</span>", unsafe_allow_html=True)
            elif prob >= 40:
                st.markdown("Risk Level: <span class='status-med'>MEDIUM</span>", unsafe_allow_html=True)
            else:
                st.markdown("Risk Level: <span class='status-low'>LOW / HEALTHY</span>", unsafe_allow_html=True)
        with res_col3:
            if prob >= 70:
                st.markdown("Maintenance Status: <span class='status-high'>REQUIRED IMMEDIATELY</span>", unsafe_allow_html=True)
            elif prob >= 40:
                st.markdown("Maintenance Status: <span class='status-med'>SCHEDULE WITHIN 7 DAYS</span>", unsafe_allow_html=True)
            else:
                st.markdown("Maintenance Status: <span class='status-low'>ROUTINE MONITORING</span>", unsafe_allow_html=True)
                
        st.markdown("---")
        st.subheader("🔍 Individual Feature Impact Breakdown")
        
        factors = {
            'CPU Temperature': min(cpu_temp / 90.0, 1.0) * 100,
            'RAM Usage': min(ram_usage / 100.0, 1.0) * 100,
            'Previous Repairs': min(prev_repairs / 5.0, 1.0) * 100,
            'Computer Age': min(comp_age / 6.0, 1.0) * 100,
            'System Error Count': min(error_count / 25.0, 1.0) * 100
        }
        
        factor_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Impact_Score']).sort_values(by='Impact_Score', ascending=True)
        fig_bar_factors = px.bar(factor_df, x='Impact_Score', y='Factor', orientation='h', color='Impact_Score', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar_factors, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: RISK ANALYSIS
# ---------------------------------------------------------
elif page == "⚠️ Risk Analysis":
    st.markdown('<div class="main-header">⚠️ College Risk Analysis & Lab Rankings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Systematic audit of high-risk units across department laboratories.</div>', unsafe_allow_html=True)
    
    st.subheader("High-Risk Computer Units (Action Priority 1)")
    high_risk_df = df[df['Risk_Level'] == 'High Risk'][
        ['Computer_ID', 'Lab_Name', 'Computer_Age', 'CPU_Temperature', 'RAM_Usage', 'Error_Count', 'Failure_Probability']
    ].sort_values(by='Failure_Probability', ascending=False)
    
    st.dataframe(high_risk_df.style.background_gradient(cmap='Reds', subset=['Failure_Probability']), use_container_width=True)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lab Unit Aggregate Risk Index")
        lab_summary = df.groupby('Lab_Name').agg(
            Total_Systems=('Computer_ID', 'count'),
            High_Risk_Count=('Risk_Level', lambda x: (x == 'High Risk').sum()),
            Avg_Failure_Prob=('Failure_Probability', 'mean')
        ).reset_index()
        st.dataframe(lab_summary.style.highlight_max(subset=['High_Risk_Count'], color='#FEE2E2'), use_container_width=True)
        
    with c2:
        st.subheader("Maintenance Lapse vs Probability")
        fig = px.scatter(
            df, x='Days_Since_Maintenance', y='Failure_Probability',
            color='Risk_Level', size='Error_Count', hover_data=['Computer_ID'],
            color_discrete_map={'Healthy': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: RECOMMENDATIONS
# ---------------------------------------------------------
elif page == "💡 Recommendations":
    st.markdown('<div class="main-header">💡 Automated Maintenance Action Plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Actionable scheduling matrix generated for college lab administrators.</div>', unsafe_allow_html=True)
    
    high_risk_count = len(df[df['Risk_Level'] == 'High Risk'])
    med_risk_count = len(df[df['Risk_Level'] == 'Medium Risk'])
    
    st.success(f"✅ **Audit Summary**: Analyzed **{len(df)}** Lab PCs. **{high_risk_count} High Risk Cases** requiring prompt technician response.")
    
    rec_data = [
        {"Priority": "🔴 Priority 1 (Immediate)", "Target Condition": f"High Risk Systems ({high_risk_count} PCs)", "Timeframe": "Within 48 Hours", "Action Plan": "Dispatch technician for thermal paste re-application, fan replacement, and error log audit."},
        {"Priority": "🟠 Priority 2 (Medium)", "Target Condition": f"Medium Risk Systems ({med_risk_count} PCs)", "Timeframe": "Within 7 Days", "Action Plan": "Perform disk defragmentation/cleanup, update operating system patches, and review RAM allocation."},
        {"Priority": "🟢 Priority 3 (Routine)", "Target Condition": "Healthy Systems", "Timeframe": "Next Scheduled Cycle", "Action Plan": "Maintain bi-monthly dusting schedule and standard performance metric logging."}
    ]
    st.table(pd.DataFrame(rec_data))
