import os
import io
import datetime
import joblib
import pandas as pd
import numpy as np
import mysql.connector
import gradio as gr
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ---------------------------------------------------------------------------
# 1. DATABASE CONFIGURATION (FreeDatabase.com / MySQL)
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("MYSQL_HOST", "sql6.freedatabase.com")
DB_USER = os.getenv("MYSQL_USER", "your_username")
DB_PASS = os.getenv("MYSQL_PASSWORD", "your_password")
DB_NAME = os.getenv("MYSQL_DB", "your_dbname")
DB_PORT = int(os.getenv("MYSQL_PORT", 3306))

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            port=DB_PORT,
            connect_timeout=5
        )
        return conn
    except Exception as e:
        print(f"[DB Warning] Could not connect to MySQL database: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS energy_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                prediction_type VARCHAR(20),
                avg_usage_kwh FLOAT,
                total_co2_tonnes FLOAT,
                timestamp DATETIME
            )
            """)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[DB Init Error] {e}")

init_db()

# ---------------------------------------------------------------------------
# 2. MODEL & ARTIFACT LOADING
# ---------------------------------------------------------------------------
MODEL_PATH = 'steel_model.joblib'
SCALER_PATH = 'scaler.joblib'
COLUMNS_PATH = 'model_columns.joblib'

# Fallback in-memory models if files are absent
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
else:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    feature_columns = [
        'Lagging_Current_Reactive_Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
        'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM',
        'Hour', 'Month', 'Reactive_Power_Ratio', 'Power_Factor_Ratio',
        'WeekStatus_Weekend', 'Day_of_week_Monday', 'Day_of_week_Saturday',
        'Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday',
        'Day_of_week_Wednesday', 'Load_Type_Maximum_Load', 'Load_Type_Medium_Load'
    ]
    scaler = StandardScaler()
    scaler.fit(np.zeros((10, len(feature_columns))))
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(np.zeros((10, len(feature_columns))), np.zeros(10))

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING & PREDICTION PIPELINE
# ---------------------------------------------------------------------------
def preprocess_and_predict(df_raw):
    df = df_raw.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(' ', '_')
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('.', '_', regex=False)
    )
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')
        df['Hour'] = df['date'].dt.hour.fillna(12).astype(int)
        df['Month'] = df['date'].dt.month.fillna(1).astype(int)
    else:
        df['Hour'] = 12
        df['Month'] = 1

    eps = 1e-5
    df['Reactive_Power_Ratio'] = (
        df['Lagging_Current_Reactive_Power_kVarh'] / 
        (df['Leading_Current_Reactive_Power_kVarh'] + eps)
    )
    df['Power_Factor_Ratio'] = (
        df['Lagging_Current_Power_Factor'] / 
        (df['Leading_Current_Power_Factor'] + eps)
    )
    
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Reindex columns to match model training setup
    X_processed = pd.DataFrame(0, index=df.index, columns=feature_columns)
    for col in feature_columns:
        if col in df_encoded.columns:
            X_processed[col] = df_encoded[col]
            
    X_scaled = scaler.transform(X_processed)
    predictions = model.predict(X_scaled)
    
    df['Predicted_Usage_kWh'] = np.round(predictions, 4)
    df['Estimated_CO2_Tonnes'] = np.round(predictions * 0.00042, 6)
    
    return df

# ---------------------------------------------------------------------------
# 4. PDF GENERATION ENGINE
# ---------------------------------------------------------------------------
def generate_pdf_report(df_results, summary_stats):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=20
    )
    
    elements = []
    elements.append(Paragraph("Steel Plant Energy & CO2 Audit Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Hugging Face Spaces Engine", subtitle_style))
    
    # Summary Table
    summary_data = [
        ["Metric Description", "Value"],
        ["Total Processed Records", f"{summary_stats['total_records']}"],
        ["Mean Electricity Demand (kWh)", f"{summary_stats['mean_kwh']:.2f} kWh"],
        ["Max Electricity Demand (kWh)", f"{summary_stats['max_kwh']:.2f} kWh"],
        ["Total Estimated CO2 Footprint", f"{summary_stats['total_co2']:.4f} Metric Tonnes"],
        ["Mitigated Potential CO2 Saved (10% Peak Reduction)", f"{summary_stats['mitigated_co2']:.4f} Metric Tonnes"]
    ]
    
    st = Table(summary_data, colWidths=[300, 220])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    
    elements.append(st)
    elements.append(Spacer(1, 20))
    
    # Sample Output Table
    elements.append(Paragraph("Sample Predictions Preview (First 10 Rows)", styles['Heading2']))
    preview_df = df_results[['Predicted_Usage_kWh', 'Estimated_CO2_Tonnes']].head(10).reset_index()
    
    preview_data = [["Row ID", "Predicted Usage (kWh)", "Estimated CO2 (Tonnes)"]]
    for _, row in preview_df.iterrows():
        preview_data.append([str(int(row['index'] + 1)), f"{row['Predicted_Usage_kWh']:.2f}", f"{row['Estimated_CO2_Tonnes']:.6f}"])
        
    pt = Table(preview_data, colWidths=[100, 210, 210])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    elements.append(pt)
    doc.build(elements)
    
    pdf_buffer.seek(0)
    pdf_path = "/tmp/energy_prediction_report.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_buffer.read())
        
    return pdf_path

# ---------------------------------------------------------------------------
# 5. GRADIO HANDLERS
# ---------------------------------------------------------------------------
def handle_single_prediction(lag_kvarh, lead_kvarh, lag_pf, lead_pf, nsm, hour, month, day_of_week, load_type):
    data = {
        'Lagging_Current_Reactive_Power_kVarh': [lag_kvarh],
        'Leading_Current_Reactive_Power_kVarh': [lead_kvarh],
        'Lagging_Current_Power_Factor': [lag_pf],
        'Leading_Current_Power_Factor': [lead_pf],
        'NSM': [nsm],
        'Hour': [hour],
        'Month': [month],
        'Day_of_week': [day_of_week],
        'Load_Type': [load_type],
        'WeekStatus': ['Weekend' if day_of_week in ['Saturday', 'Sunday'] else 'Weekday']
    }
    df_single = pd.DataFrame(data)
    res = preprocess_and_predict(df_single)
    
    kwh = float(res['Predicted_Usage_kWh'].iloc[0])
    co2 = float(res['Estimated_CO2_Tonnes'].iloc[0])
    
    # Save to FreeDatabase MySQL async/safe
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO energy_predictions (prediction_type, avg_usage_kwh, total_co2_tonnes, timestamp) VALUES (%s, %s, %s, %s)",
                ('SINGLE', kwh, co2, datetime.datetime.now())
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[DB Write Error] {e}")

    return f"{kwh:.2f} kWh", f"{co2:.6f} Metric Tonnes"

def handle_csv_upload(file):
    if file is None:
        return None, "Please upload a valid CSV file.", None
    
    df_raw = pd.read_csv(file.name)
    df_results = preprocess_and_predict(df_raw)
    
    total_records = len(df_results)
    mean_kwh = df_results['Predicted_Usage_kWh'].mean()
    max_kwh = df_results['Predicted_Usage_kWh'].max()
    total_co2 = df_results['Estimated_CO2_Tonnes'].sum()
    
    # Simulate mitigation (10% drop on peak usage > mean)
    mitigated_kwh = df_results['Predicted_Usage_kWh'].copy()
    mitigated_kwh[mitigated_kwh > mean_kwh] *= 0.90
    mitigated_co2 = (df_results['Estimated_CO2_Tonnes'].sum() - (mitigated_kwh * 0.00042).sum())
    
    summary_stats = {
        'total_records': total_records,
        'mean_kwh': mean_kwh,
        'max_kwh': max_kwh,
        'total_co2': total_co2,
        'mitigated_co2': mitigated_co2
    }
    
    pdf_path = generate_pdf_report(df_results, summary_stats)
    
    summary_text = f"""
    ### 📊 Analysis Summary
    * **Total Processed Records:** {total_records}
    * **Average Predicted Demand:** {mean_kwh:.2f} kWh
    * **Peak Demand Recorded:** {max_kwh:.2f} kWh
    * **Total Estimated Carbon Footprint:** {total_co2:.4f} Metric Tonnes
    * **Potential CO2 Saved (via Peak Factor Mitigation):** {mitigated_co2:.4f} Metric Tonnes
    """
    
    # Save batch summary to MySQL
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO energy_predictions (prediction_type, avg_usage_kwh, total_co2_tonnes, timestamp) VALUES (%s, %s, %s, %s)",
                ('BATCH_CSV', mean_kwh, total_co2, datetime.datetime.now())
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[DB Write Error] {e}")

    return df_results, summary_text, pdf_path

# ---------------------------------------------------------------------------
# 6. SLEEK & GLOSSY GRADIO UI (WITH FUNCTIONAL SIDEBAR PANEL)
# ---------------------------------------------------------------------------
custom_css = """
/* Sleek Glossy Modern Interface Styling */
body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #f8fafc !important;
}

/* Glassmorphism Cards */
.glass-panel {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5) !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
}

/* Glossy Sidebar */
.sidebar-panel {
    background: rgba(15, 23, 42, 0.85) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 24px !important;
    border-radius: 16px !important;
}

/* Custom Buttons */
.btn-primary {
    background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.39) !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(2, 132, 199, 0.5) !important;
}
"""

with gr.Blocks(css=custom_css, title="Steel Industry Energy Predictor") as demo:
    gr.Markdown(
        """
        # ⚡ Steel Industry Power Consumption & Carbon Footprint Portal
        ### *Sleek AI-Powered Energy Demand & Environmental Mitigation Analytics*
        """
    )
    
    with gr.Row():
        # --- FUNCTIONAL SIDEBAR PANEL ---
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
            gr.Markdown("### ⚙️ Input Control Sidebar")
            gr.Markdown("Configure manual parameters or run batch processes directly.")
            
            with gr.Accordion("Single Inspection Parameters", open=True):
                lag_kvarh = gr.Number(label="Lagging Reactive Power (kVarh)", value=12.5)
                lead_kvarh = gr.Number(label="Leading Reactive Power (kVarh)", value=2.1)
                lag_pf = gr.Slider(minimum=0, maximum=100, label="Lagging Power Factor (%)", value=85.0)
                lead_pf = gr.Slider(minimum=0, maximum=100, label="Leading Power Factor (%)", value=95.0)
                nsm = gr.Number(label="Number of Seconds from Midnight (NSM)", value=36000)
                hour = gr.Slider(minimum=0, maximum=23, step=1, label="Hour of Day", value=10)
                month = gr.Slider(minimum=1, maximum=12, step=1, label="Month of Year", value=6)
                day_of_week = gr.Dropdown(
                    choices=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    value='Monday',
                    label="Day of Week"
                )
                load_type = gr.Dropdown(
                    choices=['Light_Load', 'Medium_Load', 'Maximum_Load'],
                    value='Medium_Load',
                    label="Load Type Category"
                )
            
            predict_btn = gr.Button("🚀 Predict Single Record", elem_classes=["btn-primary"])

        # --- MAIN WORKFLOW PANEL ---
        with gr.Column(scale=3):
            with gr.Tabs():
                # TAB 1: CSV UPLOAD & PDF DOWNLOAD WORKFLOW
                with gr.TabItem("📂 Batch CSV Upload & PDF Export"):
                    with gr.Group(elem_classes=["glass-panel"]):
                        gr.Markdown("### Upload Steel Industry Data CSV")
                        file_input = gr.File(label="Upload CSV File (`Steel_industry_data.csv`)", file_types=[".csv"])
                        process_csv_btn = gr.Button("⚡ Analyze CSV Data & Generate Report", elem_classes=["btn-primary"])
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            output_dataframe = gr.Dataframe(label="Predictions Table Preview", interactive=False)
                        with gr.Column(scale=1):
                            summary_output = gr.Markdown(value="*Upload a CSV to view analysis summary.*")
                            pdf_download = gr.File(label="📄 Download PDF Audit Report")

                # TAB 2: SINGLE PREDICTION RESULTS
                with gr.TabItem("🎯 Single Record Result"):
                    with gr.Group(elem_classes=["glass-panel"]):
                        gr.Markdown("### Single Inspection Prediction Result")
                        out_kwh = gr.Textbox(label="Predicted Energy Consumption (Usage_kWh)", interactive=False)
                        out_co2 = gr.Textbox(label="Estimated Carbon Emissions (Tonnes CO2)", interactive=False)

    # Event Bindings
    predict_btn.click(
        fn=handle_single_prediction,
        inputs=[lag_kvarh, lead_kvarh, lag_pf, lead_pf, nsm, hour, month, day_of_week, load_type],
        outputs=[out_kwh, out_co2]
    )
    
    process_csv_btn.click(
        fn=handle_csv_upload,
        inputs=[file_input],
        outputs=[output_dataframe, summary_output, pdf_download]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
