from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
MODEL_PATH = "models/scaphoid_model.pkl"

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data.get('label_encoders', {})

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_input(input_df):
    df = input_df.copy()
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    X = df[[c for c in df.columns if c != 'fracture']]
    X_scaled = scaler.transform(X)
    return X_scaled

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df['age'] = df['age'].astype(float)
        df['range_of_motion'] = df['range_of_motion'].astype(float)
        df['days_since_injury'] = df['days_since_injury'].astype(float)
        X_scaled = preprocess_input(df)
        prob = model.predict_proba(X_scaled)[:,1][0]
        if prob < 0.3:
            risk = "Low"
            rec = "Routine follow-up recommended"
        elif prob < 0.7:
            risk = "Moderate"
            rec = "Further imaging (MRI) recommended"
        else:
            risk = "High"
            rec = "Immediate orthopedic consultation recommended"
        return jsonify({
            "fracture_probability": round(float(prob*100),2),
            "risk_level": risk,
            "recommendation": rec,
            "success": True
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
