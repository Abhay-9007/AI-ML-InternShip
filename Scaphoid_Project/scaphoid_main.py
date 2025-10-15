
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, os

def generate_synthetic_data(n_samples=1500, seed=42):
    np.random.seed(seed)
    data = {
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'injury_mechanism': np.random.choice(['Fall', 'Sports', 'Accident', 'Other'], n_samples),
        'pain_location': np.random.choice(['Radial', 'Ulnar', 'Central', 'Diffuse'], n_samples),
        'tenderness': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'swelling': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'range_of_motion': np.random.normal(75, 20, n_samples),
        'initial_xray_finding': np.random.choice(['Negative', 'Equivocal', 'Positive'], n_samples),
        'mri_available': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'days_since_injury': np.random.exponential(7, n_samples).astype(int),
    }
    df = pd.DataFrame(data)
    fracture_prob = (
        0.1 +
        0.0005 * np.maximum(0, df['age'] - 50) +
        0.3 * df['tenderness'] +
        0.2 * df['swelling'] +
        0.4 * (df['initial_xray_finding'] == 'Equivocal').astype(int) +
        0.6 * (df['initial_xray_finding'] == 'Positive').astype(int) -
        0.001 * df['range_of_motion']
    )
    fracture_prob = fracture_prob.clip(0,0.9)
    df['fracture'] = np.random.binomial(1, fracture_prob)
    # introduce some NaNs
    for col in ['range_of_motion', 'days_since_injury']:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    return df

def preprocess_and_train(df):
    df_proc = df.copy()
    categorical_cols = ['gender','injury_mechanism','pain_location','initial_xray_finding']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        label_encoders[col] = le
    num_cols = ['age','range_of_motion','days_since_injury']
    imputer = SimpleImputer(strategy='median')
    df_proc[num_cols] = imputer.fit_transform(df_proc[num_cols])
    X = df_proc[[c for c in df_proc.columns if c!='fracture']]
    y = df_proc['fracture']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print("Training complete. Test AUC (approx):", "N/A - not calculated here")
    # Save model dict
    model_data = {
        'model': clf,
        'scaler': scaler,
        'label_encoders': label_encoders
    }
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_data, 'models/scaphoid_model.pkl')
    print("Saved model to models/scaphoid_model.pkl")

if __name__ == '__main__':
    df = generate_synthetic_data(1500)
    preprocess_and_train(df)
