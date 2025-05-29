import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import io # Added for StringIO

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Ensure directories exist
DATA_DIR = 'urban_resource_data'
MODELS_DIR = 'models'
STATIC_IMG_DIR = os.path.join('static', 'images')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'urban_resource_dataset.csv')
ELECTRICITY_MODEL_PATH = os.path.join(MODELS_DIR, 'electricity_model.pkl')
EVENT_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'event_classifier.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

# Global variables to store df and models
df = pd.DataFrame()
electricity_model = None
event_classifier_model = None
event_label_encoder = None

def generate_synthetic_data():
    """Generates synthetic urban resource usage data and saves it to CSV."""
    global df
    print("Generating synthetic dataset...")
    random.seed(42)
    np.random.seed(42)

    base_time = datetime.now()
    timestamps = [base_time - timedelta(hours=i) for i in range(999, -1, -1)]

    # Base generation for waste levels
    waste_level_base = np.random.uniform(20, 100, size=1000).round(2)

    # Introduce some clear anomalies (e.g., 5 very low, 5 very high)
    num_anomalies = 10 # Number of anomalies you want to inject
    anomaly_indices = np.random.choice(1000, num_anomalies, replace=False) # Get random indices

    for i, idx in enumerate(anomaly_indices):
        if i < num_anomalies // 2: # Half of the anomalies will be very low
            waste_level_base[idx] = np.random.uniform(0, 10).round(2) # Waste near 0-10%
        else: # Other half will be very high
            waste_level_base[idx] = np.random.uniform(105, 120).round(2) # Waste above 100% (anomalous)


    data = {
        'Timestamp': timestamps,
        'Electricity Usage (kW)': np.random.normal(loc=300, scale=50, size=1000).round(2),
        'Water Consumption (Liters)': np.random.normal(loc=10000, scale=2000, size=1000).round(2),
        'Traffic Density (Vehicles/Min)': np.random.randint(10, 100, size=1000),
        'Waste Level (%)': waste_level_base,
        'Zone': [random.choice(['Zone A', 'Zone B', 'Zone C', 'Zone D']) for _ in range(1000)],
        'Event/Anomaly': [random.choice(['None', 'Festival', 'Power Outage', 'Heavy Rain', 'Maintenance', None]) for _ in range(1000)]
    }

    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Synthetic dataset generated and saved as '{DATASET_PATH}'.")

def load_and_preprocess_data():
    """Loads the dataset and performs preprocessing."""
    global df, event_label_encoder
    print("Loading and preprocessing data...")
    if not os.path.exists(DATASET_PATH):
        generate_synthetic_data()

    df = pd.read_csv(DATASET_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.drop_duplicates()
    print("Data loaded and preprocessed.")

    event_label_encoder = LabelEncoder()
    all_event_labels = df['Event/Anomaly'].astype(str).unique()
    event_label_encoder.fit(all_event_labels)
    df['Event_Label'] = event_label_encoder.transform(df['Event/Anomaly'].astype(str))
    joblib.dump(event_label_encoder, LABEL_ENCODER_PATH)


def perform_eda():
    """Performs Exploratory Data Analysis and generates visualizations."""
    if df.empty:
        load_and_preprocess_data()

    eda_results = {}

    # Basic Info
    buffer = io.StringIO()
    df.info(buf=buffer)
    eda_results['data_info'] = buffer.getvalue()
    buffer.close()

    # Descriptive Statistics
    eda_results['descriptive_stats'] = df.describe().to_html()

    # Missing Values
    eda_results['missing_values'] = df.isnull().sum().to_frame(name='Missing Count').to_html()

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['Electricity Usage (kW)'], label='Electricity Usage (kW)')
    plt.plot(df['Timestamp'], df['Water Consumption (Liters)'], label='Water Consumption (Liters)')
    plt.xlabel("Timestamp")
    plt.ylabel("Usage")
    plt.title("Electricity and Water Consumption Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'time_series_usage.png'))
    plt.close()
    eda_results['time_series_plot'] = '/static/images/time_series_usage.png'

    plt.figure(figsize=(8, 5))
    sns.histplot(df['Traffic Density (Vehicles/Min)'], bins=30, kde=True, color='skyblue')
    plt.title("Distribution of Traffic Density")
    plt.xlabel("Vehicles per Minute")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'traffic_density_distribution.png'))
    plt.close()
    eda_results['traffic_density_plot'] = '/static/images/traffic_density_distribution.png'

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['Waste Level (%)'], color='lightgreen')
    plt.title("Boxplot Waste Level (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'waste_level_boxplot.png'))
    plt.close()
    eda_results['waste_level_boxplot'] = '/static/images/waste_level_boxplot.png'

    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'correlation_heatmap.png'))
    plt.close()
    eda_results['correlation_heatmap'] = '/static/images/correlation_heatmap.png'

    print("EDA complete.")
    return eda_results

def train_electricity_prediction_model():
    """Trains and saves the Electricity Usage Prediction model."""
    global electricity_model
    if df.empty:
        load_and_preprocess_data()

    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month

    X = df[['Hour', 'DayOfWeek', 'Month', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)']]
    y = df['Electricity Usage (kW)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    electricity_model = LinearRegression()
    electricity_model.fit(X_train, y_train)

    y_pred = electricity_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    joblib.dump(electricity_model, ELECTRICITY_MODEL_PATH)
    print("Electricity Usage Prediction model trained and saved.")
    return {"r2_score": f"{r2:.2f}", "mse": f"{mse:.2f}"}

def train_event_classification_model():
    """Trains and saves the Event/Anomaly Classification model."""
    global event_classifier_model, event_label_encoder
    if df.empty:
        load_and_preprocess_data()

    if 'Event_Label' not in df.columns or event_label_encoder is None:
        load_and_preprocess_data()

    X = df[['Electricity Usage (kW)', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)']]
    y = df['Event_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    event_classifier_model = RandomForestClassifier(n_estimators=100, random_state=42)
    event_classifier_model.fit(X_train, y_train)

    y_pred = event_classifier_model.predict(X_test)

    unique_labels_in_y_test = np.unique(y_test)
    if event_label_encoder is not None:
        actual_target_names = [str(event_label_encoder.inverse_transform([label])[0]) for label in sorted(unique_labels_in_y_test)]
    else:
        actual_target_names = [str(label) for label in sorted(unique_labels_in_y_test)]


    report = classification_report(y_test, y_pred, target_names=actual_target_names, output_dict=True, zero_division=0)

    joblib.dump(event_classifier_model, EVENT_CLASSIFIER_PATH)
    print("Event Classification model trained and saved.")
    return report

def detect_waste_anomalies():
    """Detects anomalies in Waste Level (%) using the IQR method."""
    if df.empty:
        load_and_preprocess_data()

    Q1 = df['Waste Level (%)'].quantile(0.25)
    Q3 = df['Waste Level (%)'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    anomalies = df[(df['Waste Level (%)'] < lower_bound) | (df['Waste Level (%)'] > upper_bound)]

    print(f"Number of Waste Anomalies Detected: {len(anomalies)}")
    return anomalies[['Timestamp', 'Waste Level (%)', 'Zone']].to_html(index=False)

# Load data and try loading models on startup
with app.app_context():
    load_and_preprocess_data()

    try:
        if os.path.exists(ELECTRICITY_MODEL_PATH):
            electricity_model = joblib.load(ELECTRICITY_MODEL_PATH)
            print("Electricity model loaded.")
        else:
            print("Electricity model not found, training new model...")
            train_electricity_prediction_model()
    except EOFError:
        print(f"Corrupted or empty file at {ELECTRICITY_MODEL_PATH}. Retraining model...")
        train_electricity_prediction_model()
    except Exception as e:
        print(f"Error loading electricity model: {e}. Retraining model...")
        train_electricity_prediction_model()


    try:
        if os.path.exists(EVENT_CLASSIFIER_PATH):
            event_classifier_model = joblib.load(EVENT_CLASSIFIER_PATH)
            print("Event classifier model loaded.")
        else:
            print("Event classifier model not found, training new model...")
            train_event_classification_model()
    except EOFError:
        print(f"Corrupted or empty file at {EVENT_CLASSIFIER_PATH}. Retraining model...")
        train_event_classification_model()
    except Exception as e:
        print(f"Error loading event classifier model: {e}. Retraining model...")
        train_event_classification_model()

    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            event_label_encoder = joblib.load(LABEL_ENCODER_PATH)
            print("Label encoder loaded.")
        except EOFError:
            print(f"Corrupted or empty file at {LABEL_ENCODER_PATH}. Re-initializing and fitting encoder...")
            load_and_preprocess_data()
        except Exception as e:
            print(f"Error loading label encoder: {e}. Re-initializing and fitting encoder...")
            load_and_preprocess_data()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    generate_synthetic_data()
    load_and_preprocess_data()
    train_electricity_prediction_model()
    train_event_classification_model()
    return jsonify({"message": "Synthetic dataset generated successfully and models re-trained!"})

@app.route('/eda')
def eda():
    eda_results = perform_eda()
    return render_template('eda.html', eda=eda_results)

@app.route('/predictions')
def predictions_dashboard():
    electricity_results = train_electricity_prediction_model()
    event_classification_results = train_event_classification_model()

    return render_template('predictions.html',
                           electricity_results=electricity_results,
                           event_classification_results=event_classification_results)

@app.route('/predict_electricity_ajax', methods=['POST'])
def predict_electricity_ajax():
    if electricity_model is None:
        print("Electricity model not available for prediction, re-training...")
        train_electricity_prediction_model()

    try:
        data = request.json
        print(f"Received data for electricity prediction: {data}")
        hour = int(data['hour'])
        dayofweek = int(data['dayofweek'])
        month = int(data['month'])
        water_consumption = float(data['water_consumption'])
        traffic_density = int(data['traffic_density'])
        waste_level = float(data['waste_level'])

        input_data = pd.DataFrame([[hour, dayofweek, month, water_consumption, traffic_density, waste_level]],
                                  columns=['Hour', 'DayOfWeek', 'Month', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)'])
        print(f"Input DataFrame for prediction: \n{input_data}")

        prediction = electricity_model.predict(input_data)[0]
        return jsonify({"prediction": f"{prediction:.2f} kW"})
    except Exception as e:
        print(f"Error in predict_electricity_ajax: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/classify_event_ajax', methods=['POST'])
def classify_event_ajax():
    global event_label_encoder
    if event_classifier_model is None or event_label_encoder is None:
        print("Event classification model or encoder not available, re-training...")
        train_event_classification_model()
        if os.path.exists(LABEL_ENCODER_PATH):
            try:
                event_label_encoder = joblib.load(LABEL_ENCODER_PATH)
            except Exception as e:
                print(f"Error loading event_label_encoder after re-training: {e}")
                load_and_preprocess_data()

    try:
        data = request.json
        print(f"Received data for event classification: {data}")
        electricity = float(data['electricity_usage'])
        water = float(data['water_consumption'])
        traffic = int(data['traffic_density'])
        waste = float(data['waste_level'])

        input_data = pd.DataFrame([[electricity, water, traffic, waste]],
                                  columns=['Electricity Usage (kW)', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)'])
        print(f"Input DataFrame for classification: \n{input_data}")

        predicted_label_index = event_classifier_model.predict(input_data)[0]
        print(f"Predicted label index: {predicted_label_index}")

        try:
            if event_label_encoder is not None:
                predicted_event = event_label_encoder.inverse_transform([predicted_label_index])[0]
            else:
                predicted_event = "Error: Label encoder not initialized."
                print("Error: event_label_encoder is None in classify_event_ajax")
        except ValueError as ve:
            predicted_event = f"Unknown Event (label {predicted_label_index}): {ve}"
            print(f"Warning: {predicted_event}")

        return jsonify({"prediction": predicted_event})
    except Exception as e:
        print(f"Error in classify_event_ajax: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/detect_anomalies')
def detect_anomalies():
    anomalies_html = detect_waste_anomalies()
    return render_template('anomaly.html', anomalies=anomalies_html)

# NEW ROUTE FOR INFOGRAPHIC
@app.route('/infographic')
def infographic():
    return render_template('infographic.html')

# if __name__ == '__main__':
#     app.run(debug=True)




# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import joblib
# import io # Added for StringIO

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report

# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# # Ensure directories exist
# DATA_DIR = 'urban_resource_data'
# MODELS_DIR = 'models'
# STATIC_IMG_DIR = os.path.join('static', 'images')

# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(STATIC_IMG_DIR, exist_ok=True)

# DATASET_PATH = os.path.join(DATA_DIR, 'urban_resource_dataset.csv')
# ELECTRICITY_MODEL_PATH = os.path.join(MODELS_DIR, 'electricity_model.pkl')
# EVENT_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'event_classifier.pkl')
# LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

# # Global variables to store df and models
# df = pd.DataFrame()
# electricity_model = None
# event_classifier_model = None
# event_label_encoder = None

# def generate_synthetic_data():
#     """Generates synthetic urban resource usage data and saves it to CSV."""
#     global df
#     print("Generating synthetic dataset...")
#     random.seed(42)
#     np.random.seed(42)

#     base_time = datetime.now()
#     timestamps = [base_time - timedelta(hours=i) for i in range(999, -1, -1)]

#     # Base generation
#     waste_level_base = np.random.uniform(20, 100, size=1000).round(2)

#     # Introduce some clear anomalies (e.g., 5 very low, 5 very high)
#     num_anomalies = 10
#     anomaly_indices = np.random.choice(1000, num_anomalies, replace=False)

#     for i, idx in enumerate(anomaly_indices):
#         if i < num_anomalies // 2: # Half very low
#             waste_level_base[idx] = np.random.uniform(0, 10) # Waste near 0-10%
#         else: # Half very high
#             waste_level_base[idx] = np.random.uniform(105, 120) # Waste above 100% (anomalous)

#     data = {
#         'Timestamp': timestamps,
#         'Electricity Usage (kW)': np.random.normal(loc=300, scale=50, size=1000).round(2),
#         'Water Consumption (Liters)': np.random.normal(loc=10000, scale=2000, size=1000).round(2),
#         'Traffic Density (Vehicles/Min)': np.random.randint(10, 100, size=1000),
#         'Waste Level (%)': waste_level_base, # Use the modified waste_level_base
#         'Zone': [random.choice(['Zone A', 'Zone B', 'Zone C', 'Zone D']) for _ in range(1000)],
#         'Event/Anomaly': [random.choice(['None', 'Festival', 'Power Outage', 'Heavy Rain', 'Maintenance', None]) for _ in range(1000)]
#     }

#     df = pd.DataFrame(data)
#     df.to_csv(DATASET_PATH, index=False)
#     print(f"Synthetic dataset generated and saved as '{DATASET_PATH}'.")


# # def generate_synthetic_data():
# #     """Generates synthetic urban resource usage data and saves it to CSV."""
# #     global df
# #     print("Generating synthetic dataset...")
# #     random.seed(42)
# #     np.random.seed(42)

# #     base_time = datetime.now()
# #     timestamps = [base_time - timedelta(hours=i) for i in range(999, -1, -1)]

# #     data = {
# #         'Timestamp': timestamps,
# #         'Electricity Usage (kW)': np.random.normal(loc=300, scale=50, size=1000).round(2),
# #         'Water Consumption (Liters)': np.random.normal(loc=10000, scale=2000, size=1000).round(2),
# #         'Traffic Density (Vehicles/Min)': np.random.randint(10, 100, size=1000),
# #         'Waste Level (%)': np.random.uniform(20, 100, size=1000).round(2),
# #         'Zone': [random.choice(['Zone A', 'Zone B', 'Zone C', 'Zone D']) for _ in range(1000)],
# #         'Event/Anomaly': [random.choice(['None', 'Festival', 'Power Outage', 'Heavy Rain', 'Maintenance', None]) for _ in range(1000)]
# #     }

# #     df = pd.DataFrame(data)
# #     df.to_csv(DATASET_PATH, index=False)
# #     print(f"Synthetic dataset generated and saved as '{DATASET_PATH}'.")

# def load_and_preprocess_data():
#     """Loads the dataset and performs preprocessing."""
#     global df, event_label_encoder
#     print("Loading and preprocessing data...")
#     if not os.path.exists(DATASET_PATH):
#         generate_synthetic_data()

#     df = pd.read_csv(DATASET_PATH)
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     df = df.drop_duplicates()
#     print("Data loaded and preprocessed.")

#     event_label_encoder = LabelEncoder()
#     all_event_labels = df['Event/Anomaly'].astype(str).unique()
#     event_label_encoder.fit(all_event_labels)
#     df['Event_Label'] = event_label_encoder.transform(df['Event/Anomaly'].astype(str))
#     joblib.dump(event_label_encoder, LABEL_ENCODER_PATH)


# def perform_eda():
#     """Performs Exploratory Data Analysis and generates visualizations."""
#     if df.empty:
#         load_and_preprocess_data()

#     eda_results = {}

#     # Basic Info
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     eda_results['data_info'] = buffer.getvalue()
#     buffer.close()

#     # Descriptive Statistics
#     eda_results['descriptive_stats'] = df.describe().to_html()

#     # Missing Values
#     eda_results['missing_values'] = df.isnull().sum().to_frame(name='Missing Count').to_html()

#     # Visualizations
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['Timestamp'], df['Electricity Usage (kW)'], label='Electricity Usage (kW)')
#     plt.plot(df['Timestamp'], df['Water Consumption (Liters)'], label='Water Consumption (Liters)')
#     plt.xlabel("Timestamp")
#     plt.ylabel("Usage")
#     plt.title("Electricity and Water Consumption Over Time")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(STATIC_IMG_DIR, 'time_series_usage.png'))
#     plt.close()
#     eda_results['time_series_plot'] = '/static/images/time_series_usage.png'

#     plt.figure(figsize=(8, 5))
#     sns.histplot(df['Traffic Density (Vehicles/Min)'], bins=30, kde=True, color='skyblue')
#     plt.title("Distribution of Traffic Density")
#     plt.xlabel("Vehicles per Minute")
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(STATIC_IMG_DIR, 'traffic_density_distribution.png'))
#     plt.close()
#     eda_results['traffic_density_plot'] = '/static/images/traffic_density_distribution.png'

#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x=df['Waste Level (%)'], color='lightgreen')
#     plt.title("Boxplot Waste Level (%)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(STATIC_IMG_DIR, 'waste_level_boxplot.png'))
#     plt.close()
#     eda_results['waste_level_boxplot'] = '/static/images/waste_level_boxplot.png'

#     plt.figure(figsize=(8, 6))
#     corr = df.select_dtypes(include=[np.number]).corr()
#     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title("Correlation Between Numerical Features")
#     plt.tight_layout()
#     plt.savefig(os.path.join(STATIC_IMG_DIR, 'correlation_heatmap.png'))
#     plt.close()
#     eda_results['correlation_heatmap'] = '/static/images/correlation_heatmap.png'

#     print("EDA complete.")
#     return eda_results

# def train_electricity_prediction_model():
#     """Trains and saves the Electricity Usage Prediction model."""
#     global electricity_model
#     if df.empty:
#         load_and_preprocess_data()

#     df['Hour'] = df['Timestamp'].dt.hour
#     df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
#     df['Month'] = df['Timestamp'].dt.month

#     X = df[['Hour', 'DayOfWeek', 'Month', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)']]
#     y = df['Electricity Usage (kW)']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     electricity_model = LinearRegression()
#     electricity_model.fit(X_train, y_train)

#     y_pred = electricity_model.predict(X_test)

#     r2 = r2_score(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)

#     joblib.dump(electricity_model, ELECTRICITY_MODEL_PATH)
#     print("Electricity Usage Prediction model trained and saved.")
#     return {"r2_score": f"{r2:.2f}", "mse": f"{mse:.2f}"}

# def train_event_classification_model():
#     """Trains and saves the Event/Anomaly Classification model."""
#     global event_classifier_model, event_label_encoder
#     if df.empty:
#         load_and_preprocess_data()

#     if 'Event_Label' not in df.columns or event_label_encoder is None:
#         load_and_preprocess_data()

#     X = df[['Electricity Usage (kW)', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)']]
#     y = df['Event_Label']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     event_classifier_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     event_classifier_model.fit(X_train, y_train)

#     y_pred = event_classifier_model.predict(X_test)

#     unique_labels_in_y_test = np.unique(y_test)
#     # Ensure event_label_encoder is not None before inverse_transform
#     if event_label_encoder is not None:
#         actual_target_names = [str(event_label_encoder.inverse_transform([label])[0]) for label in sorted(unique_labels_in_y_test)]
#     else:
#         # Fallback if encoder is somehow missing during training (shouldn't happen with load_and_preprocess_data)
#         actual_target_names = [str(label) for label in sorted(unique_labels_in_y_test)]


#     report = classification_report(y_test, y_pred, target_names=actual_target_names, output_dict=True, zero_division=0)

#     joblib.dump(event_classifier_model, EVENT_CLASSIFIER_PATH)
#     print("Event Classification model trained and saved.")
#     return report

# def detect_waste_anomalies():
#     """Detects anomalies in Waste Level (%) using the IQR method."""
#     if df.empty:
#         load_and_preprocess_data()

#     Q1 = df['Waste Level (%)'].quantile(0.25)
#     Q3 = df['Waste Level (%)'].quantile(0.75)
#     IQR = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     anomalies = df[(df['Waste Level (%)'] < lower_bound) | (df['Waste Level (%)'] > upper_bound)]

#     print(f"Number of Waste Anomalies Detected: {len(anomalies)}")
#     return anomalies[['Timestamp', 'Waste Level (%)', 'Zone']].to_html(index=False)

# # Load data and try loading models on startup
# with app.app_context():
#     load_and_preprocess_data()

#     try:
#         if os.path.exists(ELECTRICITY_MODEL_PATH):
#             electricity_model = joblib.load(ELECTRICITY_MODEL_PATH)
#             print("Electricity model loaded.")
#         else:
#             print("Electricity model not found, training new model...")
#             train_electricity_prediction_model()
#     except EOFError:
#         print(f"Corrupted or empty file at {ELECTRICITY_MODEL_PATH}. Retraining model...")
#         train_electricity_prediction_model()
#     except Exception as e:
#         print(f"Error loading electricity model: {e}. Retraining model...")
#         train_electricity_prediction_model()


#     try:
#         if os.path.exists(EVENT_CLASSIFIER_PATH):
#             event_classifier_model = joblib.load(EVENT_CLASSIFIER_PATH)
#             print("Event classifier model loaded.")
#         else:
#             print("Event classifier model not found, training new model...")
#             train_event_classification_model()
#     except EOFError:
#         print(f"Corrupted or empty file at {EVENT_CLASSIFIER_PATH}. Retraining model...")
#         train_event_classification_model()
#     except Exception as e:
#         print(f"Error loading event classifier model: {e}. Retraining model...")
#         train_event_classification_model()

#     if os.path.exists(LABEL_ENCODER_PATH):
#         try:
#             event_label_encoder = joblib.load(LABEL_ENCODER_PATH)
#             print("Label encoder loaded.")
#         except EOFError:
#             print(f"Corrupted or empty file at {LABEL_ENCODER_PATH}. Re-initializing and fitting encoder...")
#             load_and_preprocess_data()
#         except Exception as e:
#             print(f"Error loading label encoder: {e}. Re-initializing and fitting encoder...")
#             load_and_preprocess_data()


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/generate_data', methods=['POST'])
# def generate_data():
#     generate_synthetic_data()
#     load_and_preprocess_data()
#     train_electricity_prediction_model()
#     train_event_classification_model()
#     return jsonify({"message": "Synthetic dataset generated successfully and models re-trained!"})

# @app.route('/eda')
# def eda():
#     eda_results = perform_eda()
#     return render_template('eda.html', eda=eda_results)

# # Combined route for displaying prediction dashboards
# @app.route('/predictions')
# def predictions_dashboard():
#     # Ensure models are trained and evaluation metrics are ready for display
#     electricity_results = train_electricity_prediction_model()
#     event_classification_results = train_event_classification_model()

#     return render_template('predictions.html',
#                            electricity_results=electricity_results,
#                            event_classification_results=event_classification_results)

# # Separate AJAX endpoint for Electricity Usage Prediction POST
# # In app.py
# @app.route('/predict_electricity_ajax', methods=['POST'])
# def predict_electricity_ajax():
#     if electricity_model is None:
#         print("Electricity model not available for prediction, re-training...")
#         train_electricity_prediction_model()

#     try:
#         data = request.json
#         print(f"Received data for electricity prediction: {data}") # <--- ADD THIS
#         hour = int(data['hour'])
#         dayofweek = int(data['dayofweek'])
#         month = int(data['month'])
#         water_consumption = float(data['water_consumption'])
#         traffic_density = int(data['traffic_density'])
#         waste_level = float(data['waste_level'])

#         input_data = pd.DataFrame([[hour, dayofweek, month, water_consumption, traffic_density, waste_level]],
#                                   columns=['Hour', 'DayOfWeek', 'Month', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)'])
#         print(f"Input DataFrame for prediction: \n{input_data}") # <--- ADD THIS

#         prediction = electricity_model.predict(input_data)[0]
#         return jsonify({"prediction": f"{prediction:.2f} kW"})
#     except Exception as e:
#         print(f"Error in predict_electricity_ajax: {e}") # <--- ADD THIS
#         return jsonify({"error": str(e)}), 400



# # @app.route('/predict_electricity_ajax', methods=['POST'])
# # def predict_electricity_ajax():
# #     if electricity_model is None:
# #         print("Electricity model not available for prediction, re-training...")
# #         train_electricity_prediction_model()

# #     try:
# #         data = request.json
# #         hour = int(data['hour'])
# #         dayofweek = int(data['dayofweek'])
# #         month = int(data['month'])
# #         water_consumption = float(data['water_consumption'])
# #         traffic_density = int(data['traffic_density'])
# #         waste_level = float(data['waste_level'])

# #         input_data = pd.DataFrame([[hour, dayofweek, month, water_consumption, traffic_density, waste_level]],
# #                                   columns=['Hour', 'DayOfWeek', 'Month', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)'])

# #         prediction = electricity_model.predict(input_data)[0]
# #         return jsonify({"prediction": f"{prediction:.2f} kW"})
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 400

# # Separate AJAX endpoint for Event Classification POST
# # ... (your existing imports and global variables)

# @app.route('/classify_event_ajax', methods=['POST'])
# def classify_event_ajax():
#     global event_label_encoder
#     if event_classifier_model is None or event_label_encoder is None:
#         print("Event classification model or encoder not available, re-training...")
#         train_event_classification_model()
#         # Ensure encoder is loaded/re-created after training
#         if os.path.exists(LABEL_ENCODER_PATH):
#             try:
#                 event_label_encoder = joblib.load(LABEL_ENCODER_PATH)
#             except Exception as e:
#                 print(f"Error loading event_label_encoder after re-training: {e}")
#                 # Fallback: if loading fails, try to re-initialize and fit
#                 load_and_preprocess_data() # This will recreate and save the encoder

#     try:
#         data = request.json
#         print(f"Received data for event classification: {data}") # <--- ADD THIS
#         electricity = float(data['electricity_usage'])
#         water = float(data['water_consumption'])
#         traffic = int(data['traffic_density'])
#         waste = float(data['waste_level'])

#         input_data = pd.DataFrame([[electricity, water, traffic, waste]],
#                                   columns=['Electricity Usage (kW)', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)'])
#         print(f"Input DataFrame for classification: \n{input_data}") # <--- ADD THIS

#         predicted_label_index = event_classifier_model.predict(input_data)[0]
#         print(f"Predicted label index: {predicted_label_index}") # <--- ADD THIS

#         # Use a try-except here for robustness during inverse_transform for prediction
#         try:
#             # Ensure event_label_encoder is not None before inverse_transform
#             if event_label_encoder is not None:
#                 predicted_event = event_label_encoder.inverse_transform([predicted_label_index])[0]
#             else:
#                 predicted_event = "Error: Label encoder not initialized."
#                 print("Error: event_label_encoder is None in classify_event_ajax")
#         except ValueError as ve:
#             predicted_event = f"Unknown Event (label {predicted_label_index}): {ve}"
#             print(f"Warning: {predicted_event}")

#         return jsonify({"prediction": predicted_event})
#     except Exception as e:
#         print(f"Error in classify_event_ajax: {e}") # <--- ADD THIS
#         return jsonify({"error": str(e)}), 400

# # ... (rest of your app.py code)


# # @app.route('/classify_event_ajax', methods=['POST'])
# # def classify_event_ajax():
# #     global event_label_encoder
# #     if event_classifier_model is None or event_label_encoder is None:
# #         print("Event classification model or encoder not available, re-training...")
# #         train_event_classification_model()
# #         if os.path.exists(LABEL_ENCODER_PATH):
# #             event_label_encoder = joblib.load(LABEL_ENCODER_PATH)

# #     try:
# #         data = request.json
# #         electricity = float(data['electricity_usage'])
# #         water = float(data['water_consumption'])
# #         traffic = int(data['traffic_density'])
# #         waste = float(data['waste_level'])

# #         input_data = pd.DataFrame([[electricity, water, traffic, waste]],
# #                                   columns=['Electricity Usage (kW)', 'Water Consumption (Liters)', 'Traffic Density (Vehicles/Min)', 'Waste Level (%)'])

# #         predicted_label_index = event_classifier_model.predict(input_data)[0]
# #         try:
# #             predicted_event = event_label_encoder.inverse_transform([predicted_label_index])[0]
# #         except ValueError as ve:
# #             predicted_event = f"Unknown Event (label {predicted_label_index}): {ve}"
# #             print(f"Warning: {predicted_event}")

# #         return jsonify({"prediction": predicted_event})
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 400


# @app.route('/detect_anomalies')
# def detect_anomalies():
#     anomalies_html = detect_waste_anomalies()
#     return render_template('anomaly.html', anomalies=anomalies_html)

# if __name__ == '__main__':
#     app.run(debug=True)


