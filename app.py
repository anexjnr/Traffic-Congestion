from flask import Flask, render_template, request
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv("traffic_congestion.csv")

# Display index numbers for object columns
label_encoders = {}
for column in ['Time', 'Weather', 'City']:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

# Define X and y
X = dataset[['Day', 'Time', 'Weather', 'City', 'Temperature']]
y = dataset[['Traffic Congestion', 'Number of Vehicles']]

# Feature Scaling and One-Hot Encoding
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Time', 'Weather', 'City']),
        ('scaler', StandardScaler(), ['Day', 'Temperature'])
    ],
    remainder='passthrough'
)

X_transformed = ct.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Separate Linear Regression models for Traffic Congestion and Number of Vehicles
regressor_lr_congestion = LinearRegression()
regressor_lr_vehicles = LinearRegression()

# Train the models
regressor_lr_congestion.fit(X_train, y_train['Traffic Congestion'])
regressor_lr_vehicles.fit(X_train, y_train['Number of Vehicles'])

# SVR model for Traffic Congestion
regressor_svr_congestion = SVR(kernel='rbf', C=1.0, epsilon=0.1)
regressor_svr_congestion.fit(X_train, y_train['Traffic Congestion'])

# SVR model for Number of Vehicles
regressor_svr_vehicles = SVR(kernel='rbf', C=1.0, epsilon=0.1)
regressor_svr_vehicles.fit(X_train, y_train['Number of Vehicles'])

# Neural Network for Traffic Congestion
regressor_nn_congestion = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
regressor_nn_congestion.fit(X_train, y_train['Traffic Congestion'])

# Neural Network for Number of Vehicles
regressor_nn_vehicles = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
regressor_nn_vehicles.fit(X_train, y_train['Number of Vehicles'])


def predict_ensemble(user_input):
    # Label encoding and feature scaling for user input
    user_input_transformed = ct.transform(pd.DataFrame(user_input))

    # Make predictions using each model
    y_pred_lr_congestion = regressor_lr_congestion.predict(user_input_transformed)
    y_pred_lr_vehicles = regressor_lr_vehicles.predict(user_input_transformed)

    # Make predictions using each model
    y_pred_svr_congestion = regressor_svr_congestion.predict(user_input_transformed.reshape(1, -1))
    y_pred_svr_vehicles = regressor_svr_vehicles.predict(user_input_transformed.reshape(1, -1))
    y_pred_nn_congestion = regressor_nn_congestion.predict(user_input_transformed.reshape(1, -1))
    y_pred_nn_vehicles = regressor_nn_vehicles.predict(user_input_transformed.reshape(1, -1))

    # Ensemble by taking the average
    y_pred_ensemble_congestion = (y_pred_lr_congestion[0] + y_pred_svr_congestion[0] + y_pred_nn_congestion[0]) / 3.0
    y_pred_ensemble_vehicles = (y_pred_lr_vehicles[0] + y_pred_svr_vehicles[0] + y_pred_nn_vehicles[0]) / 3.0

    return int(y_pred_nn_congestion), int(y_pred_nn_vehicles)

def map_traffic_congestion(y_ensemble_congestion):
    if y_ensemble_congestion == 1:
        return "Very Less Traffic"
    elif y_ensemble_congestion == 2:
        return "Less Traffic"
    elif y_ensemble_congestion == 3:
        return "Moderate Traffic"
    elif y_ensemble_congestion == 4:
        return "High Traffic"
    elif y_ensemble_congestion == 5:
        return "Very High Traffic"
    else:
        return "Unknown Traffic"


def save_plot_to_file(fig, filename):
    output_path = os.path.join('static', 'images', filename)  # Save images in the 'static/images' folder
    fig.savefig(output_path, format='png')
    plt.close(fig)
    return output_path


def save_plot_to_base64(fig):
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return base64.b64encode(output.getvalue()).decode('utf-8')


def plot_predictions(y_test, y_pred, model_name):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='red', label=model_name)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{model_name} - Actual vs Predicted')
    ax.legend()
    return fig


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Get user inputs
    user_input = {
        'Day': [int(request.form['day'])],
        'Time': [label_encoders['Time'].transform([request.form['time']])[0]],
        'Weather': [label_encoders['Weather'].transform([request.form['weather']])[0]],
        'Temperature': [float(request.form['temperature'])],
        'City': [label_encoders['City'].transform([request.form['city']])[0]],
    }

    # Make predictions using the ensemble
    y_ensemble_congestion, y_ensemble_vehicles = predict_ensemble(user_input)

    # Make predictions using each model
    y_pred_lr_congestion = regressor_lr_congestion.predict(X_test)
    y_pred_svr_congestion = regressor_svr_congestion.predict(X_test)
    y_pred_nn_congestion = regressor_nn_congestion.predict(X_test)

    y_pred_lr_vehicles = regressor_lr_vehicles.predict(X_test)
    y_pred_svr_vehicles = regressor_svr_vehicles.predict(X_test)
    y_pred_nn_vehicles = regressor_nn_vehicles.predict(X_test)

        # Calculate accuracy for each model
    r2_lr_congestion = regressor_lr_congestion.score(X_test, y_test['Traffic Congestion'])
    r2_lr_vehicles = regressor_lr_vehicles.score(X_test, y_test['Number of Vehicles'])
    r2_svr_congestion = regressor_svr_congestion.score(X_test, y_test['Traffic Congestion'])
    r2_svr_vehicles = regressor_svr_vehicles.score(X_test, y_test['Number of Vehicles'])
    r2_nn_congestion = regressor_nn_congestion.score(X_test, y_test['Traffic Congestion'])
    r2_nn_vehicles = regressor_nn_vehicles.score(X_test, y_test['Number of Vehicles'])
    accuracy_lr_congestion = (r2_lr_congestion + 1) / 2
    accuracy_lr_vehicles = (r2_lr_vehicles + 1) / 2
    accuracy_svr_congestion = (r2_svr_congestion + 1) / 2
    accuracy_svr_vehicles = (r2_svr_vehicles + 1) / 2
    accuracy_nn_congestion = (r2_nn_congestion + 1) / 2
    accuracy_nn_vehicles = (r2_nn_vehicles + 1) / 2

    y_acc_ensemble_congestion = ((accuracy_lr_congestion * 100) + (accuracy_svr_congestion * 100) + (accuracy_nn_congestion * 100)) / 3.0
    y_acc_ensemble_vehicles = ((accuracy_lr_vehicles * 100) + (accuracy_svr_vehicles * 100) + (accuracy_nn_vehicles * 100)) / 3.0

    # Visualizations
    fig_lr_congestion = plot_predictions(y_test['Traffic Congestion'], y_pred_lr_congestion, 'Linear Regression Congestion')
    img_lr_congestion = save_plot_to_file(fig_lr_congestion, 'lr_visualization_congestion.png')

    fig_svr_congestion = plot_predictions(y_test['Traffic Congestion'], y_pred_svr_congestion, 'SVR Congestion')
    img_svr_congestion = save_plot_to_file(fig_svr_congestion, 'svr_visualization_congestion.png')

    fig_nn_congestion = plot_predictions(y_test['Traffic Congestion'], y_pred_nn_congestion, 'Neural Network Congestion')
    img_nn_congestion = save_plot_to_file(fig_nn_congestion, 'nn_visualization_congestion.png')

    fig_lr_vehicles = plot_predictions(y_test['Number of Vehicles'], y_pred_lr_vehicles, 'Linear Regression Vehicles')
    img_lr_vehicles = save_plot_to_file(fig_lr_vehicles, 'lr_visualization_vehicles.png')

    fig_svr_vehicles = plot_predictions(y_test['Number of Vehicles'], y_pred_svr_vehicles, 'SVR Vehicles')
    img_svr_vehicles = save_plot_to_file(fig_svr_vehicles, 'svr_visualization_vehicles.png')

    fig_nn_vehicles = plot_predictions(y_test['Number of Vehicles'], y_pred_nn_vehicles, 'Neural Network Vehicles')
    img_nn_vehicles = save_plot_to_file(fig_nn_vehicles, 'nn_visualization_vehicles.png')

    # Render the result page with MSE, predictions, and visualizations
    return render_template(
        'result.html',
        y_ensemble_congestion=map_traffic_congestion(y_ensemble_congestion),
        y_ensemble_vehicles=y_ensemble_vehicles,
	y_acc_ensemble_congestion=y_acc_ensemble_congestion,
	y_acc_ensemble_vehicles=y_acc_ensemble_vehicles,
        img_lr_congestion=img_lr_congestion,
        img_svr_congestion=img_svr_congestion,
        img_nn_congestion=img_nn_congestion,
        img_lr_vehicles=img_lr_vehicles,
        img_svr_vehicles=img_svr_vehicles,
        img_nn_vehicles=img_nn_vehicles)


if __name__ == '__main__':
    app.run(debug=True)