# Crop Recommendation System 🌱

A machine learning-based web application that recommends the most suitable crops to grow based on various soil and climate parameters.

## Features 🌟

- Predicts the best crop to grow based on:
  - Nitrogen (N) content in soil
  - Phosphorus (P) content in soil
  - Potassium (K) content in soil
  - Temperature
  - Humidity
  - pH value
  - Rainfall
- Provides top 3 crop recommendations with confidence scores
- Interactive web interface
- RESTful API endpoint for predictions
- Model performance metrics and evaluation

## Tech Stack 💻

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Random Forest Classifier
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Vercel

## Prerequisites 📋

- Python 3.7+
- pip (Python package installer)

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/Sameeksha804/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage 💡

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8080
```

3. Enter the soil and climate parameters in the form to get crop recommendations.

## API Usage 📡

You can also use the prediction API endpoint:

```bash
curl -X POST http://localhost:8080/predict \
-H "Content-Type: application/json" \
-d '{
    "N": 35,
    "P": 42,
    "K": 53,
    "temperature": 38,
    "humidity": 38,
    "ph": 6,
    "rainfall": 46
}'
```

## Model Details 🤖

- **Algorithm**: Random Forest Classifier
- **Features**: 15 engineered features including:
  - Basic soil parameters (N, P, K)
  - Climate parameters (temperature, humidity, rainfall)
  - pH value
  - Interaction features (NP ratio, NK ratio, PK ratio)
  - Polynomial features
- **Training Data**: Combined dataset from multiple sources
- **Cross-validation**: 10-fold cross-validation
- **Performance Metrics**: Accuracy, Classification Report, Confusion Matrix

## Project Structure 📁

```
Crop-Recommendation-System/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel deployment configuration
├── data/                 # Dataset directory
│   ├── Crop_recommendation.csv
│   └── Crop_recommendation1.csv
├── model/                # Trained model files
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── evaluation_metrics.json
└── templates/            # HTML templates
    └── index.html
```

## Deployment 🌐

The application is configured for deployment on Vercel. The deployment process is automated through GitHub integration.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Author 👩‍💻

Sameeksha804

## Acknowledgments 🙏

- Dataset providers
- Scikit-learn team
- Flask framework
- Vercel platform 