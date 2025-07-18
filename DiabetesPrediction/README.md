# 🩺 Diabetes Prediction Model

This is a simple Machine Learning project built with Python and Flask that predicts the likelihood of diabetes based on user-provided input data.

## 📦 Features

- Predicts diabetes using a trained machine learning model
- Flask-based API backend
- Handles JSON input for predictions
- Clean separation between model and application logic
- Optional React frontend for user interface

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone https://github.com/RakshithR420/DiabetesPrediction.git
cd DiabetesPrediction
```
# 🧠 Model Details
- Built a custom `NeuralNetwork` class using pure Python (no high-level ML libraries)
- Trained on the **PIMA Indian Diabetes Dataset**
- Model training and logic are encapsulated inside `model.py`
- Model is saved using `joblib` or `pickle` for reuse
- The `predict()` function handles prediction logic based on user input

# ⚛️ Starting the Frontend (React)
1. Install Node dependencies
Make sure Node.js and npm or yarn are installed.

```
npm install
# or
yarn install
```

2. Run the React app

```
npm run dev
# or
yarn dev
```

By default, the app will run on:
🔗 http://localhost:5173

# 🐍 Starting the Backend (Python/Flask)

1. Set up Python environment
2. Install Python dependencies

```
pip install -r requirements.txt

```
3. Run the backend server
```
python app.py

```
By default, Flask will run at:
🔗 http://localhost:5000

# 🛠️ Built With
1. Python
2. Flask
3. Scikit-learn
4. Pandas
5. NumPy
6. joblib / pickle
7. React (optional frontend)

# 📝 To Do
- Add option to reset the form
- Connect a basic database to store past predictions

