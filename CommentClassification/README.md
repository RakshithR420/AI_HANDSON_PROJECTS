# Comment Classification Model

A multi-label comment classification system using BERT, built with PyTorch and Flask for the backend, and a lightweight React frontend. The model predicts whether a given comment is toxic, abusive, provocative, obscene, hate speech, or racist.

## 📦 Features

- Predicts comment toxicity using a trained ML model
- Classifies a comment across 6 toxicity-related labels.
- Fine-tuned bert-base-uncased model.
- Handles JSON input for predictions
- Clean separation between model and application logic
- Optional React frontend for user interface

## Tech Stack
### Backend:
- Python
- PyTorch
- Transformers (Hugging Face)
- Flask + Flask-CORS

### Frontend:
- React
- Vite

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone https://github.com/RakshithR420/AI_HANDSON_PROJECTS
cd CommentClassification
```
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
3. Train or Lod the model
```
python model.py
```
4. Run the backend server
```
python app.py

```
By default, Flask will run at:
🔗 http://localhost:5000


# 📝 To Do
- Use a larger, balanced dataset for better generalization.
- Integrate database to store predictions and user feedback.
