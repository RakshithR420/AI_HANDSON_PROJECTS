# 📰 News Recommendation App

A simple React-based News Recommendation application with a Python (Flask) backend.

## 📁 Project Structure

- **Frontend**: React (Vite)  
  Files: `App.jsx`, `NewsList.jsx`, `NewsItem.jsx`, `main.jsx`, `App.css`, `index.css`

- **Backend**: Python (Flask)  
  File: `app.py`

---

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone https://github.com/RakshithR420/NewsRecommendation.git
cd NewsRecommendation
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

🐍 Starting the Backend (Python/Flask)

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

Make sure your React app fetches from this backend (adjust CORS if needed).

🧾 Usage
Open your browser to http://localhost:5173

Browse dynamically displayed news

Modify NewsList.jsx or NewsItem.jsx to update behavior or styling

Connect your frontend to the Python backend for dynamic content

🛠️ Built With
1. React.js

2. Vite (optional bundler)

3. Flask (Python backend)

4. TailwindCSS

5. HTML5 / CSS3

📝 To Do
1. Add filtering options (category, date, etc.)

2. Integrate a real news API (e.g., NewsAPI, GNews)

