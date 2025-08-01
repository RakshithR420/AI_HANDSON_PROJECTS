import { useState } from 'react'
import './App.css'

function App() {
  const [company, setCompany] = useState('')
  const [date, setDate] = useState('')
  const [direction, setDirection] = useState('up')
  const [response, setResponse] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setResponse("Loading...")

    try {
      const res = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company, date, direction })
      })
      const data = await res.json()
      setResponse(data.result || data.message || "No result.")
    } catch (error) {
      setResponse("Failed to connect to server.")
    }
  }

  return (
    <div className="app-container">
      <h1>📈 Financial News Analyst</h1>
      <form onSubmit={handleSubmit} className="form">
        <input
          type="text"
          placeholder="Company Symbol (e.g., AAPL)"
          value={company}
          onChange={(e) => setCompany(e.target.value)}
          required
        />
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          required
        />
        <select value={direction} onChange={(e) => setDirection(e.target.value)}>
          <option value="up">Up</option>
          <option value="down">Down</option>
        </select>
        <button type="submit">🔍 Analyze</button>
      </form>

      {response && (
        <div className="response-box">
          <h3>📝 Result:</h3>
          <p>{response}</p>
        </div>
      )}
    </div>
  )
}

export default App
