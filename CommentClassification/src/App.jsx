import { useState } from "react";
import "./App.css";

function App() {
  const [comment, setComment] = useState("");
  const [results, setResults] = useState([])

  const handleSubmit = async (e) => {

    e.preventDefault();

    try {
      const response = await fetch("http://127.0.0.1:5000/classify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ comment })
      })
      if (response.ok) {
        const data = await response.json();
        setResult(data);
      } else {
        console.error("Error:", response.statusText);
      }
    }
    catch(err){
    console.error(error)

  }
  }
  

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h2>Toxic Comment Classifier</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          rows="4"
          cols="50"
          placeholder="Enter a comment..."
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
        <br />
        <button type="submit">Classify</button>
      </form>

      {result.length > 0 && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Classification Results:</h3>
          <ul>
            {result.map((res, index) => (
              <li key={index}>
                {res.label}: {res.predicted} (Confidence: {res.prob.toFixed(2)})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
