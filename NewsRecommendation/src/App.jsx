import { useState } from "react";
import "./App.css";
import NewsList from "./NewsList";

function App() {
  const [query, setQuery] = useState("N19639");
  const [results, setResults] = useState(null);
  const [num, setNum] = useState(5);
  const [search, setSearch] = useState("");
  const [show, setShow] = useState(false);
  const handleSearch = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:5001/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: search,
        }),
      });
      const data = await response.json();
      console.log(data);
      setResults({
        selected_article: [],
        recommendation: data,
      });
    } catch (err) {
      console.error(err);
    } finally {
      setSearch("");
      setShow(false);
    }
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:5001/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          news_id: query,
          num_recommendations: num,
        }),
      });
      const data = await response.json();
      setResults(data.data);
      console.log(data.data);
    } catch (err) {
      console.error(err);
    } finally {
      setQuery("");
      setShow(true);
    }
  };
  return (
    <div className="flex flex-col gap-10 justify-center items-center">
      <h1>Top Related News</h1>
      <div className="w-full max-w-3xl mx-auto p-8">
        <form onSubmit={handleSearch} className="mb-4 flex gap-4">
          <input
            type="text"
            placeholder="What are you looking for..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 border border-gray-300 py-4 px-5 text-lg rounded-lg focus:border-lime-400 focus:outline-none"
          />
          <button
            type="submit"
            className="bg-lime-500 text-white font-semibold py-3 px-5 rounded-lg hover:bg-lime-600 transition-colors"
          >
            Find
          </button>
        </form>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-4 items-center">
        <input
          type="text"
          placeholder="Enter News ID..."
          value={query}
          onChange={(el) => setQuery(el.target.value)}
          className="border border-gray-300 p-3 rounded-lg focus:border-lime-400
        focus:outline-none"
        />
        <input
          type="number"
          placeholder="#"
          value={num}
          onChange={(el) => setNum(el.target.value)}
          className="border border-gray-300 p-3 rounded-lg focus:border-lime-400 focus:outline-none w-24"
        />
        <button
          type="submit"
          className="bg-lime-500 text-white font-semibold py-3 px-5 rounded-lg hover:bg-lime-600 transition-colors"
        >
          Find
        </button>
      </form>

      {results && (
        <div className="space-y-4">
          {show && (
            <div>
              <h1 className="text-left">Current Article:</h1>
              <NewsList recommended_list={results.selected_article} />
            </div>
          )}

          <h1 className="text-left">Related Articles:</h1>
          <NewsList recommended_list={results.recommendation} />
        </div>
      )}
    </div>
  );
}

export default App;
