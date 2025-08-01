from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from groq import Groq  
from datetime import datetime
import finnhub
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    company = data.get("company")
    date = data.get("date")
    direction = data.get("direction")

    if not company or not date or not direction:
        return jsonify({"error": "Missing input"}), 400

    current_datetime = datetime.now().date()
    finnhub_client = finnhub.Client(api_key=os.getenv("FINHUB_API_KEY"))

    try:
        datas = finnhub_client.company_news(
            company,
            _from=date,
            to=str(current_datetime)
        )
    except Exception as e:
        return jsonify({"error": f"Error fetching news: {e}"}), 500

    tot_summary = ""
    for article in datas:
        description = article.get("summary", "")
        if description:
            tot_summary += description + "\n\n"

    if not tot_summary:
        return jsonify({"message": "There is no recent news related to this price movement."})

    chunks = splitter.split_text(tot_summary)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_texts(chunks, embedding=embeddings)

    user_text = f"{company} {direction} {date}"
    retrieved_docs = vector_db.similarity_search(user_text, k=3)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    rag_system_prompt = """
    You are a financial analyst. Analyze the context and give possible reasons for the stock price direction in the user questions.
    Only use the context and give the context for which the results were derived do not give the context in the answer
    """

    rag_user_prompt = f"""
    Context:
    {context}

    Question:
    {company} {direction} {date}
    """

    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": rag_system_prompt.strip()},
                {"role": "user", "content": rag_user_prompt.strip()},
            ],
            temperature=0,
            max_tokens=300,
        )
        return jsonify({"result": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": f"Groq API error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
