from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from llama_index.core import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os



client = Groq(api_key = os.getenv)

pdf_path = "artificial_intelligence_tutorial.pdf"
read = SimpleDirectoryReader(input_files=[pdf_path])
documents = read.load_data()
all_text = "\n\n".join(doc.text for doc in documents)
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
lc_docs = splitter.create_documents([all_text])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(lc_docs, embeddings)

def getReply(user_prompt):
    retrieved_docs = vector_store.similarity_search(user_prompt, k=3)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages = [
        {
            "role": "system", "content": "You are a smart and witty assistant that replies in a sarcastic manner.The replies given must be true and clear."
        },
        {
            "role": "user", "content": f"""
            context:{context}
            question:{user_prompt}
            """
        }
    ]
    )
    return chat_completion.choices[0].message.content

    
app = Flask(__name__)
CORS(app)  

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/submit-prompt', methods=['POST'])
def submit_prompt():
    data = request.get_json()
    prompt = data.get('prompt', '')
    message = getReply(prompt)
    print(message)
    return jsonify({'response' : message})



if __name__ == '__main__':
    app.run(debug=True,port=5000)