from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
from flask import Flask
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load models
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load documents
with open('cleaned_processed_text.txt', 'r', encoding="utf-8") as file:
    lemmatized_text = file.read()

# Split documents into sentences or paragraphs
documents = lemmatized_text.split('\n')

# Encode documents for retrieval
document_embeddings = retriever_model.encode(documents)

# Define helper functions
def retrieve(query, top_n=5):
    query_embedding = retriever_model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)
    best_indices = np.argsort(similarities[0])[::-1][:top_n]

    results = [(documents[idx], similarities[0][idx]) for idx in best_indices
               if similarities[0][idx] > 0.4 and len(documents[idx]) > 50]
    return [doc for doc, score in results]

def preprocess_retrieved_docs(docs):
    cleaned_docs = [doc for doc in docs if "Chapter" not in doc and not doc.strip().isdigit()]
    return cleaned_docs

def generate_answer(query, retrieved_docs):
    # Clean retrieved documents
    context = " ".join(preprocess_retrieved_docs(retrieved_docs))
    
    prompts = [
    f"Answer based on the following context and provide insights.\n\nContext: {context}\n\nQuestion: {query}\nAnswer:",
    f"Provide a concise, well-rounded answer to the question below.\n\nContext: {context}\n\nQuestion: {query}\nAnswer:",
    f"As a risk management expert, answer in a way that includes examples if relevant.\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    ]
    input_text = random.choice(prompts)


    inputs = generator_tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    # Update the generation parameters for diverse answers
    outputs = generator_model.generate(
        **inputs,
        max_length=250,
        num_beams=5,
        no_repeat_ngram_size=2,
        temperature=0.8,  # Increased temperature for more randomness
        top_k=30,         # Narrow top-k sampling
        top_p=0.9,        # Narrow top-p sampling
        do_sample=True    # Enable sampling for varied responses
    )

    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def preprocess_query(query):
    risk_topics = {
        "who": "responsible parties, accountability, roles",
        "what": "definition, description, explanation",
        "how": "process, methodology, implementation",
        "when": "timeline, frequency, schedule",
        "why": "purpose, reasoning, justification",
        "where": "location, scope, application",
        "monte carlo": "risk quantification, probabilistic analysis",
        "project management": "risk planning, risk assessment"
    }

    query_words = query.lower().split()
    for word in query_words:
        if word in risk_topics:
            query = f"{query} {risk_topics[word]}"

    return query

def interactive_qa(query):
    enhanced_query = preprocess_query(query)
    retrieved_docs = retrieve(enhanced_query)
    if not retrieved_docs:
        return "❌ No relevant information found. Please try rephrasing your question."
    response = generate_answer(query, retrieved_docs)
    return response

# Define API endpoint
@app.route('/get_answer', methods=['POST'])
def get_answer():
    
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = interactive_qa(question)
    return jsonify({"answer": answer})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
