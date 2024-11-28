import argparse
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import openai
import os
from dotenv import load_dotenv
from get_embedding_function import get_embedding_function
from flask import Flask, render_template, request, jsonify
from flask import session
import secrets


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

chroma = "chroma"
PROMPT_TEMPLATE= """
Tu es un assitant viruel de Direct Assurance et tu parle avec un client de l'entreprise qui pose la question .Répond en se basant seulement sur cette contexte:
{context}
---
Conversation passée:
{history}
Essaie de parler avec la meme langage du client
répond à cette question d'une manière pertinente et précise et sans trop agrandir la réponse: {question}
"""

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]
        history = session.get('history', [])
        
        # Ajouter le nouveau message de l'utilisateur à l'historique
        history.append(f"User: {msg}")
        
        # Mettre à jour l'historique dans la session
        session['history'] = history
        response = query_rag(msg,history)
        history.append(f"Assistant: {response}")
        session['history'] = history
        return response
    return jsonify({"reply": "Méthode non autorisée"})


def query_rag(query_text: str,history: list):
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=chroma, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        history_text = "\n".join(history[-10:])
        prompt = prompt_template.format(context=context_text,history=history_text, question=query_text)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Une erreur s'est produite. Veuillez réessayer."
if __name__ == "__main__":
    app.run(debug=True)
