#python -m pip install --user pymupdf sentence-transformers faiss-cpu numpy tensorflow tf-keras
#numpy==1.23.5 pandas torch transformers

#Amélioration avec un modèle de langage avancé
#python -m pip install --user pymupdf sentence-transformers faiss-cpu numpy tensorflow tf-keras

#C:\Users\thiam\AppData\Local\Programs\Python\Python311\python.exe --version
#pip install --upgrade transformers
#Solution temporaire (juste pour la session actuelle)
#$env:Path="C:\Users\thiam\AppData\Local\Programs\Python\Python311;" + $env:Path

#python -m pip install --user pymupdf
#python -m pip install notebook

#C:/Users/thiam/Desktop/IATech/
#python -m shiny run --reload app.py

#python -m pip install --upgrade pip
#python -m pip install --user rsconnect-python
# py -3.11 -m shiny run --reload app.py

#pip install -r requirements.txt

from shiny import App, ui, render
import fitz  # PyMuPDF
import os
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📌 Charger le modèle NLP
model = SentenceTransformer("all-MiniLM-L6-v2")

# 📌 URL du fichier PDF sur GitHub
PDF_URL = "https://raw.githubusercontent.com/mahmoudthiam/Projet/main/Strategie-Nationale-de-Developpement-2025-2029.pdf"
LOCAL_PDF = "document.pdf"

# 📌 Télécharger le PDF depuis GitHub
def telecharger_pdf():
    response = requests.get(PDF_URL)
    with open(LOCAL_PDF, "wb") as f:
        f.write(response.content)

# 📌 Lire le contenu du PDF
def lire_pdf(fichier):
    texte_total = []
    doc = fitz.open(fichier)
    for page in doc:
        texte_total.append(page.get_text("text"))
    return texte_total

# 📌 Charger les documents
telecharger_pdf()
documents = lire_pdf(LOCAL_PDF)
texte_corpus = " ".join(documents)

# 🔥 Encoder les documents
doc_embeddings = model.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]

# 📌 Index FAISS
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# 📌 Fonction de réponse aux questions
def repondre_question(question):
    documents_list = texte_corpus.split(". ")
    vect = TfidfVectorizer()
    tfidf_matrix = vect.fit_transform(documents_list + [question])
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_indices = np.argsort(scores[0])[-50:][::-1]  # Sélection des 50 meilleures réponses
    meilleure_reponse = ". ".join([documents_list[i] for i in top_indices])
    return meilleure_reponse

# 🎨 Interface stylée
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background-color: #121212; color: white; font-family: Arial, sans-serif; }
        .chat-container { max-width: 600px; margin: auto; padding: 20px; }
        .chat-box { background: #1E1E1E; padding: 15px; border-radius: 10px; }
        .user-input { width: 100%; padding: 10px; margin-top: 10px; border-radius: 5px; border: none; }
        .send-btn { background: #00A86B; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }
    """),
    ui.div(
        ui.h2("IATech - SENEGAL 2050"),
        ui.tags.div(
            ui.input_text("question", "Votre question :", placeholder="Tapez votre question ici..."),
            class_="user-input"
        ),
        ui.input_action_button("send", "Envoyer", class_="send-btn"),
        ui.output_text("response"),
        class_="chat-container"
    )
)

# 🚀 Serveur de l'application
def server(input, output, session):
    @output
    @render.text
    def response():
        if input.send():
            question = input.question()
            if question:
                return repondre_question(question)
            else:
                return "Veuillez poser une question."

# 🚀 Lancer l'application
app = App(app_ui, server)
