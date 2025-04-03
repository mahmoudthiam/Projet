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
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📌 Charger le modèle NLP
model = SentenceTransformer("all-MiniLM-L6-v2")

def lire_pdfs(dossier):
    texte_total = []
    for fichier in os.listdir(dossier):
        if fichier.endswith(".pdf"):
            chemin = os.path.join(dossier, fichier)
            doc = fitz.open(chemin)
            for page in doc:
                texte_total.append(page.get_text("text"))
    return texte_total

# 📌 Charger les documents
dossier_pdfs = "C:/Users/thiam/Desktop/vision2050/"  # Remplace par ton dossier PDF
documents = lire_pdfs(dossier_pdfs)
texte_corpus = " ".join(documents)

# 🔥 Encoder les documents
doc_embeddings = model.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]

# 📌 Index FAISS
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def repondre_question(question):
    documents_list = texte_corpus.split(". ")
    vect = TfidfVectorizer()
    tfidf_matrix = vect.fit_transform(documents_list + [question])
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    # Récupérer les indices des 3 meilleures phrases
    top_indices = np.argsort(scores[0])[-50:][::-1]  # Sélection des 3 meilleures réponses
    #meilleure_reponse = documents_list[scores.argmax()]
    # Combiner les phrases pour avoir une réponse plus complète
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

# Assure-toi que ce bloc est bien exécuté et accessible pour Shiny.
