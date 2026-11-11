import torch
import joblib
import re
import emoji
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer

# === Classe del modello ===
# Uguale al modello usato nel notebook

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Inizializza il classificatore softmax con Batch Normalization.

        Args:
            input_dim: Dimensione dell'input (numero di features TF-IDF)
            output_dim: Numero di classi (3 per negativo, neutro, positivo)
        """
        super(SoftmaxClassifier, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),  # Batch Normalization per stabilizzare l'addestramento
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout più alto per TF-IDF per evitare overfitting
            
            
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)




# === Funzione di pulizia testo ===
# Uguale alla funzione usata nel notebook

def clean_text(text, keep_emojis=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(?![ia]\b)\w\b', '', text)

    if keep_emojis:
        text = emoji.demojize(text, language='it')
        text = re.sub(r':', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
    else:
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text




# === Caricamento degli oggetti salvati ===
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")




# Inizializza e carica il modello
input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)
model = SoftmaxClassifier(input_dim, output_dim)
model.load_state_dict(torch.load("model.pth"))
model.eval()




# === Inizio ===
print("== Sentiment Classifier CLI ==")
print("Scrivi una frase in inglese da classificare (CTRL+C per uscire).")

while True:
    try:
        text = input("\n>> ")
        cleaned = clean_text(text)
        tfidf_vec = vectorizer.transform([cleaned])
        tensor = torch.FloatTensor(tfidf_vec.toarray())

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()
            label = label_encoder.inverse_transform([pred_idx])[0]

        print(f"[Sentiment]: {label} ({probs[pred_idx]*100:.2f}%)")
        print("Probabilità per ciascuna classe:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f" - {class_name}: {probs[i]*100:.2f}%")

    except KeyboardInterrupt:
        print("\nUscita dal programma.")
        break
