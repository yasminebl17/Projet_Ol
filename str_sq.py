import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

# Initialisation de la page Streamlit
st.set_page_config(page_title="Détection d'Intrusion", layout="wide")

st.markdown("<h2 style='color:#00ffcc;'>🔐 Détection séquentielle des attaques en temps réel </h2>", unsafe_allow_html=True)

# Zone de texte pour les logs
log_zone = st.empty()

def log(message, alert=False):
    if alert:
        log_zone.markdown(f"<p style='color:red'>{message}</p>", unsafe_allow_html=True)
    else:
        log_zone.markdown(f"<p style='color:white'>{message}</p>", unsafe_allow_html=True)

# Simulation de détection
def run_simulation():
    start_time = time.time()
    try:
        data = pd.read_csv("KDDTest+.csv")
    except FileNotFoundError:
        log("❌ Fichier 'KDDTest+.csv' introuvable !", alert=True)
        return

    data.dropna(inplace=True)

    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    historique = pd.DataFrame(columns=data.columns)

    true_labels = []
    predicted_labels = []

    ATTACK_CLASSES = [0]
    alerte_count = 0

    for i in range(10):
        st.markdown(f"<h4 style='color:#00ffcc;'>📦 Nouveau lot de données (itération {i + 1})</h4>", unsafe_allow_html=True)

        echantillon = data.sample(n=2100).reset_index(drop=True)
        X_new = scaler.transform(echantillon.iloc[:, :-1])
        predictions = knn.predict(X_new)

        echantillon["Prediction"] = predictions

        true_labels.extend(echantillon.iloc[:, -2])
        predicted_labels.extend(predictions)

        nb_attacks = sum(pred in ATTACK_CLASSES for pred in predictions)
        alerte_count += nb_attacks

        log(f"⚠️ Attaques détectées dans ce lot : {nb_attacks}", alert=(nb_attacks > 0))

        if alerte_count >= 4:
            log("🚨 ALERTE : Nombre critique d'attaques détectées ! 🚨", alert=True)

        historique = pd.concat([historique, echantillon], ignore_index=True)
        time.sleep(1.5)

    end_time = time.time()
    execution_time = end_time - start_time

    precision = precision_score(true_labels, predicted_labels, pos_label=ATTACK_CLASSES[0])

    total_flux = len(true_labels)
    nb_attacks_total = sum(label in ATTACK_CLASSES for label in true_labels)
    nb_normaux = total_flux - nb_attacks_total
    attack_percentage = (nb_attacks_total / total_flux) * 100 if total_flux > 0 else 0

    st.success("✅ Simulation terminée.")
    st.markdown(f"- Nombre total de flux traités : `{total_flux}`")
    st.markdown(f"- Nombre total de flux d’attaque : `{nb_attacks_total}`")
    st.markdown(f"- Nombre total de flux normaux : `{nb_normaux}`")
    st.markdown(f"- 📊 Pourcentage de flux d’attaque : `{attack_percentage:.2f}%`")
    st.markdown(f"- 🎯 Précision globale : `{precision:.4f}`")
    st.markdown(f"- ⏱ Temps d'exécution total : `{execution_time:.2f}` secondes")

# Bouton pour lancer la simulation
if st.button("🚀 Lancer la Simulation"):
    run_simulation()
