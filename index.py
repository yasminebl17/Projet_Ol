import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from concurrent.futures import ThreadPoolExecutor

# 🌙 Thème sombre customisé
st.set_page_config(page_title="Détection d'Intrusion", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e1e;
        color: #00ffcc;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔐 Détection parallèle des attaques en temps réel ")
st.markdown("---")

# 📤 Upload ou chargement de fichier
uploaded_file = st.file_uploader("📂 Choisir un fichier CSV (ex: KDDTest-21.csv)", type="csv")

if uploaded_file:
    with st.spinner("Chargement et préparation des données..."):
        data = pd.read_csv(uploaded_file)
        data.dropna(inplace=True)

        # Encodage
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

        st.success("✅ Données prêtes. Tu peux lancer la simulation maintenant !")

        if st.button("🚀 Lancer la Simulation"):
            with st.spinner("Simulation en cours..."):

                ATTACK_CLASSES = [0]
                true_labels = []
                predicted_labels = []

                # Fonction pour obtenir un lot de données aléatoires
                def get_batch():
                    return data.sample(n=700).reset_index(drop=True)

                # Simulation pour chaque esclave
                def simulate_slave(batch_data):
                    X_new = scaler.transform(batch_data.iloc[:, :-1])
                    predictions = knn.predict(X_new)
                    batch_data["Prediction"] = predictions
                    nb_attacks = sum(pred in ATTACK_CLASSES for pred in predictions)
                    return batch_data, nb_attacks

                start_time = time.time()

                # Traitement de 10 paquets
                total_alerts = 0  # Compteur pour les alertes globales
                for i in range(10):
                    st.markdown(f"### 📦 Lot de données {i+1}")

                    batch_data = get_batch()

                    # Traitement du paquet avec les esclaves
                    
                    
                    with ThreadPoolExecutor(max_workers=18) as executor:
                        futures = [executor.submit(simulate_slave, batch_data) for _ in range(3)]
                        for future in futures:
                            batch_result, nb_attacks = future.result()
                            true_labels.extend(batch_result.iloc[:, -2])
                            predicted_labels.extend(batch_result["Prediction"])

                            total_alerts += nb_attacks

                            st.markdown(f"⚠️ Attaques détectées dans ce lot : **{nb_attacks}**")

                            # Si le nombre d'alertes dépasse un seuil, afficher une alerte
                            if total_alerts >= 10:
                                st.error("🚨 ALERTE : Nombre critique d'attaques détectées !")

                    time.sleep(0.5)

                end_time = time.time()

                # Calcul des résultats
                precision = precision_score(true_labels, predicted_labels, pos_label=ATTACK_CLASSES[0])
                total_flux = len(true_labels)
                nb_attacks_total = sum(label in ATTACK_CLASSES for label in true_labels)
                nb_normaux = total_flux - nb_attacks_total
                attack_percentage = (nb_attacks_total / total_flux) * 100 if total_flux > 0 else 0
                execution_time = end_time - start_time

                st.success("✅ Simulation terminée")
                st.markdown("---")
                st.markdown(f"🧮 Nombre total de flux traités : **{total_flux}**")
                st.markdown(f"🛡️ Flux d’attaque détectés : **{nb_attacks_total}**")
                st.markdown(f"✅ Flux normaux : **{nb_normaux}**")
                st.markdown(f"📊 Pourcentage d’attaque : **{attack_percentage:.2f}%**")
                st.markdown(f"🎯 Précision sur les attaques : **{precision:.4f}**")
                st.markdown(f"⏱ Temps total : **{execution_time:.2f} s**")
