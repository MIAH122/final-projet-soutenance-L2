import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import sqlite3
import joblib
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="RiceInsight Madagascar",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
def load_css():
    css_path = "style.css"
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ============================================================
# AUTHENTIFICATION
# ============================================================
load_dotenv()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USERNAME_ATTENDU = os.getenv("APP_USERNAME")
PASSWORD_ATTENDU = os.getenv("APP_PASSWORD")

UTILISATEURS = {}
if USERNAME_ATTENDU and PASSWORD_ATTENDU:
    UTILISATEURS[USERNAME_ATTENDU] = hash_password(PASSWORD_ATTENDU)

def verifier_login(username, password):
    return username in UTILISATEURS and UTILISATEURS[username] == hash_password(password)

def page_login():
    st.markdown(
        """
        <div class="login-page">
            <div class="login-form">
                <div class="brand-mark">🌾</div>
                <div class="brand-name">RiceInsight</div>
                <div class="brand-subtitle">MADAGASCAR</div>
                <div class="login-line"></div>
                <h2>Bienvenue</h2>
                <p>Connectez-vous à votre plateforme d'analyse et de prévision des prix du riz.</p>
        """,
        unsafe_allow_html=True,
    )

    username = st.text_input("Nom d'utilisateur", placeholder="Votre identifiant")
    password = st.text_input("Mot de passe", type="password", placeholder="Votre mot de passe")

    if st.button("Se connecter  →", use_container_width=True, type="primary"):
        if verifier_login(username, password):
            st.session_state["connecte"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Identifiants incorrects.")

    st.markdown(
        """
            <div class="login-footer">Plateforme intelligente • Data & Machine Learning</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "connecte" not in st.session_state:
    st.session_state["connecte"] = False

if not st.session_state["connecte"]:
    page_login()
    st.stop()

# ============================================================
# DONNÉES + MODÈLE
# ============================================================
@st.cache_data
def charger_donnees():
    conn = sqlite3.connect("data/wfp_complet.db")
    df = pd.read_sql(
        """
        SELECT date, admin1, commodity, price, priceflag, pricetype, unit, currency
        FROM food_prices
        WHERE commodity = ?
        """,
        conn,
        params=("Rice (local)",),
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df["annee"] = df["date"].dt.year
    df["mois"] = df["date"].dt.month
    df["admin1_encoded"] = df["admin1"].astype("category").cat.codes
    df = df.sort_values(["admin1", "date"])
    df["price_lag1"] = df.groupby("admin1")["price"].shift(1)
    df["price_lag3"] = df.groupby("admin1")["price"].shift(3)
    df["price_rolling3"] = df.groupby("admin1")["price"].transform(
        lambda x: x.rolling(3).mean()
    )
    return df.dropna()

@st.cache_resource
def charger_modele():
    model = joblib.load("model/model_rf.pkl")
    scaler = joblib.load("model/scaler.pkl")
    features = joblib.load("model/features.pkl")
    metadata_path = "model/model_metadata.json"
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return model, scaler, features, metadata

df = charger_donnees()
model_rf, scaler, features, model_metadata = charger_modele()

regions = sorted(df["admin1"].dropna().unique().tolist())

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🌾</div>
            <div>
                <div class="sidebar-title">RiceInsight</div>
                <div class="sidebar-country">MADAGASCAR</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-label">OVERVIEW</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyse des prix", "Régions", "Prévision", "Modèle IA", "Données"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="user-card">
            <div class="avatar">👤</div>
            <div>
                <div class="user-name">{st.session_state["username"]}</div>
                <div class="user-role">Utilisateur connecté</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("↪  Se déconnecter", use_container_width=True):
        st.session_state["connecte"] = False
        st.rerun()

# ============================================================
# HELPERS
# ============================================================
def header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="eyebrow">RICEINSIGHT • MADAGASCAR</div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi(label, value, note=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """

# ============================================================
# PAGES DYNAMIC ROUTING
# ============================================================
if page == "Dashboard":
    header(
        "Tableau de bord",
        "Une vue globale de l’évolution du prix du riz à Madagascar, de 2004 à 2026.",
    )
    st.caption("Source WFP • historique disponible : 2004–2026 • les données récentes peuvent être marquées comme agrégées dans la source.")

    prix_moyen = df["price"].mean()
    prix_min = df["price"].min()
    prix_max = df["price"].max()
    dernier = df.sort_values("date").iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("OBSERVATIONS", f"{len(df):,}".replace(",", " "), "Données exploitées"), unsafe_allow_html=True)
    c2.markdown(kpi("RÉGIONS", f"{df['admin1'].nunique()}", "Régions couvertes"), unsafe_allow_html=True)
    c3.markdown(kpi("PRIX MOYEN", f"{prix_moyen:,.0f} Ar".replace(",", " "), "Prix moyen observé"), unsafe_allow_html=True)
    c4.markdown(kpi("DERNIER PRIX", f"{dernier['price']:,.0f} Ar".replace(",", " "), f"{dernier['admin1']}"), unsafe_allow_html=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("### Évolution du prix")
        evolution = df.groupby("date", as_index=True)["price"].mean()
        st.line_chart(evolution, height=350)

    with right:
        st.markdown("### Résumé")
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-icon">📊</div>
                <div>
                    <div class="insight-title">Situation actuelle</div>
                    <div class="insight-text">
                        Le prix moyen observé est de <b>{prix_moyen:,.0f} Ar</b>.
                        Le dernier prix disponible est de <b>{dernier['price']:,.0f} Ar</b>.
                    </div>
                </div>
            </div>
            <div class="mini-stat"><span>Prix minimum</span><b>{prix_min:,.0f} Ar</b></div>
            <div class="mini-stat"><span>Prix maximum</span><b>{prix_max:,.0f} Ar</b></div>
            <div class="mini-stat"><span>Dernière région</span><b>{dernier['admin1']}</b></div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )

    st.markdown("### Prix moyen par région")
    regional = df.groupby("admin1")["price"].mean().sort_values()
    st.bar_chart(regional, height=400)

elif page == "Analyse des prix":
    header(
        "Analyse des prix",
        "Explorez les tendances, les différences régionales et l’évolution du prix du riz.",
    )

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        region = st.selectbox("Région", ["Toutes"] + regions)
    with c2:
        annee_min = int(df["annee"].min())
        annee_max = int(df["annee"].max())
        annees = st.slider("Période", annee_min, annee_max, (annee_min, annee_max))
    with c3:
        affichage = st.selectbox("Granularité", ["Jour", "Mois", "Année"])

    dfa = df[(df["annee"] >= annees[0]) & (df["annee"] <= annees[1])].copy()
    if region != "Toutes":
        dfa = dfa[dfa["admin1"] == region]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("PRIX MOYEN", f"{dfa['price'].mean():,.0f} Ar".replace(",", " "), "sur la période"), unsafe_allow_html=True)
    c2.markdown(kpi("MINIMUM", f"{dfa['price'].min():,.0f} Ar".replace(",", " "), "valeur observée"), unsafe_allow_html=True)
    c3.markdown(kpi("MAXIMUM", f"{dfa['price'].max():,.0f} Ar".replace(",", " "), "valeur observée"), unsafe_allow_html=True)
    c4.markdown(kpi("ÉCART-TYPE", f"{dfa['price'].std():,.0f} Ar".replace(",", " "), "variabilité"), unsafe_allow_html=True)

    if affichage == "Jour":
        serie = dfa.groupby("date")["price"].mean()
    elif affichage == "Mois":
        serie = dfa.groupby(["annee", "mois"])["price"].mean()
        serie.index = [f"{a}-{m:02d}" for a, m in serie.index]
    else:
        serie = dfa.groupby("annee")["price"].mean()

    st.markdown("### Évolution")
    st.line_chart(serie, height=400)

    left, right = st.columns(2)
    with left:
        st.markdown("### Comparaison régionale")
        st.bar_chart(dfa.groupby("admin1")["price"].mean().sort_values(), height=380)

    with right:
        st.markdown("### Nombre d'observations")
        st.bar_chart(dfa.groupby("admin1").size().sort_values(), height=380)

elif page == "Régions":
    header(
        "Analyse régionale",
        "Comparez le niveau des prix du riz entre les régions de Madagascar.",
    )

    region = st.selectbox("Sélectionner une région", regions)
    dfr = df[df["admin1"] == region].sort_values("date")

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("PRIX MOYEN", f"{dfr['price'].mean():,.0f} Ar".replace(",", " "), region), unsafe_allow_html=True)
    c2.markdown(kpi("MINIMUM", f"{dfr['price'].min():,.0f} Ar".replace(",", " "), "historique"), unsafe_allow_html=True)
    c3.markdown(kpi("MAXIMUM", f"{dfr['price'].max():,.0f} Ar".replace(",", " "), "historique"), unsafe_allow_html=True)

    st.markdown(f"### Profil de {region}")
    st.line_chart(dfr.set_index("date")["price"], height=420)

    classement = df.groupby("admin1")["price"].mean().sort_values(ascending=False).reset_index()
    classement["Rang"] = range(1, len(classement) + 1)
    classement = classement[["Rang", "admin1", "price"]]
    classement.columns = ["Rang", "Région", "Prix moyen (Ar)"]

    st.markdown("### Classement des régions")
    st.dataframe(classement, use_container_width=True, hide_index=True)

elif page == "Prévision":
    header(
        "Prévision du prix",
        "Utilisez le modèle Machine Learning pour estimer le prix futur.",
    )

    st.markdown(
        """
        <div class="forecast-intro">
            <div>
                <div class="forecast-title">Moteur de prévision</div>
                <div class="forecast-text">Sélectionnez une région et une période future pour obtenir une estimation.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 0.8])

    with left:
        region_input = st.selectbox("Région", regions)
        c1, c2 = st.columns(2)
        with c1:
            annee_input = st.number_input(
                "Année",
                min_value=int(df["annee"].max()),
                max_value=2035,
                value=int(df["annee"].max()) + 1,
            )
        with c2:
            mois_noms = [
                "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
            ]
            mois_nom = st.selectbox("Mois", mois_noms)
            mois_input = mois_noms.index(mois_nom) + 1

        if st.button(" Lancer la prévision", use_container_width=True, type="primary"):
            historique = df[df["admin1"] == region_input].sort_values("date")

            if len(historique) < 3:
                st.error("Historique insuffisant pour effectuer la prévision.")
            else:
                admin1_encoded = historique["admin1_encoded"].iloc[0]

                input_data = pd.DataFrame([{
                    "annee": annee_input,
                    "mois": mois_input,
                    "admin1_encoded": admin1_encoded,
                    "price_lag1": historique["price"].iloc[-1],
                    "price_lag3": historique["price"].iloc[-3],
                    "price_rolling3": historique["price"].tail(3).mean(),
                }])[features]

                input_scaled = scaler.transform(input_data)
                prediction = float(model_rf.predict(input_scaled)[0])
                dernier_prix = float(historique["price"].iloc[-1])
                variation = ((prediction - dernier_prix) / dernier_prix * 100) if dernier_prix else 0

                st.session_state["prediction"] = {
                    "value": prediction,
                    "last": dernier_prix,
                    "variation": variation,
                    "region": region_input,
                    "annee": annee_input,
                    "mois": mois_nom,
                }

    with right:
        historique = df[df["admin1"] == region_input].sort_values("date")
        dernier_prix = historique["price"].iloc[-1]
        st.markdown(
            kpi(
                "DERNIER PRIX CONNU",
                f"{dernier_prix:,.0f} Ar".replace(",", " "),
                f"{region_input} • {historique['date'].iloc[-1].strftime('%d/%m/%Y')}",
            ),
            unsafe_allow_html=True,
        )

    if "prediction" in st.session_state:
        p = st.session_state["prediction"]

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">PRIX PRÉDIT</div>
                <div class="prediction-value">{p['value']:,.0f} <span>Ar / kg</span></div>
                <div class="prediction-period">{p['region']} • {p['mois']} {p['annee']}</div>
                <div class="prediction-change">
                    {"↑" if p["variation"] >= 0 else "↓"} {abs(p["variation"]):.1f} % par rapport au dernier prix
                </div>
            </div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )

        hist = df[df["admin1"] == p["region"]].sort_values("date").tail(36)
        graph = hist[["date", "price"]].set_index("date")
        pred_row = pd.DataFrame(
            {"price": [p["value"]]},
            index=[pd.Timestamp(year=p["annee"], month=mois_noms.index(p["mois"]) + 1, day=1)],
        )
        graph = pd.concat([graph, pred_row]).sort_index()

        st.markdown("### Historique et estimation")
        st.line_chart(graph, height=400)

elif page == "Modèle IA":
    header(
        "Centre Machine Learning",
        "Visualisez le modèle utilisé et ses performances.",
    )

    r2 = model_metadata.get("r2", 0.0)
    mae = model_metadata.get("mae", 0.0)
    rmse = model_metadata.get("rmse", 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("MODÈLE", "Random Forest", "modèle déployé"), unsafe_allow_html=True)
    c2.markdown(kpi("R²", f"{r2:.4f}", "performance sur test"), unsafe_allow_html=True)
    c3.markdown(kpi("MAE", f"{mae:,.2f} Ar".replace(",", " "), "erreur absolue moyenne"), unsafe_allow_html=True)
    c4.markdown(kpi("RMSE", f"{rmse:,.2f} Ar".replace(",", " "), "racine de l'erreur quadratique"), unsafe_allow_html=True)

    st.markdown("### Variables utilisées par le modèle")
    features_df = pd.DataFrame({"Variable": features})
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("### Architecture de la prédiction")
    st.markdown(
        """
        <div class="pipeline">
            <div class="pipeline-step"><b>01</b><span>Données historiques</span></div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><b>02</b><span>Prétraitement</span></div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><b>03</b><span>Standardisation</span></div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><b>04</b><span>Random Forest</span></div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><b>05</b><span>Prix prédit</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Données":
    header(
        "Data Explorer",
        "Explorez les données utilisées par la plateforme.",
    )
    st.caption("Historique exploité : 2004–2026. La série WFP utilisée combine les observations disponibles et conserve le statut de chaque donnée (actual / aggregate).")

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("LIGNES", f"{len(df):,}".replace(",", " "), "après préparation"), unsafe_allow_html=True)
    c2.markdown(kpi("COLONNES", f"{len(df.columns)}", "variables disponibles"), unsafe_allow_html=True)
    c3.markdown(kpi("VALEURS MANQUANTES", f"{int(df.isna().sum().sum())}", "dans les données chargées"), unsafe_allow_html=True)

    st.markdown("### Filtrer les données")
    region = st.selectbox("Région", ["Toutes"] + regions)

    dfd = df if region == "Toutes" else df[df["admin1"] == region]
    st.dataframe(dfd, use_container_width=True, height=520, hide_index=True)

    csv = dfd.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Télécharger les données CSV",
        csv,
        "riceinsight_donnees.csv",
        "text/csv",
    )
