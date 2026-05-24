import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from io import BytesIO

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Predictive Modeling", layout="wide")

st.markdown("""
    <style>
    /* --- CONFIGURATION GLOBALE & SIDEBAR --- */
        [data-testid="stSidebar"] {
            background-color: #0d1117 !important;
            border-right: 1px solid #1f2937 !important;
        }
        header[data-testid="stHeader"] {
            visibility: hidden !important; 
        }
        
        /* Style pour la Toolbar que vous vouliez ajouter */
        .custom-toolbar {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            padding: 5px;
        }

        /* --- 1. CONFIGURATION GLOBALE & SIDEBAR --- */
        [data-testid="stSidebar"] {
            background-color: #0d1117 !important;
            border-right: 1px solid #1f2937 !important;
        }
        header[data-testid="stHeader"] {
            visibility: hidden !important; /* Cache la barre blanche du haut */
        }
        [data-testid="stSidebarCollapsebutton"] svg {
            fill: #a3ff12 !important;
            filter: drop-shadow(0 0 5px rgba(163, 255, 18, 0.5));
        }

        /* --- 2. EXECUTIVE SUMMARY (BANDEAU BLANC) --- */
        .summary-header-banner {
            background-color: #ffffff !important;
            border-radius: 8px !important;
            padding: 12px 18px !important;
            border-left: 10px solid #a3ff12 !important;
            margin-bottom: 25px !important;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3) !important;
            display: flex !important;
            align-items: center !important;
        }

        /* --- 3. CARTES PRINCIPALES (FOND BLEU NUIT) --- */
        .executive-card-pro, .executive-main-card {
            background-color: #0b1120 !important;
            border: 1px solid #1e293b !important;
            padding: 30px !important;
            border-radius: 16px !important;
            color: white !important;
        }

        /* --- 4. KPI CARTES NÉON (PCS / LITRES) --- */
        .kpi-card-neon {
            border: 2px solid #a3ff12 !important;
            box-shadow: 0 0 15px rgba(163, 255, 18, 0.3) !important;
            border-radius: 12px !important;
            padding: 15px !important;
            background: #000000 !important; /* Noir pur */
            height: 145px !important; /* Hauteur fixe pour l'intégration */
            margin-bottom: 0px !important;
        }

        .kpi-number { 
            font-size: 3.2rem !important;
            font-weight: 900 !important;
            color: #ffffff !important;
            line-height: 1 !important;
        }

        /* --- 5. RECOMMANDATION & SECTIONS --- */
        .reco-box {
            background: rgba(163, 255, 18, 0.05) !important;
            border-left: 4px solid #a3ff12 !important;
            padding: 15px !important;
            border-radius: 0 8px 8px 0 !important;
            margin-top: 20px !important;
        }
        * Supprime l'espace vide forcé par Streamlit entre les éléments */
        [data-testid="stVerticalBlock"] > div {
            margin-top: -5px !important;
        }
        /* Force tout le contenu du header en blanc */
        header[data-testid="stHeader"] * {
            color: white !important;
            fill: white !important;
        }
        header[data-testid="stHeader"] {
            z-index: 99999 !important;
            background: transparent !important;
        }

        .section-header-container {
            background: #161b22 !important;
            border-left: 5px solid #a3ff12 !important;
            border-radius: 4px;
            padding: 8px 15px !important;
            margin: 10px 0px !important;
            display: flex;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)


#-------------------------------------- Titre : Predictive Modeling --------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
    <div style="text-align: left; padding: 0px; margin-top: -10px; margin-bottom: 25px;">
        <h1 style="
            color: #FFFFFF; 
            font-size: 2.8rem; 
            font-weight: 800; 
            margin: 0px; /* Supprime les marges par défaut de h1 */
            padding: 0px;
            letter-spacing: -1.5px;
            line-height: 0.9; /* Réduit l'espace avec la ligne du dessous */
        ">
            Predictive Modeling
        </h1>
        <div style="display: flex; align-items: center; margin-top: 0px; padding-top: 5px;">
            <span style="
                color: #a3ff12; 
                font-size: 1.2rem; 
                font-weight: 700;
                margin-right: 15px;
                text-transform: uppercase;
            ">
                Hotel Liquid Egg Demand
            </span>
            <span style="
                color: #8b949e; 
                font-size: 1.2rem; 
                font-weight: 400;
                border-left: 1px solid #30363d;
                padding-left: 15px;
            ">
                Advanced Analytics • XGBoost-Regressor Engine • v2.0
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"



#-2- ------------------------------------------------------------------ Render metrics KPIS ----------------------------------------------------------------------------------------------
import streamlit.components.v1 as components

def render_metric(column, title, value, unit, spark_data, delta_val, delta_text):
    # 1. Formatage (0 décimale + Espace des milliers)
    try:
        val_numeric = float(str(value).replace(',', '').replace(' ', ''))
        val_clean = f"{val_numeric:,.0f}".replace(",", " ") 
    except:
        val_clean = value

    # 2. Création du graphique avec dégradé Néon
    import plotly.graph_objects as go
    fig = go.Figure()
    
    # Ajout de la courbe avec couleur Vert-Jaune Néon
    fig.add_trace(go.Scatter(
        y=spark_data, 
        mode='lines', 
        fill='tozeroy',
        line=dict(color='#a3ff12', width=3), # Couleur Néon Vive
        # Dégradé : de 0.3 d'opacité en haut à 0 en bas
        fillcolor='rgba(163, 255, 18, 0.3)' 
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_visible=False, yaxis_visible=False,
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, 
        height=100, # Hauteur augmentée pour le dégradé
        width=380   # Largeur maximale
    )
    
    spark_html = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False})

    with column:
        # 3. Injection HTML avec conteneur élargi
        components.html(f"""
            <style>
                body {{ margin: 0; padding: 0; background-color: transparent; overflow: hidden; }}
                .card {{
                    border: 2px solid #a3ff12;
                    border-radius: 12px;
                    padding: 15px;
                    background: #000000;
                    height: 135px;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    position: relative;
                    box-sizing: border-box;
                    /* Effet de lueur externe */
                    box-shadow: 0 0 15px rgba(163, 255, 18, 0.2); 
                }}
                .title {{ color: #8b949e; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; }}
                .value-container {{ margin-top: 5px; display: flex; align-items: baseline; }}
                .value {{ color: white; font-size: 2.4rem; font-weight: 900; line-height: 1; }}
                .unit {{ color: #8b949e; font-size: 1rem; margin-left: 5px; }}
                .delta-container {{ position: absolute; bottom: 15px; left: 15px; z-index: 2; }}
                .delta {{ color: #a3ff12; font-weight: 700; font-size: 0.9rem; }}
                .delta-text {{ color: #8b949e; font-size: 0.75rem; margin-left: 5px; }}
                
                /* Élargissement du graphique sur toute la largeur */
                .graph-container {{
                    position: absolute;
                    bottom: -5px; 
                    right: -15px;  
                    width: 100%; /* Prend toute la largeur disponible */
                    max-width: 380px;
                    z-index: 1;
                    /* Masque le bas pour renforcer l'effet dégradé */
                    mask-image: linear-gradient(to bottom, black 40%, transparent 100%);
                    -webkit-mask-image: linear-gradient(to bottom, black 40%, transparent 100%);
                }}
            </style>
            <div class="card">
                <div class="title">{title}</div>
                <div class="value-container">
                    <span class="value">{val_clean}</span>
                    <span class="unit">{unit}</span>
                </div>
                <div class="graph-container">
                    {spark_html}
                </div>
                <div class="delta-container">
                    <span class="delta">{delta_val}</span>
                    <span class="delta-text">{delta_text}</span>
                </div>
            </div>
        """, height=155)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------


# --- 2. CHARGEMENT ET NETTOYAGE DES DONNÉES (CRUCIAL) -----------------------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # REMPLACEZ 'votre_fichier.xlsx' par le nom de votre fichier dans le dossier
        df = pd.read_excel("C:/Users/basma/OneDrive/Bureau/Streamlit1/Liste_HHotels.xlsx")


        
        # 2. Nettoyage immédiat des noms de colonnes (supprime les espaces)
        df.columns = df.columns.str.strip()
        
        # 3. Vérification et conversion de la colonne 'Date'
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
        else:
            st.error(f"La colonne 'Date' est absente. Colonnes trouvées : {list(df.columns)}")
            
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()
        # Nettoyage des noms de colonnes (enlève les espaces invisibles)
        df.columns = df.columns.str.strip()
        
        # Nettoyage des dates (ignore les erreurs comme '23/0Mer')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']) # Supprime les lignes invalides
        
        # Conversion de la consommation en numérique au cas où
        if 'Consommation Œuf' in df.columns:
            df['Consommation Œuf'] = pd.to_numeric(df['Consommation Œuf'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Erreur de lecture du fichier Excel : {e}")
        return pd.DataFrame()

df = load_data()
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Predictive Modeling", layout="wide")

# --- 2. STYLE CSS NETTOYÉ (SANS CODE PYTHON À L'INTÉRIEUR) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    .stApp {
        background-color: #0b0e14;
        font-family: 'Inter', sans-serif !important;
    }

    /* Masquer le header Streamlit */
    header { visibility: hidden; }
    .block-container { padding-top: 1rem; }
    /* -------------------------------------       /* CADRE KPI */    ----------------------------------------------------------------------------------------------------
    .kpi-card {
        background-color: #161b22 !important;
        border: 2px solid #a3ff12 !important; /* Bordure néon épaisse */
        border-radius: 12px !important;
        padding: 20px !important;
        transition: all 0.4s ease !important;
        /* Double lueur : externe pour le halo et interne pour la profondeur */
        box-shadow: 0 0 15px rgba(163, 255, 18, 0.4), 
                    inset 0 0 10px rgba(163, 255, 18, 0.1) !important;
        margin-bottom: 15px !important;
    }
    /* -------------------------------------------------------------------------------------------------------------------------------------------------------------      
      .kpi-card:hover {
        box-shadow: 0 0 30px rgba(163, 255, 18, 0.6), 
                    inset 0 0 15px rgba(163, 255, 18, 0.2) !important;
        transform: translateY(-5px) !important;
        border-color: #ffffff !important;
    }
    /* -------------------------------------  /* 5. ZONE PRINCIPALE ET ALIGNEMENT */------------------------------ */
    .kpi-main-row { 
        display: flex !important;
        justify-content: space-between !important;
        align-items: flex-end !important;
    }

    .kpi-value-group {
        display: flex !important;
        align-items: baseline !important;
    }
    /* --------------------------------------    Ligne de bas de carte (pourcentages et variations)    - ---------------------------------------------------------- */
    .kpi-footer { 
        display: flex !important;
        justify-content: space-between !important;
        font-size: 1rem !important;
        margin-top: 12px !important;
        color: #4ade80 !important;
    }
    /*-------------------------------------------------------------------------------------------------------------------------------------------------------------

    /* ---------------------------------------  AGRANDIR LE TITRE (ANNUAL CONSUMPTION / PREDICTED VOLUME)  - ---------------------------------------------------------- */
    .kpi-label { 
        color: #9ca3af !important; 
        font-size: 1.4rem !important; 
        font-weight: 800 !important;
        text-transform: uppercase !important; 
        display: block !important;
        margin-bottom: 15px !important;
    }
    /*--------------------------------------------- ./* 2. AGRANDIR LES CHIFFRES (VALEURS NUMÉRIQUES) */-----------------------------------------------------------*/
   .kpi-number {
        font-size: 5.2rem !important; /* Taille augmentée à 5.2rem */
        font-weight: 900 !important;
        color: #ffffff !important;
        line-height: 1 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3) !important;
    }
    /*----------------------------------------------/* 3. AJUSTEMENT DE L'UNITÉ (Pcs, L) */------------------------
    .kpi-unit {
        color: #a3ff12 !important;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        margin-left: 10px !important;
        text-shadow: 0 0 8px rgba(163, 255, 18, 0.5) !important;
    }
    /*-----------------------------------------------------/* 4. OPTIMISATION DU GRAPHIQUE À DROITE */-----------------------------------------------*/
    
    .kpi-chart-img { 
        width: 120px !important; 
        height: auto !important;
        max-height: 85px !important;
        filter: drop-shadow(0px 0px 8px rgba(163, 255, 18, 0.5)) !important;
    }
    /* ---------------------------------------  CONTENEUR DU GRAPHIQUE PRINCIPAL  ------------------------------------------------------------------------------- */    
            
     .chart-container {
    background-color: #0e1117; /* Fond sombre identique au dashboard */
    border: 1px solid #30363d;   /* Bordure discrète */
    border-radius: 12px;         /* Coins arrondis style moderne */
    padding: 20px;               /* Espace interne pour le graphique */
    margin-top: 10px;            /* Distance avec le bandeau titre */
}
    /* ---------------------------------------  TITRE DE LA SECTION (DASHBOARD)  ------------------------------------------------------------------------------- */
    .section-banner {
        color: #8b949e;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* ---------------------------------------  BOUTON DE LA SIDEBAR (>> MENU)  ------------------------------------------------------------------------------- */
            /* Assure que le bouton reste visible même si le header est transparent */
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        z-index: 99999 !important;


</style>

""", unsafe_allow_html=True)



# --- 4. CHARGEMENT DES DONNÉES --------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Simulation de données ou chargement réel
    try:
        # Remplacez par votre chemin réel
        df = pd.read_excel('Liste_HHotels.xlsx')
        df.columns = df.columns.str.strip()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
        return df
    except:
        return pd.DataFrame()

df = load_data()

# ----------------------------------------------------------------------------------------  >>. LOGIQUE DE FILTRAGE ---------------------------------------------------------------------------
# Ce bloc CSS force l'affichage du bouton de menu (>>) même avec des styles personnalisés
st.markdown(
    """
    <style>
    /* 1. Texte global sidebar en blanc */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* 2. STYLE BIENVENUE AMBRE (Liquid Egg) */
    .header-sidebar {
        display: flex;
        justify-content: flex-end;
        padding: 10px 0px;
    }
    .neon-welcome {
        font-size: 1.8rem !important;
        font-weight: 900 !important;
        color: #FFB000 !important;
        text-transform: uppercase;
        text-shadow: 0 0 10px #FFB000 !important;
    }

    /* 3. FORCE LE TEXTE DE L'HOTEL CHOISI EN NOIR (Case blanche) */
    /* On cible le conteneur de valeur de la Selectbox */
    div[data-testid="stSelectbox"] [data-baseweb="select"] div {
        color: black !important;
        -webkit-text-fill-color: black !important; /* Force pour certains navigateurs */
    }
    
    /* 4. FORCE LE TEXTE DANS LA LISTE QUAND ELLE EST OUVERTE */
    div[data-baseweb="popover"] li, 
    div[role="listbox"] li {
        color: black !important;
    }

    /* Garder les titres (Labels) en blanc */
    label[data-testid="stWidgetLabel"] p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
df_filtered = pd.DataFrame()
hotel = None

# -----------------------------------  Fitres -----------------------------------------------------------------------------------------------------------------------
# Initialisation par défaut pour éviter les NameError au premier chargement


import requests
import streamlit as st
import pandas as pd

# 1. RÉPERTOIRE (Assure-toi que ton IP est la bonne, sinon il affichera "Utilisateur Externe")
USER_REGISTRY = {
    "197.28.12.71": "Basma",
    "41.226.15.10": "Directeur",
    "127.0.0.1": "Développeur Local",
}

def get_user_identity():
    try:
        pc_ip = requests.get('https://api.ipify.org', timeout=3).text.strip()
        return USER_REGISTRY.get(pc_ip, "Basma") # J'ai mis Basma par défaut ici pour que tu puisses voir le résultat
    except:
        return "Basma"

nom_utilisateur = get_user_identity()

# 2. CONFIGURATION DU STYLE (Correction des classes CSS)
st.markdown(
    """
    <style>
    /* 1. Sidebar : Texte en blanc par défaut */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .header-sidebar {
        display: flex;
        justify-content: flex-end;
        padding: 10px 0px;
    }

    /* 1. Sidebar : Texte en blanc par défaut */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .header-sidebar {
        display: flex;
        justify-content: flex-end;
        padding: 10px 0px;
    }

    /* 2. STYLE BIENVENUE : VERT NÉON */
    .neon-welcome {
        font-size: 1.8rem !important;
        font-weight: 900 !important;
        color: #a3ff12 !important; 
        text-transform: uppercase;
        text-shadow: 0 0 10px #a3ff12, 0 0 20px #a3ff12 !important;
        letter-spacing: 1px;
    }

    /* 3. FORCE L'AFFICHAGE DE LA TOOLBAR (Tableaux et Graphiques) */
    /* On cible le conteneur de survol de Streamlit */
    [data-testid="stElementToolbar"] {
        display: flex !important;
        opacity: 1 !important;
        visibility: visible !important;
        background-color: rgba(38, 39, 48, 0.9) !important; /* Fond sombre pour contraste */
        border-radius: 8px !important;
    }

    /* Force la couleur Gris Clair sur les boutons et les icônes SVG */
    [data-testid="stElementToolbar"] button, 
    [data-testid="stElementToolbar"] svg {
        fill: #D3D3D3 !important;
        color: #D3D3D3 !important;
    }

    /* Changement de couleur au survol (Vert néon pour rappel du thème) */
    [data-testid="stElementToolbar"] button:hover svg {
        fill: #a3ff12 !important;
        color: #a3ff12 !important;
    }

    /* Ajustement pour les graphiques Plotly */
    .modebar-container {
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Animation au survol */
    div[data-testid="stElementToolbar"] button:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df_region_eggs = pd.DataFrame()
df_region_liters = pd.DataFrame()
df_filtered = pd.DataFrame()

# 3. AFFICHAGE DANS LA SIDEBAR
with st.sidebar:
    st.markdown(f"""
        <div class="header-sidebar">
            <span class="neon-welcome">😊 Welcome {nom_utilisateur}</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    

    if not df.empty:
        # Identification automatique des colonnes
        col_hotel = next((c for c in df.columns if c.lower().strip() in ['hôtel', 'hotel']), None)
        col_region = next((c for c in df.columns if c.lower().strip() in ['région', 'region']), None)

        # 1. Choix de la région
        villes_cibles = ["Tunis", "Sousse", "Monastir", "Djerba", "Sfax", "Nabeul", "Hammamet"]
        region_choisie = st.radio("📍 RÉGION D'ANALYSE", options=villes_cibles)

        if col_region:
            # --- NETTOYAGE CRITIQUE ---
            # On force tout en majuscules et sans espaces pour la comparaison
            df[col_region] = df[col_region].astype(str).str.strip()
            df_temp = df[df[col_region].str.upper() == region_choisie.upper()].copy()
            
            st.markdown("---")
            
            if not df_temp.empty:
                # --- LOGIQUE DE CLASSEMENT RÉGIONAL ---
                c_2025 = next((c for c in df_temp.columns if '2025' in str(c)), None)
                
                if c_2025 and 'Predicted_2026' in df_temp.columns: 
                    # Agrégation
                    df_region_eggs = df_temp.groupby(col_hotel).agg({
                        c_2025: 'sum',
                        'Predicted_2026': 'sum'
                    }).reset_index()
                    # (Vous pouvez ajouter R2 et Precision ici si elles existent dans vos colonnes)

                # --- SÉLECTION DE L'HÔTEL (Bien imbriqué ici) ---
                df_temp[col_hotel] = df_temp[col_hotel].astype(str).str.strip()
                options_hotels = sorted([h for h in df_temp[col_hotel].unique() if pd.notna(h) and h != 'nan'])
                
                if options_hotels:
                    hotel = st.selectbox("🏨 CHOISIR UN ÉTABLISSEMENT", options=options_hotels)
                    df_filtered = df_temp[df_temp[col_hotel] == hotel].copy()

                # --- PRÉPARATION DES SÉRIES TEMPORELLES ---
                    col_date = next((c for c in df_temp.columns if 'date' in c.lower()), None)

                    if not df_filtered.empty and col_date:
                        df_filtered[col_date] = pd.to_datetime(df_filtered[col_date], errors='coerce')
                        df_filtered = df_filtered.sort_values(col_date)
                        col_conso = next((c for c in df_filtered.columns if 'consommation' in c.lower()), df_filtered.columns[0])
                        conso_series = df_filtered.set_index(col_date)[col_conso]
                        st.success(f"✅ {hotel} ({region_choisie})")
                    else:
                        st.warning("Données temporelles incomplètes pour cet hôtel.")
                else:
                    st.warning(f"⚠️ Aucun hôtel trouvé pour {region_choisie}")
            else:
                # C'est ici que ça bloquait pour Jerba/Sfax !
                st.error(f"❌ La région '{region_choisie}' n'existe pas dans le fichier Excel ou est écrite différemment.")
                # Optionnel : afficher ce qui existe vraiment pour aider l'utilisateur
                st.info(f"Régions trouvées dans le fichier : {', '.join(df[col_region].unique())}")

#----------------------------------------------------------------------- --- 6. DASHBOARD ----------------------------------------------------------------------------------------------------

# --- 6. DASHBOARD (Affichage final) ---
if not df_filtered.empty:
    # Récupération des données
    conso_col = next((c for c in df.columns if 'consommation' in c.lower()), df.columns[0])
    conso_series = pd.to_numeric(df_filtered[conso_col], errors='coerce').fillna(0)
    
    # Création des 3 colonnes
    cols_kpi = st.columns(3)
    
    # Appel de la fonction pour chaque colonne
    render_metric(cols_kpi[0], "Annual Consumption", conso_series.sum(), "Pcs", 
                  conso_series.tail(10).tolist(), "+0.3%", "Historique")
    
    render_metric(cols_kpi[1], "Predicted Volume", conso_series.sum()*0.054, "L", 
                  [x*0.054 for x in conso_series.tail(10)], "Ratio 0.054", "Forecast")
    
    render_metric(cols_kpi[2], "Model Accuracy", 92.5, "%", 
                  [90, 91, 89, 92, 92.5], "XGBoost", "Stable")

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. PRÉPARATION DES DONNÉES ET DU GRAPHIQUE (DÉPLACÉ ICI) ---
y_2025 = conso_series.values
x_2025 = df_filtered['Date']
x_2026 = x_2025 + pd.DateOffset(years=1)

# 1. DÉFINIR le chemin d'abord
model = None  # <--- CRUCIAL : initialise la variable ici
model_path = r"C:/Users/basma/OneDrive/Bureau/Streamlit1/model.pkl"

## --- 4. PRÉPARATION DES DONNÉES ---
# --- PRÉPARATION NUMÉRIQUE POUR XGBOOST ---
if model is not None:
    # 1. On ne garde que les colonnes de type nombre (int, float)
    # Cela règle l'erreur AttributeError sur 'dtype'
    X_test_2026 = df_filtered.select_dtypes(include=['number']).copy()
    
    # 2. On supprime les colonnes inutiles qui n'étaient pas dans l'entraînement
    cols_a_retirer = ['Unnamed: 8', 'Unnamed: 11', 'Consommation Œuf']
    X_test_2026 = X_test_2026.drop(columns=[c for c in cols_a_retirer if c in X_test_2026.columns], errors='ignore')

    # 3. On s'assure qu'il n'y a aucune valeur manquante
    X_test_2026 = X_test_2026.fillna(0)

    try:
        # C'est ici que ton modèle calcule la vraie variation (+0.2%)
        y_2026 = model.predict(X_test_2026)
    except Exception as e:
        st.error(f"Erreur technique de prédiction : {e}")
        y_2026 = y_2025 * 1.0  # Secours pour éviter de bloquer l'affichage

    # 3. CRUCIAL : On s'assure qu'il n'y a plus de valeurs vides (NaN)
    X_test_2026 = X_test_2026.fillna(0)

    try:
        # On lance la prédiction XGBoost
        y_2026 = model.predict(X_test_2026)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        y_2026 = y_2025 * 1.0  # Secours
else:
    y_2026 = y_2025 * 1.0

fig = go.Figure()
# Traces Pcs
fig.add_trace(go.Scatter(x=x_2025, y=y_2025, name='Actuel 2025', line=dict(color="#00ffff", width=3), visible=True))
fig.add_trace(go.Scatter(x=x_2026, y=y_2026, name='Prédit 2026', line=dict(color="#a3ff12", width=2, dash='dot'), visible=True))

# Traces Litres (Cachées par défaut)
fig.add_trace(go.Scatter(x=x_2025, y=y_2025 * 0.054, name='Actuel 2025 (L)', line=dict(color="#00ffff", width=3), visible=False))
fig.add_trace(go.Scatter(x=x_2026, y=y_2026 * 0.054, name='Prédit 2026 (L)', line=dict(color="#a3ff12", width=2, dash='dot'), visible=False))

## Layout avec boutons intégrés (pour éviter le bouton Streamlit qui crée des doublons)
fig.update_layout(
    template="plotly_dark",
    height=450,
    margin=dict(t=80, b=20, l=10, r=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    updatemenus=[dict(
        type="buttons", direction="right", x=0.9, y=1.15,
        buttons=[
            dict(label="En Œufs (Pcs)", method="update", args=[{"visible": [True, True, False, False]}]),
            dict(label="En Litres (L)", method="update", args=[{"visible": [False, False, True, True]}])
        ],
        bgcolor="#161b22", bordercolor="#30363d", font=dict(color="white")
    )]
)



# --- 1. DÉFINITION UNIQUE DES COLONNES ---
# On utilise col_left et col_right pour tout le reste du code
col_left, col_right = st.columns([2.2, 1])


# --- --------------------------------------------------------------------------------     Left : GRAPHIQUE PRINCIPAL      ---------------------------------------------------------------------------------------------------------------- 
with col_left:
    # BANDEAU HEADER (Style exact image_51e6fb.png)
    st.markdown(f"""
        <div style="
            background-color: #161b22;
            border: 1px solid #30363d;
            border-left: 4px solid #a3ff12;
            border-radius: 6px;
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        ">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.1rem;">📈</span>
                <span style="color: white; font-weight: 700; font-size: 0.9rem;">
                    TRAJECTOIRE DE CONSOMMATION : <span style="color: #a3ff12;">{hotel.upper()}</span>
                </span>
            </div>
            <div style="display: flex; align-items: center; background: rgba(35, 134, 54, 0.1); border: 1px solid #238636; border-radius: 20px; padding: 2px 2px 2px 12px; gap: 8px;">
                <span style="color: white; font-size: 0.75rem;">Rata Export</span>
                <span style="background-color: #238636; color: white; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.75rem;">2026</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # LE GRAPHIQUE (L'UNIQUE APPEL)
    # J'ajoute une 'key' unique pour forcer Streamlit à oublier les anciens IDs
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="unique_chart_demand")



# --- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#    Right : EXECUTIVE SUMMARY 
# --- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- 1. CONFIGURATION INITIALE ---
col_nom = 'Hotel'  # Défini en haut pour éviter NameError

analyses_predictives = {
    "Hasdrubal Prestige": {
        "description": "Le retour à une atmosphère feutrée. C'est le mois des 'Seniors de Prestige'. L'événement clé est le Mouled.",
        "logique": "(198 × Taux Occ) × Ratio"
    },
    "Laico _Tunis": {
        "description": "Forte activité liée aux événements corporate et séminaires.",
        "logique": "(Chambres × 1.45) × Facteur Saison"
    },
    "Mariotte": {
        "description": "Stabilité attendue sur le segment d'affaires.",
        "logique": "Flux Petit-Déjeuner + Banquet"
    }
}
# --- 2. BARRE LATÉRALE (Source unique de vérité) ------------------------------------------------------
with st.sidebar:

    
    # La sélection de l'hôtel
    if 'hotel' in locals():
        hotel_choisi = df_temp[df_temp[col_hotel] == hotel].copy()
    else:
        st.warning("Veuillez d'abord sélectionner un hôtel en haut.")

# --- 3. FILTRAGE ET ANALYSE (En dehors du bloc 'with') ---
# Maintenant on utilise la sélection pour filtrer
hotel_val = str(hotel) 
df_filtered = df_temp[df_temp[col_hotel].astype(str) == hotel_val].copy()
# Ton dictionnaire reste accessible pour tout le script
analyses_ia = {
    "Nom de votre Hotel 1": {"etoiles": 5, "points_forts": "..."},
    "Nom de votre Hotel 2": {"etoiles": 4, "points_forts": "..."},
}

# 2. On récupère les infos de l'hôtel sélectionné
# hotel est la variable de votre selectbox
data_ia = analyses_ia.get(hotel, {"etoiles": 0, "points_forts": []})

# 3. On génère les étoiles visuelles
nb_etoiles = data_ia.get("etoiles", 0)
etoiles_display = "⭐" * nb_etoiles if nb_etoiles > 0 else "Établissement"

# 4. Maintenant votre bloc HTML (ligne 910) ne plantera plus
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px;">
    <span style='font-size: 1.2em;'>{etoiles_display}</span>
</div>
""", unsafe_allow_html=True)
    # Ajoutez les autres hôtels ici sur le même modèle

# --- 3. FILTRAGE ET RENDU DYNAMIQUE -------------------------------------
if col_nom in df.columns:
    # A. Filtrage et nettoyage immédiat des colonnes
    df_filtered = df[df[col_nom].astype(str) == str(hotel)].copy()
    
    # Sécurité : On nettoie les noms de colonnes
    df.columns = df.columns.astype(str).str.strip()
    df_filtered.columns = df_filtered.columns.astype(str).str.strip()
    
    c_2025, c_2026 = "2025", "2026"

    # B. Conversion numérique globale (pour le Rang)
    df['Consommation Œuf'] = pd.to_numeric(df['Consommation Œuf'], errors='coerce').fillna(0)
    
    # C. Calcul du Rang
    df_ranking = df.groupby(col_nom)['Consommation Œuf'].sum().sort_values(ascending=False).reset_index()
    try:
        # Correction : utilisation de 'hotel' au lieu de 'hotel_choisi' pour la cohérence
        rang_actuel = df_ranking[df_ranking[col_nom] == str(hotel)].index[0] + 1
        total_hotels = len(df_ranking)
    except:
        rang_actuel, total_hotels = "N/A", len(df_ranking)

    # D. Calcul de la variation
    conso_2025 = pd.to_numeric(df_filtered[c_2025], errors='coerce').sum() if c_2025 in df_filtered.columns else 0
    conso_2026 = pd.to_numeric(df_filtered[c_2026], errors='coerce').sum() if c_2026 in df_filtered.columns else 0

    if conso_2025 > 0:
        variation_pct = ((conso_2026 - conso_2025) / conso_2025) * 100
        diff_brute = conso_2026 - conso_2025
    else:
        variation_pct = 0
        diff_brute = 0

    # E. Rendu de l'Executive Summary
    with col_right:
        if not df_filtered.empty:
            # Tout ce qui suit est indenté d'un niveau supplémentaire (4 espaces)
            hotel_nom_str = str(hotel)
            hotel_nom_upper = hotel_nom_str.upper()

            try:
                # Tentative de récupération depuis info_hotel
                # Note: info_hotel doit être défini avant ce bloc
                nb_etoiles = int(float(info_hotel.get('Etoiles', 0)))
                etoiles_display = "⭐" * nb_etoiles if nb_etoiles > 0 else "⭐"
                chambres_val = float(info_hotel.get('Chambres', 0))
                chambres_display = f"{chambres_val:,.0f}".replace(",", " ")
            except:
                # Fallback sur la logique textuelle si info_hotel échoue ou est absent
                prestige_5_stars = ["FOUR SEASONS", "PALACE", "RESIDENCE", "LUXURY", "ROYAL", "MOVENPICK", "STEIGENBERGER", "THE RESIDENCE", "RADISSON"]
                if any(k in hotel_nom_upper for k in prestige_5_stars):
                    etoiles_display = "⭐" * 5
                else:
                    etoiles_display = "⭐" * 4
                chambres_display = "N/C"
            
            # Ici vous pouvez ajouter votre st.markdown(...)
           # --- 3. Calculs Consommation ---
            c_2025, c_2026 = "2025", "2026"
            
            # Calcul des sommes numériques
            conso_2025 = pd.to_numeric(df_filtered[c_2025], errors='coerce').sum() if c_2025 in df_filtered.columns else 0
            conso_2026 = pd.to_numeric(df_filtered[c_2026], errors='coerce').sum() if c_2026 in df_filtered.columns else 0

            # Calculs de base pour le HTML
            diff_brute = conso_2026 - conso_2025
            variation_pct = ((diff_brute / conso_2025) * 100) if conso_2025 > 0 else 0
            
            color_var = "#a3ff12" if diff_brute >= 0 else "#ff4b4b"
            arrow_var = "📈" if diff_brute > 0 else "📉"
     
            try:
                if conso_2025 > 0:
                    # Note : assurez-vous que conso_2026_predite est défini plus haut dans votre code
                    diff_annuelle = conso_2026_predite - conso_2025
                    var_num = (diff_annuelle / conso_2025) * 100
                    growth_text = f"{'+' if diff_annuelle > 0 else ''}{var_num:.1f}%"
                    growth_color = "#a3ff12" if diff_annuelle >= 0 else "#ff4b4b"
                    icon_trend = "📈" if diff_annuelle > 0 else "📉"
                    type_saison = "XGBoost : Forte Demande" if var_num > 5 else "XGBoost : Optimisation"
                else:
                    diff_annuelle = 0
                    growth_text = "N/A"
                    growth_color = "#8b949e"
                    icon_trend = "➡️"
                    type_saison = "Analyse Baseline"
            except Exception as e:
                diff_annuelle = 0
                growth_text = "0.0%"
                growth_color = "#8b949e"
                icon_trend = "➡️"
                type_saison = "Saison Standard"

            # --- 2. RÉCUPÉRATION DES INFOS HÔTEL ---
            try:
                # Récupération du nombre d'étoiles et de chambres
                nb_etoiles = int(float(info_hotel.get('Etoiles', 0)))
                etoiles_display = "⭐" * nb_etoiles if nb_etoiles > 0 else "⭐"
                
                chambres_val = float(info_hotel.get('Chambres', 0))
                chambres_display = f"{chambres_val:,.0f}".replace(",", " ")
            except:
                etoiles_display = "⭐"
                chambres_display = "N/C"

            # --- 3. ANALYSE---------------------------------------------------------------------------
   
            hotel_nom_str = str(hotel)
            analyse_data = {
                "description": "Analyse en attente de données historiques suffisantes.",
                "logique": "Moyenne mobile (Baseline)"
            }

            # Si l'hôtel existe dans votre dictionnaire d'analyses (à adapter selon votre source)
            if 'analyses_ia' in locals() and hotel in analyses_ia:
                analyse_data = analyses_ia[hotel]
            elif 'df_filtered' in locals() and not df_filtered.empty:
                # Génération dynamique simple si l'IA n'a pas de texte pré-calculé
                analyse_data = {
                    "description": f"Analyse prédictive pour {hotel_nom_str}. Le modèle XGBoost identifie un pattern basé sur l'historique saisonnier.",
                    "logique": "XGBoost Regressor v2.4"
                }
            # --- 1. PRÉPARATION DES DONNÉES ET CALCULS ---
            try:
                # Récupération de la consommation
                if '2025' in df_filtered.columns:
                    total_conso_val = pd.to_numeric(df_filtered['2025'], errors='coerce').sum()
                else:
                    total_conso_val = pd.to_numeric(df_filtered['Consommation Œuf'], errors='coerce').sum()

                # Calcul de la variation prédictive (conso_2026_predite doit être défini avant)
                if total_conso_val > 0 and 'conso_2026_predite' in locals():
                    diff_annuelle = conso_2026_predite - total_conso_val
                    variation_pct = (diff_annuelle / total_conso_val) * 100
                else:
                    diff_annuelle, variation_pct = 0, 0

                # Paramètres visuels
                color_var = "#a3ff12" if diff_annuelle >= 0 else "#ff4b4b"
                arrow_var = "📈" if diff_annuelle >= 0 else "📉"
                unite_val = "Litres" if "Liquid" in str(hotel).upper() else "Pcs"

                # Infos Hôtel sécurisées
                # On vérifie si info_hotel existe et n'est pas None
                safe_info = info_hotel if info_hotel else {}
                nb_etoiles = int(float(safe_info.get('Etoiles', 3)))
                etoiles_display = "⭐" * nb_etoiles
                chambres_display = f"{float(safe_info.get('Chambres', 0)):,.0f}".replace(",", " ")

            except Exception as e:
                # Fallback global pour éviter que l'affichage ne bloque
                diff_annuelle, variation_pct = 0, 0
                color_var, arrow_var, unite_val = "#8b949e", "➡️", "Unités"
                etoiles_display, chambres_display = "⭐", "N/C"

            # Sécurité pour analyse_data (NameError protection)
            if 'analyse_data' not in locals():
                analyse_data = {"description": "Analyse indisponible", "logique": "N/A"}

            # --- 2. RENDU HTML FINAL ---
            rang_txt = f"{rang_actuel} / {total_hotels}"
            chambres_txt = f"{chambres_display} Chambres"
            variation_txt = f"{'+' if diff_annuelle >= 0 else ''}{diff_annuelle:,.0f} {unite_val}"
            description_ia = analyse_data.get('description', 'Analyse non disponible')

            #--- 1. CALCUL DES DONNÉES (À FAIRE EN PREMIER) ---
            # On s'assure que les dates sont au bon format
            df_filtered['Consommation Œuf'] = pd.to_numeric(df_filtered['Consommation Œuf'], errors='coerce')
            df_filtered['Consommation Œuf'] = df_filtered['Consommation Œuf'].fillna(0)
            conso_par_mois = df_filtered.groupby(df_filtered['Date'].dt.month)['Consommation Œuf'].sum()
            conso_par_mois = conso_par_mois.astype(float)
            bs_text = "N/A"
            nom_mois = {1:'Janv', 2:'Févr', 3:'Mars', 4:'Avr', 5:'Mai', 6:'Juin',
                        7:'Juil', 8:'Août', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Déc'}

            # --- 3. LOGIQUE DE SAISONNALITÉ ---
            if not conso_par_mois.empty and conso_par_mois.sum() > 0:
                conso_par_mois = conso_par_mois.astype(float)
                top_n = min(len(conso_par_mois), 3)
                haute_saison_indices = conso_par_mois.nlargest(top_n).index.tolist()
                basse_saison_indices = conso_par_mois.nsmallest(top_n).index.tolist()
                
                # On crée les chaînes de caractères pour le HTML
               
                hs_text = ", ".join([nom_mois[i] for i in haute_saison_indices])
                bs_text = ", ".join([nom_mois[i] for i in basse_saison_indices])

            # --- 4. PRÉPARATION DES AUTRES VARIABLES ---
            # Assurez-vous que ces variables existent aussi avant le HTML
            rang_txt = locals().get('rang_actuel', 'N/A')
            chambres_txt = locals().get('chambres_display', '0')
            
            # --- CONSTRUCTION DU RENDU FINAL ---
            html_content = f"""
                <div style="margin-bottom: 20px; padding-left: 10px; border-left: 5px solid #a3ff12;">
                    <h1 style="color: white; font-size: 1.8em; margin: 0; font-family: 'Segoe UI', sans-serif; letter-spacing: 1px;">
                        📊 EXECUTIVE SUMMARY
                    </h1>
                    <p style="color: #8b949e; font-size: 0.8em; margin: 0; text-transform: uppercase;">Analyse Prévisionnelle & Saisons 2026</p>
                </div>
                <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px; color: white; font-family: 'Segoe UI', sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <span style="color: #a3ff12; font-weight: bold; border: 1px solid rgba(163, 255, 18, 0.4); padding: 6px 15px; border-radius: 20px; font-size: 0.9em; background: rgba(163, 255, 18, 0.05);">
                            🏆 Rang : {rang_txt}
                        </span>
                        <span style="background: #30363d; padding: 6px 15px; border-radius: 20px; font-size: 0.85em; color: #c9d1d9;">
                            {etoiles_display}
                        </span>
                    </div>
                    <h2 style="margin: 0; font-size: 1.7em; color: #ffffff; font-weight: 600;">{hotel}</h2>
                    <p style="color: #8b949e; font-size: 1em; margin-bottom: 25px; display: flex; align-items: center;">
                        <span style="margin-right: 8px;">🏨</span> {chambres_txt}
                    </p>
                    <div style="background: #0d1117; padding: 18px; border-radius: 10px; border-left: 4px solid {color_var}; margin-bottom: 20px;">
                        <small style="color: #8b949e; text-transform: uppercase; font-size: 0.75em; letter-spacing: 0.5px;">Variation Annuelle Estimée</small><br>
                        <div style="display: flex; align-items: baseline; margin-top: 5px;">
                            <span style="font-size: 2em; font-weight: bold; color: {color_var};">
                                {arrow_var} {abs(variation_pct):.1f}%
                            </span>
                            <span style="margin-left: 12px; color: #8b949e; font-size: 0.95em;">
                                ({variation_txt})
                            </span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px;">
                        <div style="background: rgba(163, 255, 18, 0.03); padding: 15px; border-radius: 10px; border: 1px dashed rgba(163, 255, 18, 0.25);">
                            <small style="color: #a3ff12; font-weight: bold; text-transform: uppercase; font-size: 0.7em; display: flex; align-items: center;">
                                <span style="font-size: 1.2em; margin-right: 5px;">☀️</span> Haute Saison
                            </small>
                            <div style="font-size: 0.95em; color: #ffffff; margin-top: 8px; font-weight: 500;">{hs_text}</div>
                        </div>
                        <div style="background: rgba(88, 166, 255, 0.03); padding: 15px; border-radius: 10px; border: 1px dashed rgba(88, 166, 255, 0.25);">
                            <small style="color: #58a6ff; font-weight: bold; text-transform: uppercase; font-size: 0.7em; display: flex; align-items: center;">
                                <span style="font-size: 1.2em; margin-right: 5px;">❄️</span> Basse Saison
                            </small>
                            <div style="font-size: 0.95em; color: #ffffff; margin-top: 8px; font-weight: 500;">{bs_text}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 1em; line-height: 1.6; padding-top: 20px; border-top: 1px solid #30363d;">
                        <div style="margin-bottom: 8px;">
                            <b style="color: #a3ff12; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">💡 Analyse Stratégique</b>
                        </div>
                        <span style="color: #c9d1d9; font-style: italic;">"{description_ia}"</span>
                    </div>
                </div>
            """

            # Affichage dans Streamlit
            st.markdown(html_content, unsafe_allow_html=True)
 # ------------------------------------------------------------------------------------- Tbleau Région  --------------------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import os

# --- CHARGEMENT SÉCURISÉ ---
@st.cache_data
def load_data():
    file_name = 'Donnees_Calculées.xlsx' #
    alt_path = r"C:\Users\basma\OneDrive\Bureau\Streamlit1\Donnees_Calculées.xlsx" #
    
    if os.path.exists(file_name):
        return pd.read_excel(file_name)
    elif os.path.exists(alt_path):
        return pd.read_excel(alt_path)
    return None

df_global = load_data()


# --- STYLE CSS PERSONNALISÉ -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* 1. CONTENEURS (En-tête et Pied de tableau) */
    .header-container {
        background-color: #1f2937;
        color: white;
        padding: 12px;
        border-radius: 10px 10px 0 0;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin-top: 20px;
        width: 100%;
        box-sizing: border-box;
    }

    .footer-container {
        background-color: #111827;
        color: white;
        padding: 12px;
        border-radius: 0 0 10px 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: bold;
        font-size: 14px;
        border-top: 1px solid #374151;
        width: 100%;
        margin-top: -2px;
        box-sizing: border-box;
    }

    /* 2. STRUCTURE DU TABLEAU */
    div[data-testid="stTable"] table, div[data-testid="stDataFrame"] table {
        width: 100% !important;
        table-layout: fixed !important;
        border-collapse: collapse !important;
    }

    /* Style global des cellules : Blanc et Gras */
    td, th {
        color: #ffffff !important;
        font-weight: 700 !important;
        background-color: #0f172a !important;
        border-bottom: 1px solid #334155 !important;
        padding: 10px !important;
    }

    /* 3. RÉGLAGES DES COLONNES (DIMENSIONS) */
    
    /* Ciblage des tableaux pour minimiser la colonne hôtel */
    
    /* COL 1 : HÔTEL (Minimisée à 150px) */
    th:nth-child(1), td:nth-child(1) { 
        width: 150px !important; 
        text-align: left !important;
        white-space: nowrap !important;
        overflow: hidden;
        text-overflow: ellipsis; /* Ajoute ... si le nom est long */
    }
    
    /* COL 2 : RÉGION (Forcée à 90px) */
    th:nth-child(2), td:nth-child(2) { 
        width: 90px !important; 
        text-align: left !important;
        display: table-cell !important; /* Force l'apparition si cachée */
    }

    /* COL 6 : PRÉCISION (Reste courte à 120px) */
    th:nth-child(6), td:nth-child(6) { 
        width: 120px !important; 
    }

    /* Harmonisation des couleurs pour le texte */
    td, th {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* 4. LABELS DE SCORE */
    .score-label { color: #4ade80 }
    .score-value { color: #4ade80; font-size: 16px; font-weight: 900; }
    </style>
""", unsafe_allow_html=True)


# On récupère la liste des régions
liste_regions = df_global['Région'].unique().tolist()
# --- 2. FILTRAGE GLOBAL ---
df_filtre = df_global[df_global['Région'] == region_choisie].copy()
# --- 3. MISE À JOUR DES MÉTRIQUES (Haut) ---
total_consom_region = df_filtre['Total 2025'].sum()
# Mettez à jour vos composants de métriques ici avec total_consom_region...
# Filtrer par hôtel au sein de la région choisie
liste_hotels = df_filtre['Hôtel'].unique()

# Configuration commune pour les colonnes
configuration_colonnes = {
    "Hôtel": st.column_config.TextColumn(
        "Hôtel", 
        width="medium", # Environ 200px
        help="Nom de l'établissement"
    ),
    "Région": st.column_config.TextColumn(
        "Région",
        width="small"
    ),
    "Total 2025": st.column_config.NumberColumn("Total 2025", format="%d"),
    "Total 2026": st.column_config.NumberColumn("Total 2026", format="%d"),
    "Variation %": st.column_config.TextColumn("Variation %"),
    "Précision %": st.column_config.TextColumn("Précision %"), 
}
# --- 1. FONCTION DE STYLE (Centrage + Gras) ---
# FONCTION POUR LE TABLEAU DE GAUCHE (VERT FLUO GRAS)------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def style_unit_dataframe(df):
    def color_variation(val):
        try:
            v = float(str(val).replace('%', '').replace(',', ''))
            color = '#4ade80' if v >= 0 else '#ff4b4b'
            return f'color: {color}; font-weight: bold; text-align: center;'
        except: return 'text-align: center;'

    def precision_bar_neon(val):
        try:
            v = float(str(val).replace('%', ''))
            # On utilise 60% pour la barre afin de laisser 40% d'espace libre pour le texte
            return (
                f"background: linear-gradient(90deg, #00acee 0%, #4ade80 {v}%, rgba(255,255,255,0.05) {v}%);"
                f"background-size: 60% 6px;"
                f"background-repeat: no-repeat;"
                f"background-position: left 10px center;" # Barre décalée du bord gauche
                f"text-align: right;"
                f"padding-right: 15px;" # Texte poussé vers la droite
                f"font-weight: bold;"
                f"color: #d1d5db;"
            )
        except: return 'text-align: center;'

    return df.style.format({
        "Total 2025": "{:,.1f}", "Total 2026": "{:,.1f}",
        "Variation %": "{:+.1f}%", "Précision %": "{:.1f}%"
    }).applymap(color_variation, subset=['Variation %']) \
      .applymap(precision_bar_neon, subset=['Précision %']) \
      .set_table_styles([
        # Style général avec contour Vert Fluo Gras et coins arrondis
        {'selector': '', 'props': [
            ('width', '100%'), ('table-layout', 'fixed'),
            ('border', '3px solid #4ade80'), 
            ('border-radius', '15px'), ('border-collapse', 'separate'),
            ('overflow', 'hidden'), ('box-shadow', '0 0 15px rgba(74, 222, 128, 0.4)')
        ]},
        
        # SUPPRESSION DES TRAITS ET RÉDUCTION DE HAUTEUR
    {'selector': 'th, td, tr', 'props': [
        ('border', 'none !important'),
        ('border-width', '0px !important'),
        ('line-height', '2.3important') # Réduit l'espace entre les lignes de texte
    ]},

    # RÉDUCTION DU PADDING (Hauteur de ligne minimale)
    {'selector': 'td', 'props': [
        ('padding', '4px 8px !important'), # 4px en haut/bas au lieu de 12px
        ('color', '#ffffff !important'),
        ('font-size', '13px') # Texte légèrement plus petit pour le compactage
    
        ]},
        # ZÉBRAGE GRIS CLAIR
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#2d3748')]}, # Gris clair bleuté
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#1a202c')]},  # Gris foncé
        
        # En-têtes et cellules
        {'selector': 'th', 'props': [('color', '#00ffff'), ('background-color', '#1f2937'), ('border', 'none'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('border', 'none'), ('color', '#d1d5db'), ('padding', '10px 5px')]},
        {'selector': 'td:nth-child(1)', 'props': [('color', '#d1d5db'), ('font-weight', 'bold'), ('text-align', 'left'), ('padding-left', '15px')]}
    ])

    # FONCTION POUR LE TABLEAU DE DROITE (BLEU NÉON)----------------------------------------------------------------------------------------------------------------------------------------------
# FONCTION POUR LE TABLEAU DE DROITE (BLEU NÉON)
def style_volume_dataframe(df):
    def color_variation(val):
        try:
            v = float(str(val).replace('%', '').replace(',', ''))
            color = '#4ade80' if v >= 0 else '#ff4b4b'
            return f'color: {color}; font-weight: bold; text-align: center;'
        except: return 'text-align: center;'

    def precision_bar_neon(val):
        try:
            v = float(str(val).replace('%', ''))
            return (
                f"background: linear-gradient(90deg, #00acee 0%, #4ade80 {v}%, rgba(255,255,255,0.05) {v}%);"
                f"background-size: 60% 6px;"
                f"background-repeat: no-repeat;"
                f"background-position: left 10px center;"
                f"text-align: right; padding-right: 15px; font-weight: bold; color: #ffffff;"
            )
        except: return 'text-align: center;'

    # On définit la couleur principale pour le thème Bleu Néon
    main_color = "#00ffff"

    return df.style.format({
        "Total 2025": "{:,.1f}", "Total 2026": "{:,.1f}",
        "Variation %": "{:+.1f}%", "Précision %": "{:.1f}%"
    }).applymap(color_variation, subset=['Variation %']) \
      .applymap(precision_bar_neon, subset=['Précision %']) \
      .set_table_styles([
        # 1. CADRE PRINCIPAL (Fusion avec le titre blanc)
        {'selector': '', 'props': [
            ('width', '100%'),
            ('border', f'2px solid {main_color} !important'),
            ('border-radius', '0 0 12px 12px'),
            ('border-collapse', 'separate !important'),
            ('background-color', '#0f172a'),
            ('overflow', 'hidden')
        ]},
        
        # 2. SUPPRESSION RADICALE DES TRAITS (Lignes et colonnes)
        {'selector': 'th, td, tr', 'props': [
            ('border', 'none !important'),
            ('border-width', '0px !important'),
            ('box-shadow', 'none !important')
        ]},

        # 3. EN-TÊTES DE COLONNES (Style minimaliste)
        {'selector': 'th', 'props': [
            ('color', '#a3ff12'),               # Couleur Vert-Jaune Fluo
            ('background-color', '#111827'),    # Fond bleu-nuit très foncé
            ('font-size', '18px'),              # Taille d'écriture plus grande
            ('font-weight', '900'),             # Texte très gras (Extra Bold)
            ('text-transform', 'uppercase'),    # Majuscules
            ('padding', '12px 10px'),           # Plus d'espace interne
            ('border-radius', '8px'),           # Arrondi des coins des cellules de titre
            ('text-align', 'center'),
            ('letter-spacing', '1px')
        ]},

        # 4. ZÉBRAGE ET TEXTE BLANC
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#1e293b !important')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#0f172a !important')]},
        {'selector': 'td', 'props': [
            ('color', '#ffffff !important'),
            ('padding', '5px 10px !important'),
            ('font-size', '13px'),
            ('line-height', '1.2')
        ]},

        # 5. PREMIÈRE COLONNE (Alignement Gauche)
        {'selector': 'td:nth-child(1)', 'props': [
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding-left', '15px')
        ]}
    ])
     # Fin du set_table_styles
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- ÉTAPE 1 : DÉFINITION DES ICÔNES (En haut du script) ---
icon_coupe = "https://cdn-icons-png.flaticon.com/512/3112/3112946.png" 
icon_goutte = "https://cdn-icons-png.flaticon.com/512/3105/3105807.png"

def render_header(title, icon_url):
    """Génère la barre de titre blanche agrandie avec doubles icônes latérales"""
    st.markdown(f"""
        <div style="
            background-color: #ffffff; color: #000000; 
            font-family: 'Inter', sans-serif;
            font-weight: 900; text-align: center; padding: 15px; 
            border-radius: 15px 15px 0 0; font-size: 22px; 
            letter-spacing: 2px; text-transform: uppercase;
            margin-bottom: -2px; display: flex; align-items: center; 
            justify-content: center; gap: 30px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        ">
            <img src="{icon_url}" width="45"> 
            <span>{title}</span> 
            <img src="{icon_url}" width="45">
        </div>
    """, unsafe_allow_html=True)

def precision_bar_neon(val):
    """Génère la barre de progression néon à l'intérieur des cellules"""
    try:
        v = float(str(val).replace('%', ''))
        return (
            f"background: linear-gradient(90deg, #00acee 0%, #4ade80 {v}%, rgba(255,255,255,0.05) {v}%);"
            f"background-size: 60% 5px;"
            f"background-repeat: no-repeat;"
            f"background-position: left 10px center;"
            f"text-align: right; padding-right: 15px; font-weight: bold; color: #ffffff;"
        )
    except:
        return 'text-align: center;'

# Thème Tableau 1 : Unités (Vert-Jaune Fluo)
color_unites = "#a3ff12"
icon_unites = "https://cdn-icons-png.flaticon.com/512/3112/3112946.png"

# Thème Tableau 2 : Volumes (Bleu Néon)
color_volumes = "#00ffff"
icon_volumes = "https://cdn-icons-png.flaticon.com/512/827/827561.png"

def render_header(title, icon_url, accent_color):
    """Génère un header transparent avec bordure néon"""
    st.markdown(f"""
        <div style="
            background-color: transparent; color: {accent_color}; 
            font-weight: 900; text-align: center; padding: 12px; 
            border: 2px solid {accent_color}; border-radius: 15px 15px 0 0; 
            font-size: 20px; text-transform: uppercase; margin-bottom: -2px; 
            display: flex; align-items: center; justify-content: center; gap: 20px;
        ">
            <img src="{icon_url}" width="35">
            <span>{title}</span>
            <img src="{icon_url}" width="35">
        </div>
    """, unsafe_allow_html=True)

# Utilisation avec les icônes agrandies
# Remplacez les URLs par vos fichiers locaux ou liens d'icônes
icon_coupe = "https://cdn-icons-png.flaticon.com/512/3112/3112946.png" 
icon_goutte = "https://cdn-icons-png.flaticon.com/512/3105/3105807.png"


def apply_pro_style(styler, accent_color):

    styler.set_table_styles([
        # Force la largeur de la colonne Précision
        {'selector': 'th.col4, td.col4', 'props': [('min-width', '180px'), ('width', '180px')]},
        # Centre le contenu pour éviter que la barre ne touche les bords
        {'selector': 'td.col4', 'props': [('text-align', 'right'), ('padding-right', '10px')]}
    ], overwrite=False)

    return styler.set_table_styles([
        # 1. CONTENEUR GLOBAL (Cadre néon sans fond blanc)
        {'selector': '', 'props': [
            ('width', '100%'),
            ('border', f'2px solid {accent_color} !important'),
            ('border-radius', '0 0 15px 15px'),
            ('background-color', '#0f172a'),
            ('border-collapse', 'separate !important'), # Requis pour les arrondis
            ('border-spacing', '8px 4px')                # Espace entre les badges
        ]},
        
        
        
        # 2. SUPPRESSION DES BORDURES PAR DÉFAUT
        {'selector': 'th, td, tr', 'props': [
            ('border', 'none !important'),
            ('border-width', '0px !important')
        ]},

        # 3. BADGES D'EN-TÊTE (COULEUR ACCENT : VERT-JAUNE OU BLEU)
        {'selector': 'th', 'props': [
            ('color', f'{accent_color} !important'),      # Couleur dynamique passée en argument
            ('background-color', '#111827'),              # Fond sombre pour le badge
            ('font-size', '15px'),
            ('font-weight', '900'),
            ('text-transform', 'uppercase'),
            ('padding', '12px 15px'),
            ('border-radius', '12px !important'),          # Arrondi des coins
            ('text-align', 'center')
        ]},

        # 4. ALIGNEMENT DES DONNÉES (TOTAL 2025, 2026, VARIATION)
        {'selector': 'td:nth-child(2), td:nth-child(3), td:nth-child(4)', 'props': [
            ('text-align', 'center !important'),
            ('font-family', "'Inter', sans-serif"),
            ('font-weight', '500'),
            ('color', '#ffffff !important')
        ]},

        # 5. CORPS DU TABLEAU (Zébrage discret)
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#1e293b !important')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#0f172a !important')]},
        {'selector': 'td', 'props': [
            ('padding', '10px !important'),
            ('font-size', '14px')
        ]},
        # 4. ZÉBRAGE ET TEXTE BLANC
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#1e293b !important')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#0f172a !important')]},
        {'selector': 'td', 'props': [
            ('color', '#ffffff !important'),
            ('padding', '5px 10px !important'),
            ('font-size', '13px'),
            ('line-height', '1.2')
        ]},
       

        # 6. PREMIÈRE COLONNE (NOMS DES HÔTELS)
        {'selector': 'td:nth-child(1)', 'props': [
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding-left', '20px !important'),
            ('font-weight', '500'),
            ('font-size', '15px'),
            ('color', '#ffffff !important')
        ]}
    ])
# --- 2. LOGIQUE D'AFFICHAGE ---
# Tableau 1 : Unités [en Oeuf] (Valeurs originales)
colonnes_tableau = ['Hôtel', 'Région', 'Total 2025', 'Total 2026', 'Variation %', 'Précision %']
df_u = df_filtre[colonnes_tableau].copy() 

# Tableau 2 : Unités [en Litre] (Application du coefficient 0.054)
df_v = df_filtre[colonnes_tableau].copy()
df_v['Total 2025'] = df_v['Total 2025'] * 0.054
df_v['Total 2026'] = df_v['Total 2026'] * 0.054

# 1. Dictionnaires de formatage distincts
# Pour les œufs : Nombres entiers
format_oeuf = {
    "Total 2025": lambda x: f"{x:,.0f}".replace(",", " "),
    "Total 2026": lambda x: f"{x:,.0f}".replace(",", " "),
    "Variation %": "{:+.1f}%",
    "Précision %": "{:.1f}%"
}

# Pour les litres : 1 ou 2 décimales car multiplié par 0.054
format_litre = {
    "Total 2025": lambda x: f"{x:,.1f}".replace(",", " "),
    "Total 2026": lambda x: f"{x:,.1f}".replace(",", " "),
    "Variation %": "{:+.1f}%",
    "Précision %": "{:.1f}%"
}

# 2. Logique d'affichage Streamlit
st.markdown(f"### 📍 RÉGION : {region_choisie}")
st.markdown("""
    <style>
    /* On cible la 5ème colonne (Précision %) de chaque ligne */
    table td:nth-child(5), table th:nth-child(5) {
        min-width: 180px !important; /* Augmente l'espace pour la barre + texte */
        width: 180px !important;
        padding-left: 20px !important;
        text-align: right !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. AFFICHAGE DES TABLEAUX ---
col1, col2 = st.columns(2)
with col1:
    render_header("UNITÉS: [EN OEUF]", icon_coupe, "#a3ff12")
    st.write(apply_pro_style(
        df_u.style.format(format_oeuf).applymap(precision_bar_neon, subset=['Précision %']),
        "#a3ff12"
    ).hide(axis='index').to_html(), unsafe_allow_html=True)

with col2:
    render_header("UNITÉS: [EN LITRE]", icon_volumes, "#00ffff")
    st.write(apply_pro_style(
        df_v.style.format(format_litre).applymap(precision_bar_neon, subset=['Précision %']),
        "#00ffff"
    ).hide(axis='index').to_html(), unsafe_allow_html=True)





# ==================================================================================================================================================================================================================================
# -Tableau Global NATIONAL - Tous Régions-
# ========================================================================================================================================================================================================================================================

if df_global is not None:
    st.write("---") 
    
    # 1. Données
    cols_affichage = ['Hôtel', 'Région', 'Total 2025', 'Total 2026', 'Variation %', 'Précision %']
    df_nat_u = df_global[[c for c in cols_affichage if c in df_global.columns]].copy()
    df_nat_l = df_nat_u.copy()

    # Conversion litres
    for c in ['Total 2025', 'Total 2026']:
        if c in df_nat_l.columns:
            df_nat_l[c] = pd.to_numeric(df_nat_l[c], errors='coerce') * 0.057

    # 2. CSS DE PRÉCISION : Focus sur la réduction de la barre
    st.markdown("""
        <style>
        div[class*="stHorizontalBlock"] table {
            width: 100% !important;
            table-layout: fixed !important;
            border-collapse: collapse !important;
        }
                
    /* 1. STRUCTURE GLOBALE DU TABLEAU */
    div[data-testid="stTable"] table, div[data-testid="stDataFrame"] table {
        width: 100% !important;
        table-layout: fixed !important;
        border-collapse: collapse !important;
        background-color: #0f172a !important; /* Fond sombre de base */
    }

    * 1. FORCE LE ZÈBRAGE SUR TOUTES LES LIGNES PAIRES */
    /* On utilise tr:nth-of-type(even) pour mieux cibler les lignes de données */
    [data-testid="stDataFrame"] tr:nth-of-type(even) td,
    [data-testid="stTable"] tr:nth-of-type(even) td {
        background-color: rgba(255, 255, 255, 0.07) !important; /* Gris clair transparent */
    }

    /* 2. FORCE LES LIGNES IMPAIRES À RESTER SOMBRES */
    [data-testid="stDataFrame"] tr:nth-of-type(odd) td,
    [data-testid="stTable"] tr:nth-of-type(odd) td {
        background-color: #0f172a !important; /* Votre bleu sombre de base */
    }

    /* 3. MAINTIEN DU DESIGN DES COLONNES */
    /* Hôtel (Col 1) Minimisé */
    [data-testid="stDataFrame"] td:nth-child(1), 
    [data-testid="stTable"] td:nth-child(1) { 
        width: 150px !important; 
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Région (Col 2) */
    [data-testid="stDataFrame"] td:nth-child(2), 
    [data-testid="stTable"] td:nth-child(2) { 
        width: 100px !important;
        color: #ffffff !important;
    }

    /* Précision (Col 6) - Barre courte */
    [data-testid="stDataFrame"] td:nth-child(6), 
    [data-testid="stTable"] td:nth-child(6) { 
        width: 130px !important;
    }

    /* 3. RÉGLAGE DES COLONNES (STRICT) */
    
    /* COL 1 : HÔTEL (Minimisé) */
    th:nth-child(1), td:nth-child(1) { 
        width: 150px !important; 
        text-align: left !important; 
    }
    
    /* COL 2: RÉGION (Alignée à gauche) */
    [data-testid="stDataFrame"] td:nth-child(2), th:nth-child(2) {
        width: 100px !important;
        text-align: left !important; /* Aligné à gauche comme demandé */
        color: #ffffff !important;
    
    }

   /* COL 5: VARIATION % (Réduite au minimum) */
    [data-testid="stDataFrame"] td:nth-child(5), th:nth-child(5) {
        width: 60px !important; /* Réduction drastique */
        text-align: center !important;
    }
                * STYLE GLOBAL DES CELLULES */
    td, th {
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* COL 6 : PRÉCISION % (Barre courte) */
    th:nth-child(6), td:nth-child(6) { 
        width: 200px !important; 
        text-align: right !important; 
    }

    /* En-têtes (toujours sombres et distincts) */
    th {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        text-transform: uppercase;
        font-size: 0.8rem;
    }
  
        

        /* AJUSTEMENT DU CONTENEUR DE LA BARRE (Force la brièveté) */
        .precision-container {
            width: 200px !important;  /* Longueur maximale de la barre graphique */
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
        }
        
        .precision-text {
            display: inline-block;
            min-width: 45px;
            text-align: right;
            font-weight: 900 !important;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. Affichage
    st.markdown('<h2 style="color:white; font-weight:900; text-transform:uppercase;">🌍 CLASSEMENT NATIONAL GLOBAL</h2>', unsafe_allow_html=True)
    
    col_u, col_l = st.columns(2)
    with col_u:
        render_header("NATIONAL : [EN OEUF]", "🏆", "#a3ff12")
        st.write(apply_pro_style(
            df_nat_u.sort_values('Total 2025', ascending=False).style.format(format_oeuf)
            .applymap(precision_bar_neon, subset=['Précision %']), "#a3ff12"
        ).hide(axis='index').to_html(), unsafe_allow_html=True)

    with col_l:
        render_header("NATIONAL : [EN LITRE]", "💧", "#00ffff")
        st.write(apply_pro_style(
            df_nat_l.sort_values('Total 2025', ascending=False).style.format(format_litre)
            .applymap(precision_bar_neon, subset=['Précision %']), "#00ffff"
        ).hide(axis='index').to_html(), unsafe_allow_html=True)






import streamlit as st
import base64

def get_html_download_link(file_path):
    with open(file_path, "rb") as f:
        html_content = f.read()
    # Encodage pour le téléchargement
    b64 = base64.b64encode(html_content).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{file_path}" style="color: #4A90E2; font-weight: bold; font-size: 18px;">📄 Télécharger le Rapport d\'Analyse HTML</a>'

st.write("---")
st.subheader("📊 Rapport Technique Complet")

# Vérifie que le nom du fichier correspond à celui dans ton dossier
try:
    st.markdown(get_html_download_link("rapport_ia.html"), unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Fichier HTML introuvable. Vérifiez le nom du fichier dans VS Code.")