import io

from io import BytesIO

import re

import numpy as np

import pandas as pd

import plotly.express as px

import streamlit as st

from unidecode import unidecode

import base64

# -------------------- CONFIG GÉNÉRALE --------------------

st.set_page_config(page_title="Qualité & Uniformisation – Tableaux Monday", layout="wide")

# Palette de couleurs Accor Premium

ACCOR_COLS = ["#1d3557", "#457b9d", "#a8dadc", "#f1faee", "#bde0fe", "#cdb4db"]


# Fonction : lire le logo Accor et le convertir en base64

def get_base64_image(image_path: str) -> str:

    with open(image_path, "rb") as img_file:

        return base64.b64encode(img_file.read()).decode()


image_base64 = get_base64_image("logo_accor.png")

# --- HEADER / BANNIÈRE ACCOR équilibrée ---

st.markdown(

    f"""
<style>

  /* Retirer la marge de Streamlit autour du contenu */

  .block-container {{

      padding-top: 5%;

      padding-bottom: 50%;

      max-width: 100%;

  }}

  /* Bannière dorée pleine largeur */

  .fullwidth-header {{

      width: 100%;

      background-color: #fef6e4;

      display: flex;

      align-items: center;

      justify-content: space-between;

      padding: 18px 70px;

      box-sizing: border-box;

      box-shadow: 0 4px 8px rgba(0,0,0,0.05);

      border-bottom: 1px solid #e5d9b6;

  }}

  /* Logo Accor */

  .fullwidth-header img {{

      height: 110px;  /*  Taille bien équilibrée */

  }}

  /* Titre */

  .fullwidth-header .title {{

      flex-grow: 1;

      text-align: center;

      font-size: 40px;  /*  Taille de titre équilibrée */

      font-weight: 700;

      color: #1d3557;

      letter-spacing: 0.5px;

  }}

  /* Sous-titre */

  .subtext {{

      text-align: center;

      color: #666;

      padding-left: 20%;

      font-size: 15px;

      margin-top: 10px;

      margin-bottom: 55px;

  }}

  /* KPI cards */

  .kpi-container {{

      display: flex;

      justify-content: space-around;

      margin-top: 25px;

      margin-bottom: 35px;

      gap: 10px;

  }}

  .kpi-card {{

      background-color: white;

      border: 2px solid #e5d9b6;

      border-radius: 12px;

      width: 23%;

      padding: 20px;

      text-align: center;

      box-shadow: 0 4px 8px rgba(0,0,0,0.1);

      transition: transform 0.2s ease;

  }}

  .kpi-card:hover {{

      transform: translateY(-4px);

  }}

  .kpi-title {{

      color: #1d3557;

      font-weight: 600;

      font-size: 18px;

      margin-bottom: 10px;

  }}

  .kpi-value {{

      font-size: 32px;

      font-weight: 800;

      color: #1d3557;

  }}

  .kpi-subtext {{

      font-size: 12px;

      color: #666;

      margin-top: 4px;

  }}
</style>
<!-- Bannière -->
<div class="fullwidth-header">
<img src="data:image/png;base64,{image_base64}" alt="Accor Logo">
<div class="title"> Qualité & Uniformisation – Tableaux Monday</div>
</div>
<!-- Sous-titre -->
<p class="subtext">

    Analyse de cohérence et de complétude des données issues de Monday
    
</p>

""",

    unsafe_allow_html=True,

)

# -------------------- PARAMS --------------------

# Colonnes à analyser (tu peux adapter l'intitulé EXACT de Monday ici)

COLONNES_CIBLES = [

    "Catégorie",

    "Clôture / Qualité",

    "Outil",

    "Outils secondaires",  # l'app acceptera aussi "Outils secondaire"

    "Pôle",

    "Cycle comptable",

    "Temps moyen théorique en h (1=1h, 0,25 = 15min)",

    "Récurrence",

    "Mois de clôture",

    "Statut",

    "Sociétés juridiques",

    "Niveau de pénibilité (ressenti)",

    "Notifier comptable",

    "Statut de la procédure",

    "Statut du mode opératoire",

]

# Variantes tolérées de noms de colonnes -> nom canonique

CANON_MAP = {

    "outils secondaire": "Outils secondaires",

    "outils secondaires": "Outils secondaires",

    "pole": "Pôle",

    "pôle": "Pôle",

    "societes juridiques": "Sociétés juridiques",

    "mois de cloture": "Mois de clôture",

    "temps moyen theorique en h (1=1h, 0,25 = 15min)": "Temps moyen théorique en h (1=1h, 0,25 = 15min)",

    "cloture / qualite": "Clôture / Qualité",

    "statut du mode operatoire": "Statut du mode opératoire",

    "statut de la procedure": "Statut de la procédure",

    "niveau de penibilite (ressenti)": "Niveau de pénibilité (ressenti)",

}


# -------------------- HELPERS --------------------

def normalise_col(c: str) -> str:

    """Normalise un nom de colonne (pour matcher les variantes)."""

    key = unidecode(str(c)).lower().strip()

    if key in CANON_MAP:

        return CANON_MAP[key]

    # Si le nom correspond exactement à une cible, le garder

    for cible in COLONNES_CIBLES:

        if unidecode(cible).lower().strip() == key:

            return cible

    # sinon renvoyer le nom original

    return c


def read_any_table(uploaded_file: BytesIO) -> pd.DataFrame:

    name = uploaded_file.name.lower()

    if name.endswith(".ods"):

        # moteur LibreOffice

        return pd.read_excel(uploaded_file, engine="odf")

    else:

        return pd.read_excel(uploaded_file)


def completeness_table(df: pd.DataFrame, colonne: str, by_pole: bool = True) -> pd.DataFrame:

    """Calcule le taux de remplissage global et par pôle en nettoyant les fausses valeurs vides."""

    if colonne not in df.columns:

        return pd.DataFrame(columns=["Colonne", "Pôle", "Remplies", "Vides", "Taux (%)"])

    # Nettoyage local : suppression des NaN, espaces, tabulations, retours ligne, "nan"/"None"

    s = (

        df[colonne]

        .astype(str)

        .replace(["nan", "NaN", "None"], "")

        .str.replace(r"[\r\n\t]", "", regex=True)

        .str.strip()

    )

    total = len(s)

    non_vides = s.apply(lambda x: bool(x)).sum()

    taux = round(non_vides / total * 100, 2) if total > 0 else 0

    out = [

        {

            "Colonne": colonne,

            "Pôle": "Global",

            "Remplies": non_vides,

            "Vides": total - non_vides,

            "Taux (%)": taux,

        }

    ]

    if by_pole and "Pôle" in df.columns:

        for pole, sub in df.groupby("Pôle", dropna=False):

            s_p = (

                sub[colonne]

                .astype(str)

                .replace(["nan", "NaN", "None"], "")

                .str.replace(r"[\r\n\t]", "", regex=True)

                .str.strip()

            )

            total_p = len(s_p)

            non_vides_p = s_p.apply(lambda x: bool(x)).sum()

            taux_p = round(non_vides_p / total_p * 100, 2) if total_p > 0 else 0

            out.append(

                {

                    "Colonne": colonne,

                    "Pôle": pole,

                    "Remplies": non_vides_p,

                    "Vides": total_p - non_vides_p,

                    "Taux (%)": taux_p,

                }

            )

    return pd.DataFrame(out)


def is_multivalue_column(series: pd.Series) -> bool:

    """Détecte si une colonne contient souvent plusieurs valeurs dans une même cellule."""

    sample = series.dropna().astype(str)

    multi_ratio = sample.str.contains(r"[,\n;]", regex=True).mean()

    return multi_ratio > 0.05  # seuil 5% de cellules multi-valuées


def unique_values_table(df: pd.DataFrame, colonne: str) -> pd.DataFrame:

    """Retourne les valeurs uniques d'une colonne."""

    if colonne not in df.columns:

        return pd.DataFrame(columns=["Colonne", "Valeur unique"])

    series = df[colonne].fillna("").astype(str)

    # Détecte si cette colonne doit être éclatée

    vals = []

    if is_multivalue_column(series):

        for cell in series:

            parts = re.split(r"[,\n;]", str(cell))

            for p in parts:

                p = p.strip()

                if p:

                    vals.append(p)

    else:

        vals = [v.strip() for v in series if v.strip() not in ["", "nan", "None"]]

    uniques = sorted(set(vals))

    return pd.DataFrame({"Colonne": colonne, "Valeur unique": uniques})


def normalize_value_for_compare(v: str) -> str:

    """Normalise un texte pour repérer les incohérences d'écriture (même apparence mais encodage différent)."""

    # Convertir en chaîne, retirer caractères invisibles

    v = str(v)

    v = v.replace("\xa0", " ")  # supprime les espaces insécables

    v = v.replace("\u200b", "")  # supprime les caractères zéro-width (invisibles)

    v = re.sub(r"[\r\n\t]", " ", v)  # supprime retours ligne et tabulations

    v = unidecode(v).lower().strip()  # minuscule, sans accents, sans espaces bords

    v = re.sub(r"[\s\-_]+", " ", v)  # espaces multiples ou tirets → espace unique

    v = re.sub(r"[^\w\s&/]", "", v)  # supprime tout sauf lettres, chiffres, & et /

    return v


def incoherence_table(df: pd.DataFrame, colonne: str):

    """

    Analyse les incohérences d’une colonne :

    - normalise les écritures

    - détecte les variantes

    - identifie exactement les fichiers où chaque variante apparaît

    """

    if colonne not in df.columns:

        return (

            pd.DataFrame(columns=["Colonne", "Clé normalisée", "Variantes",

                                  "Nb variantes", "Total occ."]),#, "Fichiers concernés"]),

            pd.DataFrame(columns=["Colonne", "Valeur unique propre"])

        )

    # On récupère valeur + fichier source

    if "__source_file__" in df.columns:

        series = df[[colonne, "__source_file__"]].fillna("")

        use_source = True

    elif "Source fichier" in df.columns:

        series = df[[colonne, "Source fichier"]].fillna("")

        series = series.rename(columns={"Source fichier": "__source_file__"})

        use_source = True

    else:

        # Pas possible d'identifier le fichier → analyse sans source

        series = df[[colonne]].fillna("")

        series["__source_file__"] = ""

        use_source = False

    vals = []

    # Gestion multi-valeurs

    for _, row in series.iterrows():

        cell = str(row[colonne])

        fichier = row["__source_file__"]

        parts = re.split(r"[,\n;]", cell)

        for p in parts:

            p = p.strip()

            if p:

                vals.append((p, fichier))

    if not vals:

        return (

            pd.DataFrame(columns=["Colonne", "Clé normalisée", "Variantes",

                                  "Nb variantes", "Total occ."]), #"Fichiers concernés"]),

            pd.DataFrame(columns=["Colonne", "Valeur unique propre"])

        )

    tmp = pd.DataFrame(vals, columns=["val", "file"])

    tmp["key"] = tmp["val"].apply(normalize_value_for_compare)

    incoherences = []

    propres = []

    for key, g in tmp.groupby("key"):

        variantes = sorted(g["val"].unique())

        fichiers = sorted([f for f in g["file"].unique() if f != ""])

        total = len(g)

        # Plusieurs variantes → incohérence

        if len(variantes) > 1:

            incoherences.append({

                "Colonne": colonne,

                "Clé normalisée": key,

                "Variantes": ", ".join(variantes),

                "Nb variantes": len(variantes),

                "Total occ.": total,

                #"Fichiers concernés": ", ".join(fichiers) if use_source else ""

            })

        else:

            # Une seule variante → valeur propre

            propres.append({

                "Colonne": colonne,

                "Valeur unique propre": variantes[0],

               # "Fichiers concernés": ", ".join(fichiers) if use_source else ""

            })

    incoh_df = pd.DataFrame(incoherences)

    propres_df = pd.DataFrame(propres)

    return incoh_df, propres_df
 

def to_excel_bytes(dfs: dict) -> bytes:

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:

        for sheet, frame in dfs.items():

            frame.to_excel(writer, sheet_name=sheet, index=False)

    buffer.seek(0)

    return buffer.getvalue()


# -------------------- UI : UPLOAD & OPTIONS --------------------

st.caption(

    "Charge un ou plusieurs exports Excel **.xlsx** ou **.ods** fusionnés depuis Monday. "

    "L’app calcule : KPIs, complétude globale & par pôle, valeurs distinctes, "

    "et **incohérences d’écriture** à corriger dans Monday."

)

uploaded_files = st.file_uploader(

    "Fichiers d'entrée (.xlsx ou .ods) – vous pouvez en charger plusieurs",

    type=["xlsx", "ods"],

    accept_multiple_files=True,

)

if not uploaded_files:

    st.info(" Importez au moins un fichier pour lancer l’analyse.")

    st.stop()

dfs_import = []

for f in uploaded_files:

    df_tmp = read_any_table(f)

    df_tmp["Source fichier"] = f.name

    dfs_import.append(df_tmp)

df = pd.concat(dfs_import, ignore_index=True)

df["__source_file__"] = df["Source fichier"]
 

st.success(f" {len(uploaded_files)} fichier(s) importé(s) – {len(df):,} lignes fusionnées")

# Options : colonnes analytiques à utiliser (heatmap + taux de remplissage analyse)

with st.expander(" Options d’analyse", expanded=False):

    cols_selection = st.multiselect(

        "Colonnes à analyser (pour la heatmap & le taux de remplissage d’analyse)",

        options=COLONNES_CIBLES,

        default=COLONNES_CIBLES,

    )

# -------------------- MODE D'ANALYSE --------------------

st.write("")

st.write("###  Choisir le mode d’analyse")

mode_unique = st.radio(

    "Comment analyser les données ?",

    ["Par ligne (fichier brut)", "Par tâche unique (colonne Name)"],

    index=0,

    horizontal=True,

)

# --- Nettoyage et uniformisation de base ---

if "Pôle" in df.columns:

    df["Pôle"] = (

        df["Pôle"]

        .astype(str)

        .str.strip()

        .str.upper()

        .replace({"NAN": ""})

    )

# --- Mode 'Par tâche unique' ---

if mode_unique == "Par tâche unique (colonne Name)" and "Name" in df.columns:

    lignes_avant = len(df)

    # Supprimer uniquement les lignes strictement identiques sur toutes les colonnes

    df = df.drop_duplicates(keep="first")

    lignes_apres = len(df)

    nb_doublons = lignes_avant - lignes_apres

    st.info(

        f" {nb_doublons} doublon(s) strict(s) supprimé(s). "

        f"{lignes_apres:,} lignes uniques restantes."

    )

else:

    st.caption(

        "Chaque ligne des fichiers est considérée comme un enregistrement brut "

        "(aucune suppression de doublons)."

    )

# Normaliser les noms de colonnes

df.columns = [normalise_col(c) for c in df.columns]

# Feedback colonnes manquantes

missing = [c for c in cols_selection if c not in df.columns]

if missing:

    st.warning("Colonnes non trouvées dans les fichiers : " + ", ".join(missing))

# Retenir seulement colonnes présentes

cols_presentes = [c for c in cols_selection if c in df.columns]

if mode_unique == "Par tâche unique (colonne Name)":

    st.write("**Analyse effectuée sur les tâches uniques** (doublons supprimés).")

else:

    st.write("**Analyse effectuée sur les lignes brutes des fichiers fusionnés.**")

# -------------------- KPIs --------------------

nb_fichiers = len(uploaded_files)

nb_taches = len(df)

nb_poles = df["Pôle"].nunique(dropna=True) if "Pôle" in df.columns else 0

taux_globaux = []

for c in cols_presentes:

    s = df[c].astype(str)

    taux_globaux.append((s.str.strip().ne("").sum() / len(df)) * 100)

taux_moy = round(float(np.mean(taux_globaux)), 2) if taux_globaux else 0.0

st.markdown(

    f"""
<div class="kpi-container">
<div class="kpi-card">
<div class="kpi-title">Fichiers importés</div>
<div class="kpi-value">{nb_fichiers}</div>
<div class="kpi-subtext">Exports Monday fusionnés</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Nombre total de tâches</div>
<div class="kpi-value">{nb_taches:,}</div>
<div class="kpi-subtext">Lignes après concaténation</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Pôles distincts</div>
<div class="kpi-value">{nb_poles}</div>
<div class="kpi-subtext">Valeurs uniques de la colonne Pôle</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Taux de remplissage (analyse)</div>
<div class="kpi-value">{taux_moy:.2f}%</div>
<div class="kpi-subtext">Basé uniquement sur les colonnes analytiques (heatmap)</div>
</div>
</div>

""",

    unsafe_allow_html=True,

)

st.write("")

st.write("")

# -------------------- VISU 1 : TÂCHES PAR PÔLE --------------------

if "Pôle" in df.columns:

    taches_pole = (

        df["Pôle"]

        .fillna("NON RENSEIGNÉ")

        .value_counts()

        .reset_index()

    )

    taches_pole.columns = ["Pôle", "Nombre_de_taches"]

    fig = px.bar(

        taches_pole,

        x="Pôle",

        y="Nombre_de_taches",

        title="Nombre de tâches par pôle",

        text="Nombre_de_taches",

        color="Pôle",

        color_discrete_sequence=ACCOR_COLS,

    )

    fig.update_traces(

        textfont_size=14,

        textposition="outside",

        marker_line_color="#fef6e4",

        marker_line_width=1.5,

    )

    fig.update_layout(

        plot_bgcolor="white",

        paper_bgcolor="white",

        title_font=dict(size=20, color="#1d3557", family="Segoe UI"),

        font=dict(size=14, color="#1a1a1a"),

        margin=dict(l=20, r=20, t=60, b=20),

        showlegend=False,

    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.write("")

st.write("")

# -------------------- COMPLETENESS --------------------

st.subheader("Taux de remplissage (global & par pôle)")

taux_frames = []

for c in cols_presentes:

    taux_frames.append(completeness_table(df, c, by_pole=True))

taux = (

    pd.concat(taux_frames, ignore_index=True)

    if taux_frames

    else pd.DataFrame(columns=["Colonne", "Pôle", "Remplies", "Vides", "Taux (%)"])

)

st.write("")

# Heatmap par pôle

if not taux.empty and "Pôle" in taux.columns:

    pivot = taux.pivot_table(

        index="Colonne",

        columns="Pôle",

        values="Taux (%)",

        aggfunc="mean",

    )

    fig2 = px.imshow(

        pivot,

        text_auto=True,

        aspect="auto",

        color_continuous_scale="Blues",

        title="Heatmap – Taux de remplissage par colonne et par pôle",

    )

    fig2.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))

    st.plotly_chart(fig2, use_container_width=True)

st.dataframe(

    taux.sort_values(["Colonne", "Pôle"]),

    use_container_width=True,

    height=420,

)

st.divider()

st.write("")

st.write("")

# -------------------- TOP 5 COLONNES LES MOINS REMPLIES --------------------

st.subheader("Top 5 des colonnes les moins remplies (global)")

st.write("")

try:

    # On trie du plus FAIBLE taux au plus FORT

    taux_global = taux[taux["Pôle"] == "Global"].sort_values(

        "Taux (%)", ascending=True

    )

    top5 = taux_global.head(5)

    fig_top5 = px.bar(

        top5,

        y="Colonne",

        x="Taux (%)",

        orientation="h",

        text="Taux (%)",

        color="Taux (%)",

        color_continuous_scale=[

            "#fef6e4",  # Doré clair

            "#a8dadc",  # Bleu pastel

            "#457b9d",  # Bleu doux

            "#1d3557",  # Bleu marine Accor

        ],

        title="Colonnes",

    )

    fig_top5.update_traces(

        texttemplate="%{x:.2f} %",

        textposition="outside",

        marker_line_color="#fef6e4",

        marker_line_width=1.5,

    )

    fig_top5.update_layout(

        paper_bgcolor="white",

        plot_bgcolor="white",

        title_font=dict(size=20, color="#1d3557", family="Segoe UI"),

        font=dict(size=13, color="#1a1a1a"),

        xaxis_title="Taux de remplissage (%)",

        yaxis_title=None,

        height=450,

        margin=dict(l=30, r=30, t=70, b=30),

        yaxis=dict(autorange="reversed"),

        coloraxis_colorbar=dict(

            title=dict(

                text="Taux (%)",

                font=dict(color="#1d3557", size=12, family="Segoe UI"),

            ),

            tickfont=dict(color="#1d3557", size=11),

        ),

    )

    st.plotly_chart(fig_top5, use_container_width=True)

except Exception as e:

    st.error(f"Erreur lors de la génération du graphique : {e}")

# -------------------- VALEURS UNIQUES & INCOHERENCES --------------------

st.write("")

st.write("")

st.subheader("Valeurs distinctes & incohérences d’écriture")

st.write("")

# Calcul global une seule fois (puis utilisé dans les onglets + export)

incoh_list = []

propres_list = []

uniques_list = []

for c in cols_presentes:

    incoh_c, clean_c = incoherence_table(df, c)

    if not incoh_c.empty:

        incoh_list.append(incoh_c)

    if not clean_c.empty:

        propres_list.append(clean_c)

    uniques_list.append(unique_values_table(df, c))

incoherences = (

    pd.concat(incoh_list, ignore_index=True)

    if incoh_list

    else pd.DataFrame(

        columns=[

            "Colonne",

            "Clé normalisée",

            "Variantes",

            "Nb variantes",

            "Total occ.",

            #"Fichiers concernés",

        ]

    )

)

propres_final = (

    pd.concat(propres_list, ignore_index=True)

    if propres_list

    else pd.DataFrame(

        columns=["Colonne", "Valeur unique propre" ] #"Fichiers concernés"]

    )

)

uniques = (

    pd.concat(uniques_list, ignore_index=True)

    if uniques_list

    else pd.DataFrame(columns=["Colonne", "Valeur unique"])

)

tab1, tab2, tab3 = st.tabs(

    [

        "Valeurs propres (sans variantes)",

        "Incohérences (écritures différentes)",

        "Valeurs brutes (toutes les écritures)",

    ]

)

# --- Valeurs propres ---

with tab1:

    st.dataframe(propres_final, use_container_width=True, height=450)

    st.caption(

        "Ces valeurs apparaissent sous une seule forme dans les fichiers "

        "(aucune variante détectée)."

    )

# --- Incohérences ---

with tab2:

    st.dataframe(incoherences, use_container_width=True, height=450)

    if incoherences.empty:

        st.info("Aucune incohérence détectée. ")

    else:

        st.caption(

            "Chaque ligne représente un groupe de valeurs qui devraient être homogènes, "

            "mais qui sont écrites différemment. La colonne **Fichiers concernés** "

            "indique dans quels fichiers on trouve ces incohérences."

        )

# --- Valeurs brutes ---

with tab3:

    st.dataframe(uniques, use_container_width=True, height=450)

    st.caption(

        " Ce tableau liste toutes les écritures brutes (avant regroupement ou normalisation)."

    )

# -------------------- EXPORT EXCEL --------------------

st.write("")

st.write("")

st.subheader(" Exporter les résultats")

synthese_poles = (

    df["Pôle"]

    .value_counts()

    .rename_axis("Pôle")

    .reset_index(name="Nombre de tâches")

    if "Pôle" in df.columns

    else pd.DataFrame()

)

synthese_fichiers = (

    df["Source fichier"]

    .value_counts()

    .rename_axis("Source fichier")

    .reset_index(name="Nombre de lignes")

    if "Source fichier" in df.columns

    else pd.DataFrame()

)

dfs_export = {

    "Synthese_poles": synthese_poles,

    "Synthese_fichiers": synthese_fichiers,

    "Taux_remplissage": taux,

    "Valeurs_uniques": uniques,

    "Incoherences": incoherences,

    "Valeurs_propres": propres_final,

}

excel_bytes = to_excel_bytes(dfs_export)

st.write("")

st.download_button(

    label=" Télécharger l’analyse complète (Excel)",

    data=excel_bytes,

    file_name="analyse_taches_fusionnees.xlsx",

    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

)
