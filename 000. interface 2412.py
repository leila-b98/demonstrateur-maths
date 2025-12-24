# app.py
import streamlit as st
import pandas as pd
import numpy as np

# --- import robuste du module (Deepnote / Streamlit)
import os, sys, importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_FILENAME = "module_functions_core_science_3.py"
MODULE_PATH = os.path.join(SCRIPT_DIR, MODULE_FILENAME)

if not os.path.exists(MODULE_PATH):
    raise FileNotFoundError(
        f"Impossible de trouver {MODULE_FILENAME} dans le dossier du script.\n"
        f"Script dir: {SCRIPT_DIR}\n"
        f"Fichiers: {os.listdir(SCRIPT_DIR)}"
    )

spec = importlib.util.spec_from_file_location("module_functions_core_science_3", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["module_functions_core_science_3"] = m
spec.loader.exec_module(m)

# --- Streamlit config
st.set_page_config(page_title="Simulateur Parcoursup", layout="wide")


# =========================
# Chargement des données
# =========================
@st.cache_data
def load_data():
    # df_select = pd.read_pickle("df_int_results_admis.pkl")
    # df_int_select = pd.read_pickle("df_int_results_proposal.pkl")
    df_select = pd.read_pickle("df_select.pkl") #Résultats sur les admis
    df_int_select = pd.read_pickle("df_int_select.pkl")
    df_int_edu_bac_stats = pd.read_csv("int_edu_bac_stats.csv")
    df_int_edu_bac_stats_eds = pd.read_pickle("int_edu_bac_stats_eds_computed.pkl")
    df_edu_bac_stats = pd.read_pickle("edu_bac_stats.pkl")

    # --- NOUVEAU : table UAI -> lycée + académie
    df_uai_ref = pd.read_pickle("df_uai_results_bac.pkl")

    return (
        df_select,
        df_int_select,
        df_int_edu_bac_stats,
        df_int_edu_bac_stats_eds,
        df_edu_bac_stats,
        df_uai_ref,
    )


(
    df_select,
    df_int_select,
    df_int_edu_bac_stats,
    df_int_edu_bac_stats_eds,
    df_edu_bac_stats,
    df_uai_ref,
) = load_data()

# branche EDS globale utilisée dans le module
m.df_eds = df_int_edu_bac_stats_eds

# tables principales
df_proposition = df_int_select
df_admis = df_select
df_edu = df_edu_bac_stats
df_int_edu = df_int_edu_bac_stats


# =========================
# UI – constantes
# =========================
st.title("🎓 Démonstrateur score Parcoursup 2025")

st.link_button(
    "📘 Explications sur le modèle 📘 ",
    "https://www.canva.com/design/DAG8aLOyIs0/gaIzrqdGvFNsIczoMI4eBg/view?utm_content=DAG8aLOyIs0&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h7c306db998#1"
)

def rgb_tuple_to_hex(rgb):
    r, g, b = [int(255 * x) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


COLORS = {
    "prop_recues": rgb_tuple_to_hex(m.c_1),
    "prop_acceptees": rgb_tuple_to_hex(m.c_2),
    "taux_mentions": rgb_tuple_to_hex(m.c_3),
}

# --- ordre colonnes : taux_mentions d'abord
MODELES = [
    ("taux_mentions", "Aires par mention"),
    ("prop_recues", "Propositions reçues"),
    ("prop_acceptees", "Propositions acceptées"),
]


# =========================
# Préparation menu UAI
# =========================
def _prepare_uai_reference(df: pd.DataFrame) -> pd.DataFrame:
    # On garde uniquement les colonnes utiles, on nettoie
    cols = ["school_uai", "academy_code", "school_title"]
    df = df.copy()
    df = df[[c for c in cols if c in df.columns]].copy()

    # normalisation minimale
    if "school_uai" in df.columns:
        df["school_uai"] = df["school_uai"].astype(str).str.strip().str.upper()
    if "school_title" in df.columns:
        df["school_title"] = df["school_title"].astype(str).str.strip()
    if "academy_code" in df.columns:
        df["academy_code"] = df["academy_code"].astype(str).str.strip()

    # enlever lignes vides
    df = df.dropna(subset=["school_uai"]).copy()
    df = df[df["school_uai"].str.len() > 0].copy()

    # dédoublonnage (un UAI -> une ligne)
    df = df.drop_duplicates(subset=["school_uai"]).reset_index(drop=True)
    return df


df_uai_menu = _prepare_uai_reference(df_uai_ref)

# options du selectbox : on ajoute une option "aucun"
UAI_NONE_KEY = "__NONE__"
uai_options = [UAI_NONE_KEY] + df_uai_menu["school_uai"].tolist()

# map uai -> infos (pour affichage)
_uai_to_info = (
    df_uai_menu.set_index("school_uai")[["school_title", "academy_code"]]
    .to_dict(orient="index")
)


def format_uai_option(uai_key: str) -> str:
    if uai_key == UAI_NONE_KEY:
        return "— Aucun (désactiver bonus lycée) —"
    info = _uai_to_info.get(uai_key, {})
    title = info.get("school_title", "")
    acad = info.get("academy_code", "")
    if title and acad:
        return f"{uai_key} — {title} (acad {acad})"
    if title:
        return f"{uai_key} — {title}"
    return str(uai_key)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Entrées")

    note = st.slider("Note", 10.0, 20.0, 15.0, 0.1)
    type_bac = st.radio(
        "Type de bac", ["general", "pro", "techno", "all"], horizontal=True
    )

    rang_label = st.radio(
        "Rang dans la classe", ["top1", "top5", "autres"], horizontal=True
    )
    rang = None if rang_label == "autres" else rang_label

    # --- NOUVEAU : UAI via menu déroulant (recherchable)
    selected_uai_key = st.selectbox(
        "Lycée d'origine (UAI / nom)",
        options=uai_options,
        format_func=format_uai_option,
        index=0,
    )

    # uai final utilisé par le bonus
    uai = "" if selected_uai_key == UAI_NONE_KEY else str(selected_uai_key).strip().upper()

    # affichage infos lycée + académie quand sélectionné
    if uai:
        info = _uai_to_info.get(uai, {})
        st.caption(f"🏫 **{info.get('school_title','')}**  \n🎓 Académie : **{info.get('academy_code','')}**")

    ids_ok = m.get_valid_ids(df_proposition, df_admis, type_bac=type_bac)
    if not ids_ok:
        st.error("Aucune formation disponible pour ce type de bac.")
        st.stop()

    default_id = 31463 if 31463 in ids_ok else ids_ok[0]
    id_parcoursup = st.selectbox(
        "id_parcoursup", ids_ok, index=ids_ok.index(default_id)
    )


ok, msg = m.check_id_and_bac(df_proposition, df_admis, int(id_parcoursup), type_bac)
if not ok:
    st.error(msg)
    st.stop()

base_url = "https://dossier.parcoursup.fr/Candidats/public/fiches/afficherFicheFormation"
formation_url = f"{base_url}?g_ta_cod={int(id_parcoursup)}&typeBac=0&originePc=0"
st.markdown(
    f"🔗 **Lien formation** : "
    f"[Accéder à la fiche Parcoursup]({formation_url})"
)


# =========================
# Bonus
# =========================
bonus_active = (type_bac != "all") and (uai != "")

if type_bac == "all":
    st.info("ℹ️ Pour activer le bonus (rang + lycée), choisis un type de bac spécifique.")

if uai == "":
    st.info("ℹ️ Sélectionne un lycée pour activer le bonus lycée.")

if bonus_active:
    # vérifie cohérence avec ta table lycées utilisée par select_school (année/bac)
    _, err_uai = m.select_school(df_edu, uai, type_bac)
    if err_uai:
        st.warning(
            "⚠️ UAI introuvable pour ce type de bac / année dans la table lycées (edu_bac_stats). "
            "Bonus lycée désactivé."
        )
        bonus_active = False

if bonus_active:
    try:
        bonus_total = m.bonus(df_edu, df_int_edu, uai, rang, type_bac, int(id_parcoursup))
        st.caption(f"✅ Bonus total appliqué : {bonus_total:+.3f} (rang={rang_label}, UAI={uai})")
    except Exception as e:
        st.warning(f"⚠️ Bonus désactivé : {e}")
        bonus_active = False


# =========================
# 1) Courbes
# =========================
st.subheader("1) Courbes : sigmoïdes (avec bonus si applicable)")

try:
    if bonus_active:
        fig = m.plot_comparison_sigmoids_with_bonus(
            df_edu=df_edu,
            df_int_edu=df_int_edu,
            df_proposals=df_proposition,
            df_admis=df_admis,
            uai=uai,
            rang=rang,
            bac_type=type_bac,
            id_parcoursup=int(id_parcoursup),
            note=note,
        )
    else:
        fig = m.plot_comparison_sigmoids(
            df_proposition,
            df_admis,
            int(id_parcoursup),
            type_bac,
            note_origine=note,
            note_avec_bonus=None,
        )

    st.pyplot(fig, clear_figure=True)

except Exception as e:
    st.error(f"Erreur lors du tracé : {e}")


# =========================
# 2) Scores
# =========================
st.divider()
st.subheader("2) Scores par modèle (avec bonus) + spécialités")

cols = st.columns(3)

for (modele, label), col in zip(MODELES, cols):
    with col:
        try:
            if bonus_active:
                res = m.calcul_score_with_bonus(
                    df_edu=df_edu,
                    df_int_edu=df_int_edu,
                    df_proposals=df_proposition,
                    df_admis=df_admis,
                    uai=uai,
                    rang=rang,
                    bac_type=type_bac,
                    id_parcoursup=int(id_parcoursup),
                    note=note,
                    modele=modele,
                )
                bonus_note = m.bonus(df_edu, df_int_edu, uai, rang, type_bac, int(id_parcoursup))
                note_cible = note + bonus_note
            else:
                res = m.calcul_score(
                    df_proposition,
                    df_admis,
                    int(id_parcoursup),
                    type_bac,
                    note,
                    modele,
                )
                note_cible = note

            score = float(res[0])          # 0..1
            fiabilite = float(res[1])      # 0..4

            col_hex = COLORS[modele]

            col.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                  <div style="width:14px;height:14px;border-radius:50%;background:{col_hex};"></div>
                  <div style="font-size:22px;font-weight:700;">{label}</div>
                </div>
                <div style="font-size:54px;font-weight:800;line-height:1.0;margin-bottom:10px;">
                  {100*score:.1f} %
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Fiabilité UNIQUEMENT pour taux_mentions
            if modele == "taux_mentions":
                if np.isfinite(fiabilite):
                    col.markdown(
                        f"<div style='font-size:18px;font-weight:650;margin-bottom:10px;'>"
                        f"Fiabilité : {m.stars(fiabilite)} "
                        f"<span style='opacity:0.7;'>({int(round(fiabilite))}/4)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    col.markdown(
                        "<div style='font-size:18px;font-weight:650;margin-bottom:10px;opacity:0.7;'>"
                        "Fiabilité : n/a</div>",
                        unsafe_allow_html=True,
                    )

            # --- Scores EDS
            df_score_eds = m.calcul_score_with_eds(
                df_proposition,
                df_admis,
                int(id_parcoursup),
                type_bac,
                note_cible,
                modele,
            )

            if df_score_eds is None or df_score_eds.empty:
                col.info("Pas de données EDS pour cette formation / bac.")
            else:
                df_show = (
                    df_score_eds.copy()
                    .assign(
                        eds_slugs=lambda d: d["eds_slugs"].apply(m.format_eds_slugs),
                        score_eds=lambda d: (100*d["score_eds"]).round(1),
                    )
                    .sort_values("score_eds", ascending=False)
                    .head(8)
                )

                col.markdown("#### 🎯 Score par doublette")
                for _, r in df_show.iterrows():
                    col.markdown(
                        m.render_bar(
                            label=str(r["eds_slugs"]),
                            value=float(r["score_eds"]),
                            color=col_hex,
                            bg="#f0f0f0",
                            suffix=" %",
                            height=18,
                        ),
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            col.error(f"Erreur modèle {modele} : {e}")
