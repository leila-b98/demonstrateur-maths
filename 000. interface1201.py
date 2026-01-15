import os
import sys
import importlib.util

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Import du module modèle
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_FILENAME = "module_functions_core_science_1201.py"  # ton fichier model.py renommé
MODULE_PATH = os.path.join(SCRIPT_DIR, MODULE_FILENAME)

if not os.path.exists(MODULE_PATH):
    raise FileNotFoundError(
        f"Impossible de trouver {MODULE_FILENAME} dans le dossier du script.\n"
        f"Script dir: {SCRIPT_DIR}\n"
        f"Fichiers: {os.listdir(SCRIPT_DIR)}"
    )

spec = importlib.util.spec_from_file_location(
    "module_functions_core_science_1201", MODULE_PATH
)
m = importlib.util.module_from_spec(spec)
sys.modules["module_functions_core_science_1201"] = m
spec.loader.exec_module(m)

# =========================
# Remplacement de trapz
# (pas trapz sur streamlit visiblement)
# =========================
import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    # NumPy récent : on recolle trapz sur trapezoid
    np.trapz = np.trapezoid


# =========================
# Config Streamlit
# =========================

st.set_page_config(page_title="Simulateur Parcoursup – Nouveau modèle", layout="wide")
st.title("🎓 Démonstrateur Parcoursup – Nouveau modèle (taux de mention)")

st.link_button(
    "📘 Explications sur le modèle 📘",
    "https://www.canva.com/design/DAG8aLOyIs0/gaIzrqdGvFNsIczoMI4eBg/view?utm_content=DAG8aLOyIs0&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h7c306db998#1",
)



# =========================
# Chargement des données
# =========================

@st.cache_data
def load_data():
    # Tables précalculées pour le nouveau modèle
    df_int_select = pd.read_pickle("df_int_results_taux_bac.pkl")  # stats formations (taux mentions)
    df_int_edu_bac_stats = pd.read_pickle("int_edu_bac_stats.pkl")  # stats formations pour plot distrib
    df_eds = pd.read_pickle("int_edu_bac_stats_eds_computed.pkl")  # spécialités
    df_edu_bac_stat = pd.read_pickle("edu_bac_stats.pkl")  # stats lycées (bonus lycée)

    # Référence UAI -> lycée / académie
    df_uai_ref = pd.read_pickle("df_uai_results_bac.pkl")

    return df_int_select, df_int_edu_bac_stats, df_eds, df_edu_bac_stat, df_uai_ref

(
    df_int_select,
    df_int_edu_bac_stats,
    df_eds,
    df_edu_bac_stat,
    df_uai_ref,
) = load_data()

# branche globale df_eds pour le module
m.df_eds = df_eds


# =========================
# Helpers UI
# =========================

def rgb_tuple_to_hex(rgb):
    r, g, b = [int(255 * x) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


COLORS = {
    "sigmoid": rgb_tuple_to_hex(m.c_2),
    "areas": rgb_tuple_to_hex(m.c_3),
}


def _prepare_uai_reference(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["school_uai", "academy_code", "school_title"]
    df = df.copy()
    df = df[[c for c in cols if c in df.columns]].copy()

    if "school_uai" in df.columns:
        df["school_uai"] = df["school_uai"].astype(str).str.strip().str.upper()
    if "school_title" in df.columns:
        df["school_title"] = df["school_title"].astype(str).str.strip()
    if "academy_code" in df.columns:
        df["academy_code"] = df["academy_code"].astype(str).str.strip()

    df = df.dropna(subset=["school_uai"]).copy()
    df = df[df["school_uai"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["school_uai"]).reset_index(drop=True)
    return df


df_uai_menu = _prepare_uai_reference(df_uai_ref)
UAI_NONE_KEY = "__NONE__"

# Dictionnaire info UAI -> (titre, académie)
_uai_to_info = (
    df_uai_menu.set_index("school_uai")[["school_title", "academy_code"]]
    .to_dict(orient="index")
)

# Tri alphabétique des codes UAI (chaînes mélange lettres/chiffres)
uai_sorted = sorted(df_uai_menu["school_uai"].tolist())
uai_options = [UAI_NONE_KEY] + uai_sorted


def format_uai_option(uai_key: str) -> str:
    if uai_key == UAI_NONE_KEY:
        return "Pas de lycée sélectionné (bonus lycée désactivé)"
    info = _uai_to_info.get(uai_key, {})
    title = info.get("school_title", "")
    acad = info.get("academy_code", "")
    if title and acad:
        return f"{uai_key} — {title} (acad {acad})"
    if title:
        return f"{uai_key} — {title}"
    return str(uai_key)


def get_valid_ids_for_bac(df: pd.DataFrame, bac_type: str):
    mask = df["bac_type"] == bac_type
    ids = sorted(df.loc[mask, "id_parcoursup"].dropna().unique().tolist())
    return ids


def format_eds_slugs_local(x):
    """Petit formatter simple pour afficher les doublettes."""
    if isinstance(x, (list, tuple)):
        return " + ".join(map(str, x))
    return str(x)


# =========================
# Fonctions de plot locales
# =========================

def plot_sigmoid_new_model_local(
    df_1,
    id_parcoursup,
    type_bac,
    note_origine=None,
    note_avec_bonus=None,
):
    """
    Version locale de plot_sigmoid_new_model qui retourne fig pour Streamlit.
    Utilise les fonctions et constantes du module m.
    """

    bins = m.BINS
    x_range = (10, 20)
    n_curve = 800

    row_1 = m.filtre_id(df_1, id_parcoursup, type_bac)
    if row_1 is None:
        raise ValueError(
            f"Aucune ligne trouvée pour id_parcoursup={id_parcoursup}, type_bac={type_bac}"
        )

    school = row_1.get("school_title", "")
    program = row_1.get("program_title", "")
    formation_name = " – ".join(
        [s for s in [school, program] if isinstance(s, str) and len(s) > 0]
    )
    if not formation_name:
        formation_name = f"id_parcoursup {id_parcoursup}"

    # target (%)
    target_1 = np.asarray(m.extract_target_from_row(row_1), dtype=float)

    # nb propositions
    prop_cols = ["none_nb", "ab_nb", "b_nb", "tb_nb", "tbf_nb"]
    nb_props = np.asarray(
        [row_1.get(c, np.nan) for c in prop_cols], dtype=float
    )

    # taux (%)
    taux_cols = ["taux_none", "taux_ab", "taux_b", "taux_tb", "taux_tbf"]
    taux_percent = np.asarray(
        [row_1.get(c, np.nan) for c in taux_cols], dtype=float
    )

    # fit modèle "aires"
    params_fit_df1 = np.array([1.0, 14.0, 2.0, 0.6, 1.0], dtype=float)
    opt_params_df1 = m.fit_generalized_sigmoid_areas(target_1, params_fit_df1)

    xs = np.linspace(*x_range, n_curve)
    y_opt_df1 = m.generalized_sigmoid_Lq(xs, *opt_params_df1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # zones cibles + annotations
    for (x0, x1), ty, n_prop, taux_p in zip(
        bins, target_1 / 100, nb_props, taux_percent
    ):
        ax.fill_between([x0, x1], 0, ty, alpha=0.18)
        ax.hlines(ty, x0, x1, linewidth=2, alpha=0.35)

        x_mid = (x0 + x1) / 2
        y_mid = ty / 2 if np.isfinite(ty) else 0.05

        if np.isfinite(n_prop) and np.isfinite(taux_p) and taux_p > 0:
            taux = taux_p / 100.0
            n_cand = n_prop / taux
            txt = f"t : {int(round(n_prop))}/{int(round(n_cand))}"
        elif np.isfinite(n_prop):
            txt = f"{int(round(n_prop))}/?"
        else:
            txt = "?/?"

        ax.text(x_mid, y_mid, txt, ha="center", va="center", fontsize=8, alpha=0.5)

    color_areas = m.c_3

    ax.plot(
        xs,
        y_opt_df1,
        "-",
        color=color_areas,
        label='Nouveau modèle "aires" — propositions reçues',
        lw=2,
    )

    # points note d'origine (cercles creux)
    legend_points = []
    if note_origine is not None:
        note_o = float(np.clip(note_origine, *x_range))
        y_o = m.generalized_sigmoid_Lq([note_o], *opt_params_df1)[0]
        ax.scatter(
            note_o,
            y_o,
            s=90,
            facecolors="none",
            edgecolors=color_areas,
            linewidths=2,
            zorder=6,
        )
        legend_points.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=color_areas,
                markeredgewidth=2,
                markersize=8,
                label="Note d'origine",
            )
        )

    # points note avec bonus (cercles pleins)
    if note_avec_bonus is not None:
        note_b = float(np.clip(note_avec_bonus, *x_range))
        y_b = m.generalized_sigmoid_Lq([note_b], *opt_params_df1)[0]
        ax.scatter(
            note_b,
            y_b,
            s=90,
            color=color_areas,
            zorder=7,
        )
        legend_points.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color_areas,
                markeredgecolor=color_areas,
                markersize=8,
                label="Note avec bonus",
            )
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + legend_points,
        labels=labels + [lp.get_label() for lp in legend_points],
        title="t : propositions/candidats",
    )

    ax.set_xlim(*x_range)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Note")
    ax.set_ylabel("Probabilité")
    ax.grid(True, linestyle="--", alpha=0.35)

    def truncate(txt, n=90):
        return txt if len(txt) <= n else txt[:n] + "…"

    ax.set_title(
        f"{truncate(formation_name)}\n"
        f"id_parcoursup = {id_parcoursup} | bac = {type_bac}"
    )

    fig.tight_layout()
    return fig


def plot_distribution_school_vs_formation_local(
    df_edu,
    df_int_edu,
    uai,
    id_parcoursup,
    type_bac,
):
    """
    Version locale de plot_distribution_school_vs_formation qui retourne fig.
    """

    row_school, err = m.select_school(df_edu, uai, type_bac)
    if err:
        raise ValueError(err)
    row_form, err = m.select_formation(df_int_edu, id_parcoursup, type_bac)
    if err:
        raise ValueError(err)

    row_school = row_school.iloc[0]
    row_form = row_form.iloc[0]

    mentions = ["fail", "none", "ab", "b", "tb", "tbf"]

    p = np.array(
        [
            row_school["fail_pc"],
            row_school["none_pc"],
            row_school["ab_pc"],
            row_school["b_pc"],
            row_school["tb_pc"],
            row_school["tbf_pc"],
        ],
        dtype=float,
    )
    p = p / p.sum()

    q = np.array(
        [
            row_form["fail_pc"],
            row_form["none_pc"],
            row_form["ab_pc"],
            row_form["b_pc"],
            row_form["tb_pc"],
            row_form["tbf_pc"],
        ],
        dtype=float,
    )
    q = q / q.sum()

    x = np.arange(len(mentions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width / 2, p, width, label="Lycée", alpha=0.7)
    ax.bar(x + width / 2, q, width, label="Formation", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in mentions])
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, max(p.max(), q.max()) * 1.25)

    ax.set_title(
        f"Distribution des mentions\n"
        f"Lycée {uai} vs formation {id_parcoursup} ({type_bac})"
    )

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# =========================
# Sidebar – Entrées
# =========================

with st.sidebar:
    st.header("Entrées")

    # 1) id_parcoursup d'abord (tous les ids existants dans df_int_select)
    # On force les id_parcoursup à être des entiers pour ne pas avoir .0 dans le menu
    ids_raw = df_int_select["id_parcoursup"].dropna().unique().tolist()
    ids_all = sorted({int(x) for x in ids_raw})  # cast en int + tri

    if not ids_all:
        st.error("Aucun id_parcoursup disponible pour df_int_results_taux_bac.")
        st.stop()

    default_id = ids_all[0]
    id_parcoursup = st.selectbox(
        "id_parcoursup",
        ids_all,
        index=ids_all.index(default_id) if default_id in ids_all else 0,
    )

    # 2) type de bac ensuite
    type_bac = st.radio(
        "Type de bac",
        ["general", "techno", "pro"],
        horizontal=True,
    )

    # note au bac et moyenne actuelle
    note_bac = st.slider("Note au bac (sur 20)", 10.0, 20.0, 15.0, 0.1)
    current_average = st.slider("Moyenne actuelle (sur 20)", 10.0, 20.0, 15.0, 0.1)

    # rang
    rang_label = st.radio(
        "Rang dans la classe",
        ["top1", "top5", "autres"],
        horizontal=True,
    )
    rang = None if rang_label == "autres" else rang_label

    # lycée (UAI)
    selected_uai_key = st.selectbox(
        "Lycée d'origine (UAI / nom)",
        options=uai_options,
        format_func=format_uai_option,
        index=0,
    )
    uai = None if selected_uai_key == UAI_NONE_KEY else str(selected_uai_key).strip().upper()

    if uai is not None:
        info = _uai_to_info.get(uai, {})
        st.caption(
            f"🏫 **{info.get('school_title','')}**  \n"
            f"🎓 Académie : **{info.get('academy_code','')}**"
        )

    # doublettes
    st.markdown("### Spécialités (doublettes)")
    # Par défaut :
    # - bac general  -> "Avec doublettes"
    # - bac techno/pro -> "Pas de doublettes"
    default_doublettes_index = 0 if type_bac == "general" else 1

    doublettes_mode = st.radio(
        "Prise en compte des spécialités",
        ["Avec doublettes", "Pas de doublettes"],
        horizontal=True,
    )

    if doublettes_mode == "Avec doublettes":
        spe1 = st.text_input("Spécialité 1 (slug ou texte)", value="maths")
        spe2 = st.text_input("Spécialité 2 (slug ou texte)", value="ses")
        doublettes = [spe1, spe2]
    else:
        st.caption("Les doublettes ne seront pas prises en compte dans le calcul du score.")
        doublettes = None

    run_button = st.button("Calculer le score et afficher les courbes")


# =========================
# Corps – Calcul et affichage
# =========================

if not run_button:
    st.info("🧪 Choisis les paramètres dans la barre latérale puis clique sur *Calculer*.")
    st.stop()

# Vérification que la combinaison (id_parcoursup, type_bac) est bien représentée dans les données
mask_combo = (
    (df_int_select["id_parcoursup"] == int(id_parcoursup))
    & (df_int_select["bac_type"] == type_bac)
)

if not mask_combo.any():
    st.error("type de bac non représenté pour cet id")
    st.stop()

# calcul de la note "de départ"
note_depart = (note_bac + current_average) / 2

try:
    # --- calcul du nouveau score avec bonus (nouvelle interface) ---
    result = m.calcul_score_new_with_bonus(
        df_edu_bac_stat,
        df_int_select,
        uai,
        rang,
        type_bac,
        int(id_parcoursup),
        note_bac,
        current_average,
        doublettes,  # ⬅️ ici on passe soit [spe1, spe2], soit None
    )
except Exception as e:
    st.error(f"Erreur lors du calcul du score : {e}")
    st.stop()

# Récupération des champs de sortie de la nouvelle fonction
score_percent = float(result["score"])
indice = float(result["indice"])
factors = result.get("factors", [])
messages_list = result.get("messages", [])

# On reconstruit note_cible côté app (même logique que dans la fonction)
try:
    bonus_rang_return = m.bonus_rang(rang)
    if uai is None:
        bonus_lycee_return = 0.0
    else:
        bonus_lycee_return = m.bonus_lycee(
            df_edu_bac_stat,
            df_int_select,
            int(id_parcoursup),
            type_bac,
            uai,
        )
except Exception:
    bonus_rang_return = 0.0
    bonus_lycee_return = 0.0

note_cible = note_depart + bonus_rang_return + bonus_lycee_return

# Séparation des facteurs positifs / négatifs
positive_factors = [f for f in factors if f.get("impact") == "positive"]
negative_factors = [f for f in factors if f.get("impact") == "negative"]

# =========================
# Affichage du score
# =========================
base_url = "https://dossier.parcoursup.fr/Candidats/public/fiches/afficherFicheFormation"
formation_url = f"{base_url}?g_ta_cod={int(id_parcoursup)}&typeBac=0&originePc=0"
st.markdown(
    f"🔗 **Lien formation** : "
    f"[Accéder à la fiche Parcoursup]({formation_url})"
)

st.subheader("1) Résultats du nouveau modèle")

col_score, col_info = st.columns([1, 1.4])

with col_score:
    st.markdown(
        f"""
        <div style="font-size:18px;font-weight:600;margin-bottom:6px;">
            Probabilité d'admission (avec bonus)
        </div>
        <div style="font-size:64px;font-weight:800;line-height:1.0;margin-bottom:10px;">
            {score_percent:.1f} %
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-size:18px;font-weight:650;margin-bottom:10px;'>"
        f"Fiabilité : {m.stars(indice)} "
        f"<span style='opacity:0.7;'>({int(round(indice))}/5)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-size:14px;opacity:0.7;'>"
        f"Note de départ : <strong>{note_depart:.2f}/20</strong><br>"
        f"Note cible (avec bonus) : <strong>{note_cible:.2f}/20</strong><br>"
        f"Bonus rang : <strong>{bonus_rang_return:+.2f}</strong> | "
        f"Bonus lycée : <strong>{bonus_lycee_return:+.2f}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_info:
    # Facteurs positifs / négatifs
    st.markdown("#### 🟢 Facteurs positifs")
    if positive_factors:
        for fct in positive_factors:
            title = fct.get("Title", fct.get("key", ""))
            msg = fct.get("Message", "")
            weight = fct.get("weight", 1)
            st.markdown(
                f"- **{title}** ({'⭐' * int(weight)})  \n"
                f"  <span style='opacity:0.8;font-size:13px;'>{msg}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("- *(aucun facteur positif explicite)*")

    st.markdown("#### 🔴 Facteurs négatifs")
    if negative_factors:
        for fct in negative_factors:
            title = fct.get("Title", fct.get("key", ""))
            msg = fct.get("Message", "")
            weight = fct.get("weight", 1)
            st.markdown(
                f"- **{title}** ({'⭐' * int(weight)})  \n"
                f"  <span style='opacity:0.8;font-size:13px;'>{msg}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("- *(aucun facteur négatif explicite)*")

    st.markdown("#### ⚠️ Messages")
    if messages_list:
        for msg in messages_list:
            if isinstance(msg, str):
                st.markdown(f"- `{msg}`")
            else:
                st.markdown(f"- `{msg}`")
    else:
        st.markdown("- *(aucun message particulier)*")


# =========================
# Courbe sigmoïde
# =========================
st.divider()
st.subheader("2) Courbe sigmoïde du modèle (taux de mentions)")

try:
    fig_sigmo = plot_sigmoid_new_model_local(
        df_int_select,
        int(id_parcoursup),
        type_bac,
        note_origine=note_depart,
        note_avec_bonus=note_cible,
    )
    st.pyplot(fig_sigmo, clear_figure=True)
except Exception as e:
    st.error(f"Erreur lors du tracé de la sigmoïde : {e}")


# =========================
# Distribution lycée vs formation
# =========================
st.divider()
st.subheader("3) Distribution des mentions : lycée vs formation")

if uai is None:
    st.info("📊 Distribution lycée vs formation non affichée : aucun lycée sélectionné.")
else:
    try:
        fig_dist = plot_distribution_school_vs_formation_local(
            df_edu_bac_stat,
            df_int_edu_bac_stats,
            uai,
            int(id_parcoursup),
            type_bac,
        )
        st.pyplot(fig_dist, clear_figure=True)
    except Exception as e:
        st.error(f"Erreur lors du tracé des distributions : {e}")


# =========================
# Score par doublette de spécialités
# =========================
st.divider()
st.subheader("4) Score par doublette de spécialités")

try:
    # score "de base" pour la note cible, sans EDS
    base_arr = m.calcul_score_new(
        df_int_select,
        int(id_parcoursup),
        type_bac,
        note_cible,
    )
    base_score = float(base_arr[0])  # entre 0 et 1

    df_eds_sel = m.select_eds(df_eds, int(id_parcoursup), type_bac)

    if df_eds_sel is None or df_eds_sel.empty:
        st.info("Pas de données EDS pour cette formation / type de bac.")
    else:
        df_show = df_eds_sel.copy()
        df_show["eds_label"] = df_show["eds_slugs"].apply(format_eds_slugs_local)
        df_show["score_eds"] = (base_score * df_show["facteur_eds"]).clip(upper=1.0)

        df_show = df_show.sort_values("score_eds", ascending=False)

        st.markdown("Voici le score simulé pour chaque doublette de spécialités :")

        for _, r in df_show.iterrows():
            st.markdown(
                m.render_bar(
                    label=str(r["eds_label"]),
                    value=float(r["score_eds"] * 100),
                    color=COLORS["areas"],
                    bg="#f0f0f0",
                    suffix=" %",
                    height=18,
                ),
                unsafe_allow_html=True,
            )

except Exception as e:
    st.error(f"Erreur lors du calcul ou de l'affichage des scores par spécialités : {e}")
