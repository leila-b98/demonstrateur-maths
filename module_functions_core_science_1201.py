# model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import beta, norm
from scipy.optimize import minimize
from math import sqrt

import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

import ast
import re
import unicodedata


# ==========================
# STYLES ET CONSTANTES
# ==========================

# 3 couleurs contrastées issues de plasma
COLORS_PLASMA = sns.color_palette("plasma", 10)
c_1 = COLORS_PLASMA[8]   # orange vif
c_2 = COLORS_PLASMA[4]   # violet
c_3 = COLORS_PLASMA[1]   # bleu foncé

_TAUX_COLS = ["taux_none", "taux_ab", "taux_b", "taux_tb", "taux_tbf"]
_NB_COLS   = ["none_nb", "ab_nb", "b_nb", "tb_nb", "tbf_nb"]

# Définition des "ranges" de mentions
BINS = [(10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]


def set_plot_style(colormap="mako"):
    sns.set_palette(colormap)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "figure.figsize": (6, 4),
        "savefig.dpi": 600,
        "savefig.format": "png"
    })


set_plot_style("mako")


# ==========================
# FONCTIONS D'AFFICHAGE TEXTE
# ==========================

def stars(n: float) -> str:
    """Affiche une note sur 5 symboles, avec un max de 4 étoiles pleines."""
    try:
        k = int(round(float(n)))
    except Exception:
        k = 0
    k = max(0, min(4, k))           # max = 4 étoiles pleines
    return "⭐" * k + "☆" * (5 - k)  # total = 5 symboles


def render_bar(label, value, color="#cc0000", bg="#f0f0f0", suffix=" %", height=18):
    """
    Génère un HTML de barre horizontale pour Streamlit.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return (
            f"<p style='font-size:14px'><strong>{label}</strong> : n/a</p>"
        )

    v = max(0.0, min(float(value), 100.0))
    return f"""
        <p style='font-size:14px;margin:0 0 6px 0;'>
            <strong>{label}</strong> : {v:.1f}{suffix}
        </p>
        <div style='background-color:{bg};border-radius:999px;height:{height}px;margin-bottom:14px;'>
            <div style='width:{v:.2f}%;background-color:{color};
                        height:100%;border-radius:999px;'></div>
        </div>
    """


# ==========================
# SIGMOÏDE CLASSIQUE
# ==========================

def filtre_id(bac_stats, id_parcoursup, type_bac=None):
    """
    Renvoie la ligne (row) correspondant à (id_parcoursup, type_bac).
    - Sinon : filtre sur bac_type == type_bac.
    """

    if "id_parcoursup" not in bac_stats.columns:
        print("⚠️ Colonne 'id_parcoursup' absente.")
        return None

    subset = bac_stats.loc[bac_stats["id_parcoursup"] == id_parcoursup]
    if subset.empty:
        print(f"⚠️ Aucun enregistrement pour id_parcoursup={id_parcoursup}.")
        return None

    if "bac_type" not in bac_stats.columns:
        print("⚠️ type_bac ignoré : la colonne 'bac_type' n'existe pas.")
    else:
        subset = subset.loc[subset["bac_type"] == type_bac]
        if subset.empty:
            dispo = bac_stats.loc[
                bac_stats["id_parcoursup"] == id_parcoursup, "bac_type"
            ].unique()
            print(
                f"⚠️ Aucun enregistrement pour id_parcoursup={id_parcoursup} "
                f"avec bac_type='{type_bac}'.\n"
                f"Types de bac disponibles pour cet id : {dispo}"
            )
            return None

    # Renvoie la première ligne trouvée
    row = subset.iloc[0]
    return row
    

def plot_generalized_sigmoid(bac_stats, id_parcoursup, type_bac):
    """
    Extrait les valeurs de bac_stats pour un id_parcoursup donné et affiche
    la courbe sigmoïde généralisée correspondante.

    INPUTS : 
    - bac_stats : table précalculée utilisée 
    - id_parcoursup : id_formation (nombre)
    - bac_type : string, 'general', 'techno', 'all', 'etranger', 'autre'
    """

    # # Extraire les valeurs nécessaires
    # row = subset.iloc[0]
    row = filtre_id(bac_stats, id_parcoursup, type_bac)
    
    #modification : extraction des pt1
    pt1 = row["pt1"]
    pt2 = row["pt2"]
    pt3 = row["pt3"]   
    if np.isnan(pt1).any() or np.isnan(pt2).any() or np.isnan(pt3).any():
        print(f"⚠️ Données incomplètes pour id_parcoursup {id_parcoursup}.")
        return

    # Calcul des paramètres
    a, b, c = fit_generalized_sigmoid(pt1, pt2, pt3)

    if a is None or b is None or c is None:
        print(f"⚠️ Impossible d'ajuster une sigmoïde pour {id_parcoursup}.")
        return

    # Génération des valeurs pour tracer la courbe
    x_values = np.linspace(10, 21, 400)
    y_values = generalized_sigmoid(x_values, a, b, c)

    # Tracer la courbe sigmoïde
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="Sigmoïde ajustée", color="b")

    # Placer les points spécifiques
    plt.scatter([pt1[0], pt2[0], pt3[0]], [pt1[1], pt2[1], pt3[1]], color="red", zorder=5, label="Points de référence")

    # Ajouter des annotations pour les points
    for pt, label in zip([pt1, pt2, pt3], ["pt1", "pt2", "pt3"]):
        plt.annotate(label, (pt[0], pt[1]), textcoords="offset points", xytext=(-10, 5), ha='center')

    # Titres et labels
    plt.xlabel("x")
    plt.ylabel("y")
    school = row["school_title"]
    title = row["program_title"]
    plt.title(f"Courbe sigmoïde généralisée pour id_parcoursup {id_parcoursup}: {school} - {title}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Affichage
    plt.show()


def score_admission_sigmo(bac_stats, id_parcoursup, type_bac, note):
    '''
    INPUTS : 
    - bac_stats : table précalculée avec les valeurs des fits 
    - note : moyenne de l'élève
    - id_parcoursup : id de la formation 
    - model : 'sigmo' -> sigmoïde (Modèle original JR) 
    OUTPUTS : 
    - score d'admission en pourcentage
    '''
    # 1) Filtrer sur l'id
    if "id_parcoursup" not in bac_stats.columns:
        print("⚠️ Colonne 'id_parcoursup' absente.")
        return

    subset = bac_stats[bac_stats["id_parcoursup"] == id_parcoursup]
    if subset.empty:
        print(f"⚠️ Aucun enregistrement pour id_parcoursup={id_parcoursup}.")
        return

    # 2) Gérer type_bac :
    # - None  : on ne filtre pas
    # - 'all' : on agrège tous les bacs pour cet id
    # - autre : on filtre sur ce bac_type
    if type_bac is not None and type_bac != "all":
        if "bac_type" not in bac_stats.columns:
            print("⚠️ type_bac ignoré : la colonne 'bac_type' n'existe pas.")
        else:
            subset = subset[subset["bac_type"] == type_bac]
            if subset.empty:
                dispo = bac_stats.loc[
                    bac_stats["id_parcoursup"] == id_parcoursup, "bac_type"
                ].unique()
                print(
                    f"⚠️ Aucun enregistrement pour id_parcoursup={id_parcoursup} "
                    f"avec bac_type='{type_bac}'.\n"
                    f"Types de bac disponibles pour cet id : {dispo}"
                )
                return

    # Extraire les valeurs nécessaires
    row = subset.iloc[0]

    # Extraire les valeurs nécessaires
    pt1 = row["pt1"]
    pt2 = row["pt2"]
    pt3 = row["pt3"]

    if np.isnan(pt1).any() or np.isnan(pt2).any() or np.isnan(pt3).any():
        print(f"⚠️ Données incomplètes pour id_parcoursup {id_parcoursup}.")
        return

    # Calcul des paramètres de la sigmo
    a, b, c = fit_generalized_sigmoid(pt1, pt2, pt3)

    if a is None or b is None or c is None:
        print(f"⚠️ Impossible d'ajuster une sigmoïde pour {id_parcoursup}.")
        return

    #renvoit la valeur de la sigmoïde pour la note de départ, en pourcentage
    return generalized_sigmoid(note, a, b, c)*100


#### TAUX DE MENTION BAC
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def generalized_sigmoid(x, a, b, c):
    """Sigmoïde généralisée permettant un point d'inflexion asymétrique."""
    return 1 / (1 + np.exp(-a * (x - b))) ** c

def fit_generalized_sigmoid(pt1, pt2, pt3):
    """
    Ajuste a et c pour la sigmoïde généralisée qui passe par les points pt1, pt2,
    tout en ayant un point d'inflexion à pt3.

    Paramètres :
    - pt1, pt2, pt3 : arrays contenant [x, y] pour chaque point.

    Retourne :
    - Les paramètres a, b, c de la sigmoïde généralisée.
    """
    Pt_x1, Pt_y1 = pt1
    Pt_x2, Pt_y2 = pt2
    Pt_x3, Pt_y3 = pt3

    if not (0 < Pt_y1 < 1 and 0 < Pt_y2 < 1 and 0 < Pt_y3 < 1):
        return None, None, None  # Évite les valeurs non valides

    # Fixe b = Pt_x3 (point d'inflexion forcé)
    b = Pt_x3

    # Fonction d'erreur à minimiser
    def error_function(params):
        a, c = params
        y1_pred = generalized_sigmoid(Pt_x1, a, b, c)
        y2_pred = generalized_sigmoid(Pt_x2, a, b, c)
        y3_pred = generalized_sigmoid(Pt_x3, a, b, c)
        return (y1_pred - Pt_y1) ** 2 + (y2_pred - Pt_y2) ** 2 + (y3_pred - Pt_y3) ** 2

    # Optimisation de a et c
    res = minimize(error_function, x0=[1, 1], bounds=[(0.01, 10), (0.1, 10)], method="L-BFGS-B")

    if not res.success:
        return None, None, None  # Évite l'utilisation de valeurs incorrectes

    a_opt, c_opt = res.x
    return a_opt, b, c_opt
    

def as_point(pt):
    """
    Convertit pt en np.array([x,y]) float.
    Gère : list/tuple/np.ndarray, pd.Series (1 élément), string "[x, y]".
    Retourne None si parsing impossible / taille != 2 / NaN/inf.
    """
    if pt is None:
        return None

    # Cas: pd.Series (ex: 19052    [x,y]  Name: pt1, dtype: object)
    if isinstance(pt, pd.Series):
        if len(pt) == 0:
            return None
        pt = pt.iloc[0]

    # Cas: string sérialisée "[x, y]"
    if isinstance(pt, str):
        try:
            pt = ast.literal_eval(pt)
        except Exception:
            return None

    # Convertir en array 1D float
    try:
        arr = np.asarray(pt, dtype=float).reshape(-1)
    except Exception:
        return None

    if arr.size != 2:
        return None

    if not np.isfinite(arr).all():
        return None

    return arr

def fit_generalized_sigmoid_row(df, id_parcoursup, type_bac=None):
    """
    Renvoie les paramètres (a, b, c) de la sigmoïde généralisée
    pour une ligne donnée de la base.
    """

    row = filtre_id(df, id_parcoursup, type_bac)
    if row is None:
        return None, None, None

    # Extraction + normalisation des points
    pt1 = as_point(row["pt1"])
    pt2 = as_point(row["pt2"])
    pt3 = as_point(row["pt3"])

    # Données réellement incomplètes = parsing impossible
    if pt1 is None or pt2 is None or pt3 is None:
        return None, None, None

    return fit_generalized_sigmoid(pt1, pt2, pt3)


def generalized_sigmoid_Lq(x, a, b, c, L, q):
    """
    Sigmoïde généralisée avec plafond + paramètre q.

    f(x) = L / (1 + q * exp(-a * (x - b)))^c
    """
    x = np.asarray(x, dtype=float)
    return L / (1.0 + q * np.exp(-a * (x - b)))**c


def param_init(a, b, c):
    """
    Initialise les paramètres (a, b, c, L, q) de la sigmoïde généralisée
    à partir d'une sigmoïde à 3 paramètres.

    Par construction, la sigmoïde 5-paramètres est identique
    à la 3-paramètres au point de départ.
    """
    L0 = 1.0
    q0 = 1.0
    return (a, b, c, L0, q0)

#Définition des "ranges" de mentions
BINS = [(10,12), (12,14), (14,16), (16,18), (18,20)]

def bin_means_trapz(params5, bins=BINS, n_per_bin=200):
    """
    params5 = (L, q, a, b, c)
    Retourne les 5 moyennes de f(x) sur chaque bin, via intégration trapèze.
    """
    a, b, c, L, q = params5
    means = []

    for x0, x1 in bins:
        xs = np.linspace(x0, x1, n_per_bin)
        ys = generalized_sigmoid_Lq(xs, a, b, c, L, q)
        area = np.trapz(ys, xs)
        means.append(area / (x1 - x0))

    return np.asarray(means, dtype=float)

def extract_target_from_row(row, as_percent=True):
    """
    Extrait la target [taux_p, taux_ab, taux_b, taux_tb, taux_tbf] depuis une row.

    Retourne toujours un np.array de shape (5,)
    """
    cols = ["taux_none", "taux_ab", "taux_b", "taux_tb", "taux_tbf"]

    try:
        # -> force array 1D (5,)
        target = np.asarray([row[c] for c in cols], dtype=float).reshape(-1)
    except KeyError as e:
        raise KeyError(f"Colonne manquante pour la target : {e}")

    if target.size != len(cols):
        raise ValueError(f"Target mal formée : shape={target.shape}, attendu ({len(cols)},)")

    if as_percent:
        return target
    else:
        return target / 100.0

from scipy.optimize import minimize


def fit_generalized_sigmoid_areas(
    target_means_percent,      # [taux_p, taux_ab, taux_b, taux_tb, taux_tbf] en %
    init_params5,              # (L0,q0,a0,b0,c0)
    bins=BINS,
    n_per_bin=200,
    weights=None
):
    """
    Ajuste (L,q,a,b,c) pour que les moyennes par bin collent aux taux (en %).
    Philosophie identique à fit_generalized_sigmoid : error_function + minimize + bounds.
    """

    # 1) target en [0,1]
    target = np.asarray(target_means_percent, dtype=float) / 100.0
    if target.shape != (len(bins),):
        print(f"⚠️ target_means_percent doit être de taille {len(bins)}")
        return None, None, None, None, None

    # sécurité
    if np.any(~np.isfinite(target)) or np.any(target < 0) or np.any(target > 1):
        return None, None, None, None, None

    # poids
    if weights is None:
        weights = np.ones_like(target, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # init
    x0 = np.asarray(init_params5, dtype=float)
    if x0.shape != (5,) or np.any(~np.isfinite(x0)):
        return None, None, None, None, None

    # bornes (L,q,a,b,c)
    bounds = [
        (1e-4, 50.0),   # a
        (8.0, 22.0),    # b
        (1e-3, 50.0),   # c
        (0.0, 1.0),     # L
        (1e-6, 1e3),    # q
    ]

    # Fonction d'erreur à minimiser (SSE pondérée sur les 5 bins)
    def error_function(p):
        a, b, c, L, q = p
        pred = bin_means_trapz((a, b, c, L, q), bins=bins, n_per_bin=n_per_bin)
        r = pred - target
        return np.sum((weights * r) ** 2)

    res = minimize(error_function, x0=x0, bounds=bounds, method="L-BFGS-B")

    if not res.success:
        return None, None, None, None, None

    a_opt, b_opt, c_opt, L_opt, q_opt = res.x
    return a_opt, b_opt, c_opt, L_opt, q_opt


#VERSION 2 TEST : À AMÉLIORER POUR RESPECTER LA CROISSANCE DES BINS
def fit_generalized_sigmoid_areas(
    target_means_percent,      # [taux_p, taux_ab, taux_b, taux_tb, taux_tbf] en %
    init_params5,              # (a0, b0, c0, L0, q0) point de départ
    bins=BINS,
    n_per_bin=200,
    weights=None,              # gardé pour compatibilité mais ignoré
):
    """
    Ajuste (a, b, c, L, q) pour que les moyennes par bin collent aux taux (en %).
    Les NaN dans target_means_percent sont ignorés.
    """

    # cible en [0,1]
    target_full = np.asarray(target_means_percent, dtype=float) / 100.0

    # masque: on garde seulement les bins pour lesquels la cible est définie
    valid_mask = np.isfinite(target_full)
    target = target_full[valid_mask]

    # bins correspondants
    bins_valid = [b for b, m in zip(bins, valid_mask) if m]

    # paramètres initiaux
    x0 = np.asarray(init_params5, dtype=float)

    # bornes (a, b, c, L, q)
    bounds = [
        (1e-4, 50.0),   # a
        (8.0, 22.0),    # b
        (1e-3, 50.0),   # c
        (0.0, 1.0),     # L
        (1e-6, 1e3),    # q
    ]

    # SSE sur les seuls bins valides
    def error_function(p):
        a, b, c, L, q = p
        pred = bin_means_trapz((a, b, c, L, q), bins=bins_valid, n_per_bin=n_per_bin)
        r = pred - target
        return np.sum(r**2)

    res = minimize(error_function, x0=x0, bounds=bounds, method="L-BFGS-B")

    if not res.success:
        return None, None, None, None, None

    a_opt, b_opt, c_opt, L_opt, q_opt = res.x
    return a_opt, b_opt, c_opt, L_opt, q_opt


def fit_generalized_sigmoid_means(df, id_parcoursup, type_bac):
    """
    Fonction qui fait le fit pour une formation donnée.
    """
    #On sélectionne le row
    row = filtre_id(df, id_parcoursup, type_bac)

    #On définit les targets qui sont les moyennes
    target_mean = extract_target_from_row(row)
    
    #On définit les paramètres initiaux
    # [a0, b0, c0] = fit_generalized_sigmoid_row(df, id_parcoursup, type_bac)
    
    # Ici on change et on part de paramètres standards (au lieu de partir de la sigmoïde précédente comme avant)
    # init_param = param_init(a0, b0, c0)
    init_params5 = np.array([1.0, 14.0, 2.0, 0.6, 1.0], dtype=float)

    return fit_generalized_sigmoid_areas(target_mean, init_params5)


# FIABILITE
# Je n'arrive pas à importer les formules toutes faites de python
def wilson_ci(k, n, alpha=0.05):
    """
    Intervalle de confiance de Wilson pour une proportion binomiale.
    Renvoie (lo, hi) dans [0,1].
    """
    if n <= 0:
        return (np.nan, np.nan)

    z = norm.ppf(1 - alpha/2)
    phat = k / n

    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    radius = z * sqrt(
        (phat*(1 - phat) + z**2 / (4*n)) / n
    ) / denom

    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return (lo, hi)


def mention_bin_index(note, bins=BINS):
    """
    Détermine l'indice du bin de mention correspondant à une note donnée.

    INPUTS :
    - note : float

    - bins : list of tuple, optionnel
        Liste des intervalles de notes (borne_inf, borne_sup).

    OUTPUTS :
    int ou None
        Indice du bin contenant la note, ou None si la note est hors des bornes.
    """
    x = float(note)

    for i, (x0, x1) in enumerate(bins):
        # Tous les bins sont [x0, x1[, sauf le dernier qui inclut la borne sup
        if (x >= x0 and x < x1) or (i == len(bins) - 1 and x >= x0 and x <= x1):
            return i

    return None


def ci_amplitude_wilson_for_note(row, note, alpha=0.05):
    """
    Calcule l'amplitude de l'intervalle de confiance (IC) de Wilson
    pour la probabilité d'admission associée à la mention correspondant
    à une note donnée.

    INPUTS :
    row : pandas.Series
        Ligne issue de df_proposition correspondant à une formation
        et un type de bac donnés.

    note : float
        Note du candidat (sur 20).

    alpha : float, optionnel
        Niveau de risque de l'IC (0.05 -> 95 %).

    OUTPUTS :
    float
        Amplitude de l'intervalle de confiance de Wilson pour le bin
        correspondant à la note.

        Renvoie np.nan si l'IC ne peut pas être calculé
        (données manquantes, taux nul, effectif invalide).
    """
    # 1) Identifier le bin de mention
    i = mention_bin_index(note)
    if i is None:
        return np.nan

    # 2) Récupérer taux (%) et nombre de succès
    taux = row.get(_TAUX_COLS[i], np.nan)
    N = row.get(_NB_COLS[i], np.nan)

    if not np.isfinite(taux) or not np.isfinite(N):
        return np.nan

    p_hat = float(taux) / 100.0
    N = int(round(N))

    # 3) Vérifications de cohérence
    if p_hat < 0 or p_hat > 1 or N < 0:
        return np.nan

    # Cas p̂ = 0 : impossible de reconstruire n sans info externe
    if p_hat == 0:
        return np.nan

    # 4) Reconstruction du nombre total de candidats
    n = int(round(N / p_hat))
    if n <= 0:
        return np.nan

    N = min(N, n)

    # 5) Calcul de l'IC de Wilson
    lo, hi = wilson_ci(N, n, alpha=alpha)

    # 6) Amplitude de l'intervalle
    return float(hi - lo)


def compute_fiabilite_score(row, note, alpha=0.05):
    """
    Calcule l'amplitude de l'IC de Wilson et en déduit un score de fiabilité entre 0 et 4.
    Retourne (score_fiabilite, amp_ci).
    """
    amp_ci = ci_amplitude_wilson_for_note(row, note, alpha=alpha)

    if not np.isfinite(amp_ci):
        score_fiabilite = 0
    else:
        score_fiabilite = int(np.round(4 * (1 - amp_ci)))

    return score_fiabilite, amp_ci



# SCORE DE BASE 

def get_row_or_raise(df, id_parcoursup, type_bac):
    """Récupère la ligne correspondant à (id_parcoursup, type_bac) ou lève une erreur."""
    row = filtre_id(df, id_parcoursup, type_bac)
    if row is None:
        raise ValueError(
            f"Aucune ligne trouvée dans df pour id_parcoursup={id_parcoursup}, type_bac={type_bac}"
        )
    return row


def get_target_means_percent(row):
    """
    Extrait la target [taux_none, taux_ab, taux_b, taux_tb, taux_tbf] en %,
    sous forme de np.array de shape (5,).
    """
    target_means_percent = np.asarray(
        extract_target_from_row(row),
        dtype=float
    ).reshape(-1)
    return target_means_percent


def init_params5_from_target(target_means_percent):
    """
    Initialise (a0, b0, c0, L0, q0) à partir de la target.
    L0 est pris comme le max observé (en [0,1]) si possible, sinon 0.6.
    """
    target01 = target_means_percent / 100.0
    if np.isfinite(target01).any():
        L0 = float(np.nanmax(target01))
    else:
        L0 = 0.6

    a0 = 1.0
    b0 = 14.0
    c0 = 2.0
    q0 = 1.0

    return np.array([a0, b0, c0, L0, q0], dtype=float)


def fit_sigmoid_params_from_target(target_means_percent):
    """
    Fait le fit des paramètres (a, b, c, L, q) à partir de la target en %.
    Retourne opt_params5 = (a_opt, b_opt, c_opt, L_opt, q_opt).
    """
    init_params5 = init_params5_from_target(target_means_percent)
    opt_params5 = fit_generalized_sigmoid_areas(target_means_percent, init_params5)
    return opt_params5


def compute_score_from_params(note, opt_params5):
    """
    Calcule le score à partir des paramètres optimisés.
    Si le fit a échoué (opt_params5[0] est None), retourne NaN.
    """
    if opt_params5[0] is None:
        return np.nan
    return generalized_sigmoid_Lq(note, *opt_params5)



def calcul_score_new(df_proposition, id_parcoursup, type_bac, note):
    """
    Calcule le score de probabilité d'admission pour un candidat donné.
    Par défaut, utilise le modèle 'taux_mentions'.
    
    Retourne: np.array([score, score_fiabilite])

    C'est la fonction de base qui calcule le fit de la sigmoïde et le score de "départ".
    """
    alpha = 0.05

    # 1) Récupération de la ligne
    row = filtre_id(df_proposition, id_parcoursup, type_bac)

    # 2) Target en %
    target_means_percent = get_target_means_percent(row)

    # 3) Fit des paramètres de la sigmoïde
    opt_params5 = fit_sigmoid_params_from_target(target_means_percent)

    # 4) Score pour la note donnée
    score = compute_score_from_params(note, opt_params5)

    # 5) Score de fiabilité
    score_fiabilite, amp_ci = compute_fiabilite_score(row, note, alpha=alpha)

    # 6) Clip du score entre 0 et 1
    score = float(np.clip(score, 0.0, 1.0))

    return np.array([score, score_fiabilite], dtype=float)



# BONUS 

def bonus_rang(rang): 
    '''
    INPUT : 
    - 'top1', 'top5', None 

    OUTPUT : 
    float : le bonus à afficher sur la note, pour ensuite le mettre dans la fonction qui calcule le résultat à propos de la sigmoïde'''

    if rang == 'top1': 
        return 1
    
    elif rang == 'top5':
        return 0.2

    else :
        return 0


# Définition des calculs de distances et divergences

# Kullback-Leibler divergence
def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# Jensen–Shannon divergence
def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Cumulative difference of probability distributions
def cumdiff(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sum((np.cumsum(q) - np.cumsum(p)))
    
# Normalize distance to a 0-100 scale
def calculate_distance100(distance, min_dist, max_dist):
    if distance is None or pd.isna(distance):
        return None
    elif distance < 0:
        distance100 = int(50 - 50 * distance / min_dist)
        distance100 = max(distance100, 0)
        return distance100
    elif distance > 0:
        distance100 = int(50 + 50 * distance / max_dist)
        distance100 = min(distance100, 100)
        return distance100
    elif distance == 0:
        return 50


def select_school(df_edu, uai, type_bac):
    '''INPUTS : 
    - df_edu : contient toutes les stats des lycées df
    - uai : code du lycée string
    - type_bac : type de bac (général, pro, techno) string

    OUTPUT : 
    - df : dataframe contenant les stats du lycée (1 seule ligne)
    - error : message d'erreur si le lycée n'est pas trouvé
    '''

    df_school = df_edu.query(
        "stats_type == 'results' and year == 2023"
    )

    result = df_school.query(
        'uai == @uai and bac_type == @type_bac'
    )

    if result.empty:
        return None, "100-uai-and-bac-not-found"

    # on garde la ligne la plus récente (updated_at diffère)
    result = (
        result
        .sort_values("updated_at", ascending=False)
        .head(1)
    )

    return result, None


def select_formation(df_int_edu, id_parcoursup, type_bac):
    '''INPUTS : 
    - df_edu : contient toutes les stats des formations df (int_edu_bac_stats)
    - uai : code du lycée string
    - type_bac : type de bac (général, pro, techno) string

    OUTPUT : 
    - df : dataframe contenant les stats de la formation
    - error : message d'erreur si le lycée n'est pas trouvé'''

    df_school = df_int_edu.query("stats_type == 'intake'")

    result = df_school.query(
        " id_parcoursup == @id_parcoursup and bac_type == @type_bac and intake_stage == 'candidates' "
    )

    if result.empty:
        return None, f"101-formation-and-type-bac-not-found"

    return result, None


def bonus_lycees_all(df_edu, df_int_edu, id_parcoursup, type_bac):
    """
    Pour une formation cible, calcule la distance normalisée (-1 -> 1) pour tous les lycées. 
    
    INPUTS :
    - df_edu : DataFrame avec les stats des lycées
    - df_int_edu : DataFrame avec les stats des formations
    - id_parcoursup : id de la formation visée
    - type_bac : type de bac (général, pro, techno)
    
    OUTPUT :
    - DataFrame : colonnes ['uai', 'distance', 'distance100']
    """
    
    # Récupère la formation cible
    row_formation, err_form = select_formation(df_int_edu, id_parcoursup, type_bac)
    if err_form:
        raise ValueError(err_form)
    
    row_formation = row_formation.iloc[0]
    
    # Filtre les lycées pour ce type de bac
    df_lycees = df_edu.query("stats_type == 'results' and year == 2023 and bac_type == @type_bac")
    
    # Ne garder que la ligne la plus récente par lycée
    df_lycees = df_lycees.sort_values("updated_at", ascending=False).drop_duplicates(subset=["uai"])
    
    results = []
    
    # Calcul des distances pour chaque lycée
    for _, row_lycee in df_lycees.iterrows():
        p = np.array([
            row_lycee['fail_pc'], row_lycee['none_pc'], row_lycee['ab_pc'],
            row_lycee['b_pc'], row_lycee['tb_pc'], row_lycee['tbf_pc']
        ], dtype=float)
        p = p / p.sum()
        
        q = np.array([
            row_formation['fail_pc'], row_formation['none_pc'], row_formation['ab_pc'],
            row_formation['b_pc'], row_formation['tb_pc'], row_formation['tbf_pc']
        ], dtype=float)
        q = q / q.sum()
        
        js = js_divergence(p, q)
        cd = cumdiff(p, q)
        
        # distance signée
        distance = abs(js) * cd / abs(cd) if cd != 0 else 0
        results.append({"uai": row_lycee["uai"], "distance": distance})
    
    df_results = pd.DataFrame(results)
    
    # Normalisation dans [-1, 1]
    min_dist = df_results["distance"].min()
    max_dist = df_results["distance"].max()
    
    def normalize_signed(dist):
        if dist > 0 and max_dist > 0:
            return dist / max_dist
        elif dist < 0 and min_dist < 0:
            return dist / abs(min_dist)
        else:
            return 0.0
    
    df_results["distance_norm"] = df_results["distance"].apply(normalize_signed)
    df_results["distance_norm"] = df_results["distance_norm"] +1
    return df_results


def bonus_lycee(df_edu, df_int_edu, id_parcoursup, type_bac, uai):
    """
    Calcule la distance normalisée dans l’intervalle [-1, 1] pour un lycée donné,
    relativement à une formation cible.

    Principe :
    - On compare la distribution des mentions du lycée à celle de la formation
      via une divergence de Jensen–Shannon.
    - Le signe est déterminé par les distributions cumulées (profil globalement
      plus favorable ou moins favorable que la référence).
    - Les distances sont ensuite normalisées sur l’ensemble des lycées afin
      d’obtenir un score centré (0 = neutre) et comparable entre formations.

    INPUTS :
    - df_edu : DataFrame contenant les statistiques des lycées
    - df_int_edu : DataFrame contenant les statistiques des formations
    - id_parcoursup : identifiant de la formation cible
    - type_bac : type de baccalauréat (général, pro, techno)
    - uai : identifiant UAI du lycée

    OUTPUT :
    - float : distance normalisée dans [-1, 1]
    """
    if uai == None : 
        return 0
    else :
        df_all = bonus_lycees_all(df_edu, df_int_edu, id_parcoursup, type_bac)

        row = df_all.loc[df_all["uai"] == uai]
        if row.empty:
            raise ValueError(f"UAI de ce lycée introuvable dans les données : {uai}")

    return float(row["distance_norm"].iloc[0])


import numpy as np
import re
import unicodedata

def select_eds(df, id_parcoursup, bac_type):
    return df.loc[
        (df["id_parcoursup"] == id_parcoursup) &
        (df["bac_type"] == bac_type),
        ["eds", "eds_slugs","proba_proposal", "facteur_eds"]
    ]


def eds_bonus_slugs(df, id_parcoursup, bac_type, eds1, eds2):
    """
    Renvoie facteur_eds pour la doublette (eds1, eds2) en se basant sur eds_slugs.
    eds1/eds2 peuvent être des slugs OU des mots (ex: "Maths", "Physique chimie", "LLCER", ...),
    ordre indifférent.

    Règles :
    - Si proba_proposal == 0 -> "Aucun candidat avec tes spécialités n'a été admis"
    - Si doublette absente -> "Pas d'information sur tes doublettes de spécialité"
    - Si eds1/eds2 non reconnues -> message dédié
    """

    def normalize_text(s):
        """lower + sans accents + alphanum/espaces/- only + espaces normalisés"""
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.replace("_", "-")
        s = re.sub(r"[^a-z0-9\s\-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def slug_candidates_from_df(df_sub):
        """récupère l'ensemble des slugs présents dans eds_slugs (doublettes uniquement)"""
        slugs = set()
        for x in df_sub["eds_slugs"].dropna():
            if isinstance(x, (list, tuple)) and len(x) == 2:
                slugs.add(str(x[0]))
                slugs.add(str(x[1]))
        return slugs

    # --- sous-table pertinente (programme + bac) ---
    df_sub = df.loc[
        (df["id_parcoursup"] == id_parcoursup) &
        (df["bac_type"] == bac_type),
        ["eds_slugs", "proba_proposal", "facteur_eds"]
    ].copy()

    if df_sub.empty:
        return "Pas d'information sur tes doublettes de spécialité"

    # --- garder uniquement les vraies doublettes ---
    mask = df_sub["eds_slugs"].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 2)
    df_sub = df_sub.loc[mask].copy()

    if df_sub.empty:
        return "Pas d'information sur tes doublettes de spécialité"

    # --- dictionnaire de correspondance "texte normalisé" -> "slug" ---
    # On construit depuis les slugs réellement présents (donc robuste par programme/bac)
    slugs_set = slug_candidates_from_df(df_sub)
    slug_map = {normalize_text(slug): slug for slug in slugs_set}

    # mini-synonymes utiles (tu peux en ajouter)
    # clé = texte normalisé possible, valeur = texte normalisé de slug existant ou slug direct
    synonyms = {
        "maths": "maths",
        "mathematiques": "maths",
        "mathematique": "maths",
        "physique chimie": "pc",
        "physique-chimie": "pc",
        "pc": "pc",
        "svt": "svt",
        "sciences de la vie et de la terre": "svt",
        "hlp": "hlp",
        "histoire geo geopolitique sciences politiques": "hggsp",
        "hggsp": "hggsp",
        "ses": "ses",
        "llcer": "llcer",
        "cinema audiovisuel": "cinema-audiovisuel",
        "cinema-audiovisuel": "cinema-audiovisuel",
    }

    def resolve_to_slug(user_input):
        """essaie de transformer un input user (slug ou mots) en slug présent dans df_sub"""
        u = normalize_text(user_input)
        if not u:
            return None

        # 1) match exact sur slug normalisé
        if u in slug_map:
            return slug_map[u]

        # 2) via synonymes (si dispo) -> on retente
        if u in synonyms:
            u2 = normalize_text(synonyms[u])
            if u2 in slug_map:
                return slug_map[u2]
            # parfois synonyms[u] est déjà le slug exact
            if synonyms[u] in slugs_set:
                return synonyms[u]

        # 3) match "contient" (ex: user tape "cinema" et slug est "cinema-audiovisuel")
        # On prend le slug le plus court qui matche (souvent le plus pertinent)
        candidates = [slug for slug in slugs_set if u in normalize_text(slug)]
        if candidates:
            candidates.sort(key=lambda s: len(s))
            return candidates[0]

        return None

    s1 = resolve_to_slug(eds1)
    s2 = resolve_to_slug(eds2)

    if s1 is None or s2 is None:
        return "Pas d'information sur tes doublettes de spécialité"

    # --- clé hashable pour comparer (ordre indifférent) ---
    target = tuple(sorted([s1, s2]))
    df_sub["eds_key"] = df_sub["eds_slugs"].apply(lambda x: tuple(sorted([str(x[0]), str(x[1])])))

    match = df_sub.loc[df_sub["eds_key"] == target]
    if match.empty:
        return "Pas d'information sur tes doublettes de spécialité"

    proba = match["proba_proposal"].iloc[0]
    facteur = match["facteur_eds"].iloc[0]

    if not np.isfinite(proba) or not np.isfinite(facteur):
        return "Pas d'information sur tes doublettes de spécialité"

    if proba == 0:
        return "Aucun candidat avec tes spécialités n'a été admis"

    return float(facteur)

def calcul_score_with_eds(df_proposition, df_admis, id_parcoursup, type_bac, note, modele):
    '''OUTPUT : 
    tableau avec le score par doublette de spécialité, capé à 1

    INPUTS :       
    Pour le modèle :
        - 'prop_recues'
      - 'prop_acceptees'
      - 'taux_mentions '''
    
    s = calcul_score(df_proposition, df_admis, id_parcoursup, type_bac, note, modele)
    df = select_eds(df_eds, id_parcoursup, type_bac)
    return None if df is None or df.empty else (
        df.assign(
            eds_slugs=df["eds_slugs"].where(df["eds_slugs"].notna(), "Autres doublettes"),
            score_eds=(s * df["facteur_eds"]).clip(upper=1).round(2)
        )[["eds_slugs", "score_eds"]]
    )


# SCORE FINAL

def calcul_score_new_with_bonus(df_edu, df_int_edu, uai, rang, bac_type, id_parcoursup, note_bac, current_average, doublettes):
    """
    Renvoie un score (float entre 0 et 1) évalué à la note (float),

    INPUTS : 
    df_edu : DF contient les résultats du bac
    df_int_edu : DF contient les résultats des propositions recues
    uai : string lycée d'origine
    rang : string 'top1' ou 'top5' ou None
    bac_type : string 'general' / 'techno' / 'pro'
    id_parcoursup : int identifiant de la formation
    note_bac : int note au bac
    current_average : int
    doublettes : array de deux strings avec les noms "réduits" ['maths', 'ses']

    OUTPUTS : (modification 1301 sur le style des outputs)
    [0] : score (float)
    [1] : bonus (float)
    [2] : factors (string)
    [3] : Messages 


    """

    # Initialisation des facteurs
    factors = []
    messages = []

    #Calcul de la note de départ 
    note = (note_bac + current_average)/2

    # Calcul du bonus
    #Rang (uniquement un bonus)
    bonus_rang_return = bonus_rang(rang)
    if bonus_rang_return ==1  : 
        factors.append({ "key" : "rank", "impact" : "positive", "weight" : 3, "Title" : "Ton rang dans la classe", "Message" : "XXX"})
    elif bonus_rang_return == 0.2 : 
        factors.append({ "key" : "rank", "impact" : "positive", "weight" : 1, "Title" : "Ton rang dans la classe", "Message" : "XXX"})

    #Lycée (Uniquement un bonus)
    bonus_lycee_return = bonus_lycee(df_edu, df_int_edu,id_parcoursup, bac_type, uai)
    if bonus_lycee_return >= 1.66 : 
        factors.append({ "key" : "highschool-level", "impact" : "positive", "weight" : 3, "Title" : "Le niveau de ton lycée", "Message" : "XXX"})
    elif bonus_lycee_return >= 1.33 :
        factors.append({ "key" : "highschool-level", "impact" : "positive", "weight" : 2, "Title" : "Le niveau de ton lycée", "Message" : "XXX"})
    elif bonus_lycee_return >= 1 :
        factors.append({ "key" : "highschool-level", "impact" : "positive", "weight" : 1, "Title" : "Le niveau de ton lycée", "Message" : "XXX"})

    #Bonus total et calcul de la note cible
    bonus_note = bonus_rang_return + bonus_lycee_return
    note_cible = note + bonus_note

    #Calcul du score
    score = calcul_score_new(df_int_edu, id_parcoursup, bac_type, note_cible) 
    resultat = score[0]
    fiabilite = score[1]

    #Prise en compte des doublettes 
    # on extrait les spés
    spe_1, spe_2 = doublettes
    df = select_eds(df_eds, id_parcoursup, bac_type)     #sélection du tableau avec les bonus des doublettes
    mask_doublette = df['eds_slugs'].apply(lambda x: isinstance(x, list) and set(x) == {spe_1, spe_2})
    df_select = df[mask_doublette]
    if not df_select.empty:
        result = df_select.iloc[0]
        bonus_doublettes = result['facteur_eds']

    else:
        # Si la doublette n'est pas trouvée, on regarde la ligne "Autres doublettes"
        mask_autres = df['eds'].apply(lambda x: isinstance(x, list) and 'Autres doublettes' in x)
        df_autres = df[mask_autres]
        if not df_autres.empty:
            result = df_autres.iloc[0]
            bonus_doublettes = result['facteur_eds']

        else:
        #Si la doublette n'est pas du tout représentée
            result = None
            bonus_doublettes = 0
            messages.append('010-doublettes-info-no-match')

    resultat = resultat * bonus_doublettes

     #Explication des bonus à améliorer car plusieurs choix possibles pour les spés.
    if bonus_doublettes >= 1 :
        factors.append({ "key" : "specialties", "impact" : "positive", "weight" : 1, "Title" : "Ton choix de spécialités", "Message" : "XXX"})
    elif bonus_doublettes < 1 and bonus_doublettes > 0 :
        factors.append({ "key" : "specialties", "impact" : "negative", "weight" : 1, "Title" : "Ton choix de spécialités", "Message" : "XXX"})


    #Structure de sortie
    result_dict = {
        "score": min(resultat * 100, 100),       # Score final / 100, maximum 100
        "indice": fiabilite,        # Fiabilité / 5 (jamais plus de 4)
        "factors": factors,
        "messages": messages
            }

    return result_dict


# TESTS DES FONCTIONS ET PLOTS 

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm

import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style(colormap="mako"):
    sns.set_palette(colormap)
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],   # Serif disponible dans Deepnote
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "figure.figsize": (6, 4),
        "savefig.dpi": 600,
        "savefig.format": "png"
    })

set_plot_style("mako")

# 3 couleurs contrastées issues de plasma
COLORS_PLASMA = sns.color_palette("plasma", 10)

c_1 = COLORS_PLASMA[8]   # orange vif
c_2 = COLORS_PLASMA[4]   # violet
c_3    = COLORS_PLASMA[1]   # bleu foncé


#FONCTION 2 avec erreur
def plot_sigmoid_new_model(
    df_1, id_parcoursup, type_bac,
    note_origine=None, note_avec_bonus=None
):

    bins = BINS
    x_range = (10, 20)
    n_curve = 800

    # --- FILTRAGE
    row_1 = filtre_id(df_1, id_parcoursup, type_bac)

    school = row_1.get("school_title", "")
    program = row_1.get("program_title", "")
    formation_name = " – ".join(
        [s for s in [school, program] if isinstance(s, str) and len(s) > 0]
    )
    if not formation_name:
        formation_name = f"id_parcoursup {id_parcoursup}"

    # --- TARGETS (%)
    target_1 = np.asarray(extract_target_from_row(row_1), dtype=float)

    # --- Numérateur : propositions
    prop_cols = ["none_nb", "ab_nb", "b_nb", "tb_nb", "tbf_nb"]
    nb_props = np.asarray([row_1.get(c, np.nan) for c in prop_cols], dtype=float)

    # --- Taux (%)
    taux_cols = ["taux_none", "taux_ab", "taux_b", "taux_tb", "taux_tbf"]
    taux_percent = np.asarray([row_1.get(c, np.nan) for c in taux_cols], dtype=float)

    # --- AIRES MODÈLE
    params_fit_df1 = np.array([1.0, 14.0, 2.0, 0.6, 1.0], dtype=float)
    opt_params_df1 = fit_generalized_sigmoid_areas(target_1, params_fit_df1)

    # --- COURBES
    xs = np.linspace(*x_range, n_curve)
    y_opt_df1 = generalized_sigmoid_Lq(xs, *opt_params_df1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- ZONES CIBLES + annotations
    for (x0, x1), ty, n_prop, taux_p in zip(bins, target_1 / 100, nb_props, taux_percent):
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

    # --- COULEURS (celles définies ensemble)
    color_fit = c_2
    color_areas = c_3

    # --- COURBES
    ax.plot(xs, y_opt_df1, "-", color=color_areas,
            label='Nouveau modèle "aires" — propositions reçues', lw=2)

    # --- POINTS : note d'origine (cercles creux, couleur = courbe)
    if note_origine is not None:
        note_o = float(np.clip(note_origine, *x_range))
        for y, col in [
            (generalized_sigmoid_Lq([note_o], *opt_params_df1)[0], color_areas),
        ]:
            ax.scatter(
                note_o, y,
                s=90,
                facecolors="none",
                edgecolors=col,
                linewidths=2,
                zorder=6
            )

    # --- POINTS : note avec bonus (cercles pleins, couleur = courbe)
    if note_avec_bonus is not None:
        note_b = float(np.clip(note_avec_bonus, *x_range))
        for y, col in [
            (generalized_sigmoid_Lq([note_b], *opt_params_df1)[0], color_areas),
        ]:
            ax.scatter(
                note_b, y,
                s=90,
                color=col,
                zorder=7
            )

    # --- LÉGENDE : points (UNE couleur, choisie parmi celles du modèle)
    legend_points = []
    if note_origine is not None:
        legend_points.append(
            Line2D(
                [0], [0],
                marker='o',
                color='none',
                markerfacecolor='none',
                markeredgecolor=color_areas,
                markeredgewidth=2,
                markersize=8,
                label="Note d'origine"
            )
        )
    if note_avec_bonus is not None:
        legend_points.append(
            Line2D(
                [0], [0],
                marker='o',
                color='none',
                markerfacecolor=color_areas,
                markeredgecolor=color_areas,
                markersize=8,
                label="Note avec bonus"
            )
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + legend_points,
        labels=labels + [lp.get_label() for lp in legend_points],
        title="t : propositions/candidats"
    )

    # --- AXES / TITRE
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

    plt.show()


def plot_distribution_school_vs_formation(
    df_edu, df_int_edu,
    uai, id_parcoursup, type_bac
):
    """
    Trace sur un même graphique :
    - la distribution des mentions du lycée
    - la distribution des mentions de la formation

    Axe x : mentions (ordonnées)
    Axe y : proportion (0–1)

    Objectif : visualisation / tests pour comprendre la JS divergence
    """

    # --- Sélection des données
    row_school, err = select_school(df_edu, uai, type_bac)
    if err:
        raise ValueError(err)

    row_form, err = select_formation(df_int_edu, id_parcoursup, type_bac)
    if err:
        raise ValueError(err)

    row_school = row_school.iloc[0]
    row_form = row_form.iloc[0]

    # --- Mentions et distributions
    mentions = ["fail", "none", "ab", "b", "tb", "tbf"]

    p = np.array([
        row_school["fail_pc"], row_school["none_pc"], row_school["ab_pc"],
        row_school["b_pc"], row_school["tb_pc"], row_school["tbf_pc"]
    ], dtype=float)
    p = p / p.sum()

    q = np.array([
        row_form["fail_pc"], row_form["none_pc"], row_form["ab_pc"],
        row_form["b_pc"], row_form["tb_pc"], row_form["tbf_pc"]
    ], dtype=float)
    q = q / q.sum()

    # --- Positions
    x = np.arange(len(mentions))
    width = 0.35

    # --- Plot
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width/2, p,
        width,
        label="Lycée",
        alpha=0.7
    )

    ax.bar(
        x + width/2, q,
        width,
        label="Formation",
        alpha=0.7
    )

    # --- Axes et labels
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

    plt.tight_layout()
    plt.show()



