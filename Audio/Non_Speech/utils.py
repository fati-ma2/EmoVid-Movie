"""Code partagé de la partie audio — Projet Données Multimédia, M2 SID 2026.

Toutes les expériences audio utilisent ces fonctions, pour que les résultats soient
comparables : même partition, même classifieur, même évaluation, même format de sortie.

Utilisation dans un notebook Colab :

    from google.colab import drive
    drive.mount("/content/drive")
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/Projet Multimédia/Partie audio/ghislain")
    import utils

    P = utils.setup()                     # données + chemins
    meta = utils.load_metadata(P)         # un extrait par ligne
    X = utils.extract_features("ast", meta, ma_fonction, P)
    clf = utils.train_classifier(X[meta.split == "train"], meta.label[meta.split == "train"])
    ...

Contenu :
    setup / load_metadata / load_audio        -> données
    extract_features / load_features           -> vecteurs (calcul reprenable, mis en cache)
    train_classifier / select_C                -> classifieur commun
    evaluate / plot_confusion / log_results    -> évaluation commune (F1 macro)
    select_weights / fuse                      -> fusion tardive (poids choisis sur le dev)
    save_preds / load_preds                    -> prédictions au format commun (fusion)
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

EMOTIONS = ["amusement", "awe", "contentment", "excitement",
            "anger", "disgust", "fear", "sadness"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
SPLITS = ("train", "dev", "test")
SEED = 42

# Drive du groupe (Colab)
DRIVE_PROJET = Path("/content/drive/MyDrive/Projet Multimédia")
ZIP_ID = "1ykUsEsbMoMlsji4eewSPVgUgqjT98GJH"      # PROJET_SID_2026.zip
COLAB_DATA = Path("/content/data/PROJET_SID_2026")


# --------------------------------------------------------------------------- données

def _in_colab():
    return "google.colab" in sys.modules


def _run(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def setup(parts=("audio", "annotation")):
    """Prépare les données et renvoie les chemins utiles.

    Sur Colab : monte le Drive (si besoin), copie le zip du groupe depuis le Drive et
    n'en décompresse que les dossiers demandés (`parts`) dans /content/data.
    En local : utilise le dossier du projet.

    Renvoie un objet P avec : P.DATA (contient movie/), P.SPLITS_DIR, P.OUT (dossier ghislain),
    P.FEATURES, P.PREDS, P.FIGS.
    """
    if _in_colab():
        if not DRIVE_PROJET.exists():
            from google.colab import drive
            drive.mount("/content/drive")
        if not DRIVE_PROJET.exists():
            raise FileNotFoundError(
                "Dossier 'Projet Multimédia' introuvable dans Mon Drive : clic droit sur le dossier "
                "dans 'Partagés avec moi' > Organiser > Ajouter un raccourci > Mon Drive.")
        data = COLAB_DATA
        manquants = [p for p in parts if not (data / "movie" / p).exists()]
        if manquants:
            zip_local = Path("/content/PROJET_SID_2026.zip")
            if not zip_local.exists():
                zip_drive = DRIVE_PROJET / "PROJET_SID_2026.zip"
                if zip_drive.exists():
                    print("Copie du zip depuis le Drive (~2-3 min)...")
                    _run(f'cp "{zip_drive}" {zip_local}')
                else:
                    _run(f'pip install -q gdown && gdown --fuzzy '
                         f'"https://drive.google.com/file/d/{ZIP_ID}/view" -O {zip_local}')
            motifs = " ".join(f"'PROJET_SID_2026/movie/{p}/*'" for p in manquants)
            print("Décompression de :", ", ".join(manquants))
            _run(f"mkdir -p /content/data && unzip -q -o {zip_local} {motifs} -d /content/data")
        P = SimpleNamespace(
            DATA=data,
            SPLITS_DIR=DRIVE_PROJET / "Repartition_Splits" / "Répartition",
            OUT=DRIVE_PROJET / "Partie audio" / "ghislain",
        )
    else:
        root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / "movie").exists() and (p / "splits").exists())
        P = SimpleNamespace(DATA=root, SPLITS_DIR=root / "splits", OUT=root / "audio" / "ghislain")

    P.FEATURES = P.OUT / "features"
    P.PREDS = P.OUT / "preds"
    P.FIGS = P.OUT / "figures"
    for d in (P.OUT, P.FEATURES, P.PREDS, P.FIGS):
        d.mkdir(parents=True, exist_ok=True)
    for p in parts:
        assert (P.DATA / "movie" / p).exists(), f"{P.DATA / 'movie' / p} introuvable"
    print("Données  :", P.DATA)
    print("Résultats:", P.OUT)
    return P


def _read_split_list(P, split):
    # Drive : train.txt ; local : train_films.txt (même contenu)
    for name in (f"{split}.txt", f"{split}_films.txt"):
        f = P.SPLITS_DIR / name
        if f.exists():
            return f.read_text().split()
    raise FileNotFoundError(f"Liste '{split}' introuvable dans {P.SPLITS_DIR}")


def load_metadata(P):
    """Un extrait par ligne : video_id, film_id, emotion, label (0-7), split, audio (chemin),
    les annotations fournies (caption, brightness, colorfulness, hue, duration)
    et, si b0 a été exécuté, statut / langue / duree_parole (fichier langues.csv)."""
    film2split = {f: s for s in SPLITS for f in _read_split_list(P, s)}
    rows = []
    for f in sorted((P.DATA / "movie" / "annotation").glob("*/*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"video_id": a["video_id"], "film_id": f.parent.name,
                     "emotion": a["emotion"], "duration": a["duration"],
                     "caption": a.get("caption", ""), "brightness": a.get("brightness"),
                     "colorfulness": a.get("colorfulness"), "hue": a.get("hue")})
    meta = pd.DataFrame(rows)
    meta["label"] = meta.emotion.map(EMO2ID)
    meta["split"] = meta.film_id.map(film2split)
    meta["audio"] = [str(P.DATA / "movie" / "audio" / f / f"{v}.mp3")
                     for f, v in zip(meta.film_id, meta.video_id)]
    assert meta.label.notna().all(), "Émotion inconnue dans les annotations"
    assert meta.split.notna().all(), "Des films ne sont dans aucune liste train/dev/test"

    langues = P.OUT / "langues.csv"
    if langues.exists():
        lg = pd.read_csv(langues)[["video_id", "statut", "langue", "duree_parole"]]
        meta = meta.merge(lg, on="video_id", how="left")
        meta["parole"] = meta.statut.isin(["anglais", "autre_langue", "incertain"])
    else:
        print("langues.csv absent : lancer b0_nettoyage_audio pour avoir le statut parole/langue")
    return meta.sort_values("video_id").reset_index(drop=True)


def load_audio(path, sr=16000):
    """Charge un fichier audio en mono, rééchantillonné à `sr` Hz (float32, entre -1 et 1)."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- vecteurs

def extract_features(name, meta, fn, P, chunk=200, column="audio"):
    """Calcule un vecteur par extrait avec `fn` et le met en cache dans P.FEATURES/<name>/.

    fn : fonction qui reçoit une liste d'entrées (par défaut les chemins audio, ou le contenu
    d'une autre colonne de `meta`, par ex. column="caption") et renvoie un tableau (n, d).
    Le calcul se fait par blocs de `chunk` extraits, chacun sauvegardé dès qu'il est fini :
    si la session Colab coupe, relancer reprend au bloc suivant.
    Renvoie X (n_extraits, d), dans l'ordre des lignes de `meta`.
    """
    from tqdm.auto import tqdm
    d = P.FEATURES / name
    d.mkdir(parents=True, exist_ok=True)
    ids = meta.video_id.tolist()
    paths = meta[column].tolist()
    blocs = range(0, len(ids), chunk)
    for i in tqdm(blocs, desc=f"extraction {name}"):
        part = d / f"part_{i // chunk:04d}.npz"
        if part.exists():
            continue
        X = np.asarray(fn(paths[i:i + chunk]), dtype=np.float32)
        assert len(X) == len(ids[i:i + chunk]), "fn doit renvoyer une ligne par fichier"
        np.savez(part, ids=np.array(ids[i:i + chunk]), X=X)
    X = load_features(name, P, ids)
    np.savez(P.FEATURES / f"{name}.npz", ids=np.array(ids), X=X)   # fichier unique, pratique
    return X


def load_features(name, P, ids=None):
    """Relit les vecteurs déjà calculés, dans l'ordre de `ids` si fourni."""
    single = P.FEATURES / f"{name}.npz"
    if single.exists():
        z = np.load(single)
        all_ids, X = z["ids"], z["X"]
    else:
        parts = sorted((P.FEATURES / name).glob("part_*.npz"))
        if not parts:
            raise FileNotFoundError(f"Aucun vecteur '{name}' dans {P.FEATURES}")
        zs = [np.load(p) for p in parts]
        all_ids = np.concatenate([z["ids"] for z in zs])
        X = np.concatenate([z["X"] for z in zs])
    if ids is None:
        return X
    pos = {v: i for i, v in enumerate(all_ids)}
    manquants = [v for v in ids if v not in pos]
    if manquants:
        raise KeyError(f"{len(manquants)} extraits sans vecteur '{name}' (ex. {manquants[:3]})")
    return X[[pos[v] for v in ids]]


# --------------------------------------------------------------------------- classifieur

def train_classifier(X, y, C=1.0, kind="logreg"):
    """Classifieur commun à toutes les expériences.

    Standardisation + régression logistique avec pondération des classes (class_weight=
    'balanced') pour compenser le déséquilibre (anger ~22 %, amusement ~8 %).
    kind="svm" : SVM à noyau RBF, même pondération, avec probabilités.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    if kind == "logreg":
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    elif kind == "svm":
        model = SVC(C=C, kernel="rbf", class_weight="balanced", probability=True,
                    random_state=SEED)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), model).fit(X, y)


CS_DEFAUT = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0)


def select_C(X_train, y_train, X_dev, y_dev, Cs=CS_DEFAUT, kind="logreg", tol=0.01):
    """Choisit C sur le dev (F1 macro), réentraîne et renvoie (classifieur, C, tableau des scores).

    Règle de prudence : parmi les C dont le F1 macro dev est à moins de `tol` du meilleur, on garde
    le PLUS PETIT (classifieur le plus régularisé, donc le moins sujet au surapprentissage).
    Le test n'est jamais utilisé pour ce choix."""
    from sklearn.metrics import f1_score
    scores = []
    for C in Cs:
        clf = train_classifier(X_train, y_train, C=C, kind=kind)
        scores.append({"C": C, "f1_macro_dev": f1_score(y_dev, clf.predict(X_dev), average="macro")})
    scores = pd.DataFrame(scores)
    meilleur = scores.f1_macro_dev.max()
    choisi = float(scores.loc[scores.f1_macro_dev >= meilleur - tol, "C"].min())
    return train_classifier(X_train, y_train, C=choisi, kind=kind), choisi, scores


# --------------------------------------------------------------------------- évaluation

def evaluate(y_true, proba, parole=None, name="", split="", verbose=True):
    """Métriques communes. `proba` : (n, 8) probabilités (ou vecteur d'étiquettes prédites).

    Renvoie un dict : f1_macro (métrique principale), balanced_acc, accuracy, f1 par émotion,
    et, si `parole` (booléens) est fourni, le F1 macro séparé sur les extraits avec / sans parole.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = proba.argmax(1) if proba.ndim == 2 else proba
    res = {"modele": name, "split": split, "n": len(y_true),
           "f1_macro": f1_score(y_true, y_pred, average="macro"),
           "balanced_acc": balanced_accuracy_score(y_true, y_pred),
           "accuracy": accuracy_score(y_true, y_pred)}
    for e, f in zip(EMOTIONS, f1_score(y_true, y_pred, average=None, labels=range(8))):
        res[f"f1_{e}"] = f
    if parole is not None:
        parole = np.asarray(parole, dtype=bool)
        for nom, m in (("avec_parole", parole), ("sans_parole", ~parole)):
            res[f"f1_macro_{nom}"] = (f1_score(y_true[m], y_pred[m], average="macro")
                                     if m.any() else np.nan)
    if verbose:
        print(f"[{name} | {split}] F1 macro = {res['f1_macro']:.3f} | "
              f"acc. équilibrée = {res['balanced_acc']:.3f} | accuracy = {res['accuracy']:.3f}")
        if parole is not None:
            print(f"   F1 macro avec parole = {res['f1_macro_avec_parole']:.3f} | "
                  f"sans parole = {res['f1_macro_sans_parole']:.3f}")
        print("   F1 par émotion : " + ", ".join(f"{e} {res[f'f1_{e}']:.2f}" for e in EMOTIONS))
    return res


def plot_confusion(y_true, proba, title="", P=None, filename=None):
    """Matrice de confusion normalisée par ligne (chaque ligne = une vraie émotion, somme à 100 %)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    proba = np.asarray(proba)
    y_pred = proba.argmax(1) if proba.ndim == 2 else proba
    cm = confusion_matrix(y_true, y_pred, labels=range(8), normalize="true") * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(8), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(8), EMOTIONS)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > 50 else "black", fontsize=8)
    ax.set(xlabel="émotion prédite", ylabel="vraie émotion", title=title)
    fig.colorbar(im, ax=ax, label="% de la vraie émotion")
    fig.tight_layout()
    if P is not None and filename:
        fig.savefig(P.FIGS / filename, dpi=150)
    return fig


def log_results(res, P):
    """Ajoute (ou remplace) une ligne dans resultats.csv : le tableau comparatif de tous les modèles."""
    f = P.OUT / "resultats.csv"
    new = pd.DataFrame([res])
    if f.exists():
        old = pd.read_csv(f)
        old = old[~((old.modele == res["modele"]) & (old.split == res["split"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)
    return new


# --------------------------------------------------------------------------- fusion tardive

def select_weights(probas_dev, y_dev, step=0.1):
    """Fusion tardive : cherche les poids (somme = 1, pas de `step`) qui maximisent le F1 macro
    sur le dev de la moyenne pondérée des probabilités de plusieurs modèles.
    probas_dev : liste de tableaux (n, 8). Renvoie (poids, F1 macro dev)."""
    import itertools
    from sklearn.metrics import f1_score
    k = len(probas_dev)
    grille = np.round(np.arange(0, 1 + 1e-9, step), 6)
    best_w, best_f1 = None, -1.0
    for w in itertools.product(grille, repeat=k - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = (*w, 1 - sum(w))
        f1 = f1_score(y_dev, fuse(probas_dev, w).argmax(1), average="macro")
        if f1 > best_f1 + 1e-12:
            best_w, best_f1 = tuple(float(round(x, 6)) for x in w), f1
    return best_w, best_f1


def fuse(probas, weights):
    """Moyenne pondérée de probabilités (n, 8) ; chaque ligne reste une distribution."""
    out = sum(w * np.asarray(p) for w, p in zip(weights, probas))
    return out / out.sum(1, keepdims=True)


# --------------------------------------------------------------------------- prédictions

def save_preds(video_ids, proba, name, split, P):
    """Enregistre les probabilités au format commun : P.PREDS/<name>_<split>.csv
    (colonnes : video_id + une colonne par émotion, dans l'ordre de EMOTIONS)."""
    proba = np.asarray(proba, dtype=float)
    assert proba.shape == (len(video_ids), len(EMOTIONS)), proba.shape
    assert np.allclose(proba.sum(1), 1, atol=1e-3), "chaque ligne doit sommer à 1"
    df = pd.DataFrame(proba, columns=EMOTIONS)
    df.insert(0, "video_id", list(video_ids))
    out = P.PREDS / f"{name}_{split}.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    return out


def load_preds(name, split, P, video_ids=None):
    """Relit des prédictions (n, 8), dans l'ordre de `video_ids` si fourni."""
    df = pd.read_csv(P.PREDS / f"{name}_{split}.csv")
    if video_ids is not None:
        df = df.set_index("video_id").loc[list(video_ids)].reset_index()
    return df[EMOTIONS].to_numpy()
