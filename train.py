"""
train.py -- train a character n-gram Naive Bayes language detector for EN/FR/VI/ES.

The model is implemented directly (not via a library) so its parameters export
cleanly to JavaScript and run identically in the browser. Pipeline:
  preprocess -> char n-grams (1..3) -> per-language Laplace-smoothed NB
  -> held-out accuracy + confusion matrix -> export model/lang_model.json
"""
from __future__ import annotations
import json, re, sys, math, random
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from corpus import CORPUS  # noqa: E402

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260617
LANGS = ["en", "fr", "vi", "es"]
NMIN, NMAX = 1, 2
ROOT = Path(__file__).resolve().parent

# keep letters (incl. accented/Vietnamese), drop digits/punct, collapse spaces, pad
_keep = re.compile(r"[^a-zà-ÿăâđêôơưạ-ỹ\s]", re.IGNORECASE)
def preprocess(text: str) -> str:
    t = text.lower()
    t = _keep.sub(" ", t)
    t = " " + re.sub(r"\s+", " ", t).strip() + " "
    return t

def ngrams(text: str):
    t = preprocess(text)
    for n in range(NMIN, NMAX + 1):
        for i in range(len(t) - n + 1):
            yield t[i:i+n]

def split_data(seed=SEED, test_frac=0.25):
    rng = random.Random(seed)
    train, test = [], []
    for lang in LANGS:
        items = list(CORPUS[lang]); rng.shuffle(items)
        k = max(1, int(round(len(items) * test_frac)))
        test += [(s, lang) for s in items[:k]]
        train += [(s, lang) for s in items[k:]]
    rng.shuffle(train); rng.shuffle(test)
    return train, test

def train_model(train, min_global_count=2):
    counts = {l: Counter() for l in LANGS}
    n_docs = Counter()
    for s, lang in train:
        n_docs[lang] += 1
        counts[lang].update(ngrams(s))
    # prune globally-rare n-grams (mostly noise) to keep the model compact
    global_count = Counter()
    for l in LANGS:
        global_count.update(counts[l])
    vocab = {g for g, c in global_count.items() if c >= min_global_count}
    counts = {l: {g: c for g, c in counts[l].items() if g in vocab} for l in LANGS}
    total = {l: sum(counts[l].values()) for l in LANGS}
    N = sum(n_docs.values())
    log_prior = {l: math.log(n_docs[l] / N) for l in LANGS}
    return {"langs": LANGS, "nmin": NMIN, "nmax": NMAX, "V": len(vocab),
            "log_prior": log_prior, "total": total, "counts": counts}

def score(model, text):
    V, out = model["V"], {}
    for l in model["langs"]:
        c, tot = model["counts"][l], model["total"][l]
        s = model["log_prior"][l]
        denom = tot + V
        for g in ngrams(text):
            s += math.log((c.get(g, 0) + 1) / denom)
        out[l] = s
    return out

def predict(model, text):
    sc = score(model, text)
    return max(sc, key=sc.get)

def softmax(d):
    m = max(d.values()); e = {k: math.exp(v - m) for k, v in d.items()}
    z = sum(e.values()); return {k: v / z for k, v in e.items()}

def main():
    train, test = split_data()
    model = train_model(train)
    # evaluate
    y_true = [l for _, l in test]
    y_pred = [predict(model, s) for s, _ in test]
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    idx = {l: i for i, l in enumerate(LANGS)}
    cm = np.zeros((4, 4), int)
    for a, b in zip(y_true, y_pred):
        cm[idx[a], idx[b]] += 1

    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.6, 4.2))
    plt.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
    names = {"en": "English", "fr": "Français", "vi": "Tiếng Việt", "es": "Español"}
    plt.xticks(range(4), [names[l] for l in LANGS], rotation=20, ha="right")
    plt.yticks(range(4), [names[l] for l in LANGS])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Language detector — held-out accuracy {acc*100:.0f}%")
    plt.tight_layout(); plt.savefig(ROOT / "results" / "figures" / "confusion_matrix.png", dpi=150)
    plt.close()

    # export compact JSON (counts kept as plain dicts)
    export = {"langs": model["langs"], "nmin": NMIN, "nmax": NMAX, "V": model["V"],
              "log_prior": model["log_prior"], "total": model["total"],
              "counts": {l: dict(model["counts"][l]) for l in LANGS}}
    (ROOT / "model").mkdir(exist_ok=True)
    mj = ROOT / "model" / "lang_model.json"
    mj.write_text(json.dumps(export, ensure_ascii=False))
    json.dump({"held_out_accuracy": round(acc, 3), "n_test": len(y_true),
               "vocab_size": model["V"], "n_train": len(train),
               "confusion_matrix": cm.tolist(), "langs": LANGS},
              open(ROOT / "results" / "metrics.json", "w"), indent=2)

    print(f"train={len(train)}  test={len(y_true)}  vocab(n-grams)={model['V']}")
    print(f"held-out accuracy = {acc*100:.1f}%")
    print(f"model JSON size = {mj.stat().st_size/1024:.0f} KB")
    print("examples:")
    for ex in ["Tôi thích bơi lội và ba môn phối hợp.",
               "J'adore le triathlon et la natation.",
               "Me encanta nadar y el triatlón.",
               "I love swimming and triathlon."]:
        p = softmax(score(model, ex))
        top = max(p, key=p.get)
        print(f"  [{top}] {p[top]*100:4.1f}%  <- {ex}")

if __name__ == "__main__":
    main()
