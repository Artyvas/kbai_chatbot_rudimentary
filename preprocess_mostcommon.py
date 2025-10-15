import re
import unicodedata
from pathlib import Path
import spacy

# 1) Load English pipeline (tagger + lemmatizer already included)
#    If needed first: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", disable=[])

# 2) Read your list
lines = Path("mostcommon.txt").read_text(encoding="utf-8").splitlines()
NAMES = [w.strip() for w in lines[:20] if w.strip()]      # keep exact case for names
WORDS = [w.strip() for w in lines[20:] if w.strip()]

# --- Helpers for ASCII normalization (avoid curly quotes etc.) ---
def ascii_quotes(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace("’", "'").replace("‘", "'").replace("`", "'")

# 3) Generate candidate surface forms to bake into a lookup (lowercase, ASCII-only)
def candidate_forms(w: str):
    w = ascii_quotes(w.lower())
    forms = {w}
    if len(w) > 2:
        forms |= {w + "s", w + "ed", w + "ing", w + "'s"}
    if len(w) > 3:
        forms |= {w + "er", w + "est"}
    return forms

surfaces = set()
for w in WORDS:
    surfaces |= candidate_forms(w)

# Common function words (lowercase)
surfaces |= {"to","for","at","in","on","of","with","and","or","not","no","a","an","the"}

# 4) Lemmatize every surface; gather coarse POS buckets
LEMMA_LOOKUP = {}
VERBS, NOUNS, ADJS, ADVS = set(), set(), set(), set()

surfaces_list = sorted(surfaces)

# Process ONE surface per Doc to avoid tokenization drift (e.g., "dog's")
for doc in nlp.pipe(surfaces_list, batch_size=1000):
    token = next((t for t in doc if not t.is_space), None)
    if token is None:
        continue
    surface = ascii_quotes(doc.text.lower())
    lemma = ascii_quotes(token.lemma_.lower())
    LEMMA_LOOKUP[surface] = lemma

    p = token.pos_
    if p in ("VERB", "AUX"):
        VERBS.add(lemma)
    elif p in ("NOUN", "PROPN"):
        NOUNS.add(lemma)
    elif p == "ADJ":
        ADJS.add(lemma)
    elif p == "ADV":
        ADVS.add(lemma)

# 5) Emit Python you can paste into SentenceReadingAgent.py
def as_py_set(name, items):
    items = sorted(items)
    return f"{name} = set({items!r})"

def as_py_list(name, items):
    items = list(items)
    return f"{name} = {items!r}"

def as_py_dict(name, d):
    items = dict(sorted(d.items()))
    return f"{name} = {items!r}"

print("# ====== BEGIN AUTO-GENERATED (paste into SentenceReadingAgent.py) ======")
print(as_py_list("NAMES", NAMES))       # keep exact-case names
print(as_py_dict("LEMMA_LOOKUP", LEMMA_LOOKUP))  # lowercase, ASCII-only keys
print(as_py_set("VERBS", VERBS))
print(as_py_set("NOUNS", NOUNS))
print(as_py_set("ADJS", ADJS))
print(as_py_set("ADVS", ADVS))
print("# ====== END AUTO-GENERATED ======")