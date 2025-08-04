import code
import nltk
from datasets import load_dataset

ds = load_dataset("laion/laion400m", split="train", streaming=True)

url_iter = iter(ds["url"])
cap_iter = iter(ds["caption"])

caps = [next(cap_iter) for _ in range(100000)]

toks = [nltk.word_tokenize(cap.lower()) for cap in caps if cap is not None]

toks_flat = [tok for sublist in toks for tok in sublist if tok.isalpha()]
toks_unique = list(set(toks_flat))
tagged = nltk.pos_tag(toks_unique)

parts = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]

filtered = [tag for tag, pos in tagged if pos in parts]

appearances = [(tag, sum(
        tag in caption_tok for caption_tok in toks))
    for tag in filtered]

top_k = sorted(appearances, key=lambda x: x[1], reverse=True)[:1000]


with open("tmp/caps.txt", "w") as f:
    for cap, _score in top_k:
        f.write(cap + "\n")


