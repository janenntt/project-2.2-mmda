# Multimodal Movie Genre Classification (Project 2.2)

Multi-label genre prediction (23 genres) from plot summaries and posters on **Simplified MM-IMDb** (25,959 movies).
Frozen **DistilBERT** (text) and **CLIP ViT-B/32** (poster) embeddings feed lightweight heads, compared across:

| ID | Approach |
|---|---|
| T / I | Text-only / image-only MLP baselines |
| EF | Early fusion: per-modality LayerNorm, concatenation, joint MLP |
| EF-MD | Early fusion + modality dropout (p = 0.3) |
| GMU | Gated Multimodal Unit (Arevalo et al., 2017) |
| LF-avg / LF-w | Late fusion: equal weights / per-genre weights tuned on validation |

## Repository layout

```
project-2.2.ipynb                 # all code, sectioned 1–10, with outputs
requirements.txt
data/                             # data.npy, images.npz (not committed; see below)
features/                         # cached embeddings (created by Section 4)
results/                          # CSV/JSON tables used in the report (not committed)
figures/                          # report figures (300 dpi PNG, not committed)
```

## Reproduce

1. **Get the data.** Download *Simplified MM-IMDb* from Kaggle (Ureña, 2024): https://www.kaggle.com/datasets/javierurea/simplified-mm-imdb
   and put `data.npy` and `images.npz` in `data/`.
2. **Install dependencies** (Python ≥ 3.10):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run the notebook** top to bottom:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace project2.2.ipynb --ExecutePreprocessor.timeout=-1
   ```
   * Section 4 downloads both encoders from Hugging Face and caches embeddings to `features/`
     (≈ 40 min on a 2-core CPU, a few minutes on a GPU; it is skipped when the cache exists. Set `RECOMPUTE = True` to rebuild).
   * Everything after Section 4 runs on CPU in a few minutes.
4. All settings (split, seeds, model sizes, learning rate, threshold grids) live in the `CFG` dictionary in Section 1.

## Notes

* Splits are leakage-safe: duplicate summaries/posters are grouped and kept within one split (seed 42, 60/10/30 → 15,581 / 2,595 / 7,783).
* `images.npz` is stored uncompressed, so posters are memory-mapped rather than loaded into RAM.
* Results are reported as mean ± sd over three seeds (42, 7, 2026); per-genre thresholds are tuned on validation only.
* Reported results come from Python 3.13, PyTorch 2.15 on a 2-core CPU. Other hardware or library versions can change the third decimal place.
