# Raga Recognition from Audio: A Machine Learning Approach to Carnatic Music Classification

**Author:** Akshaya Bharadhwaj | Master of Data Science, UC Irvine  
**Dataset:** Saraga Carnatic v1.5 (mirdata)
**Link to the Dataset:** https://github.com/MTG/saraga 
---

## Overview

This project investigates whether static and temporal acoustic features 
extracted from audio recordings can reliably discriminate Carnatic ragas — 
the complex melodic framework at the heart of South Indian classical music.

Using the Saraga Carnatic v1.5 dataset (249 professionally recorded tracks, 
96 unique ragas), we develop a 228-dimensional hybrid feature vector and 
evaluate five classifiers on a 3-class raga identification task 
(Todi, Kamas, Saurastram).


## Project Structure
```
├── Carnatic Music Raga Classification.ipynb   # Main notebook (run end-to-end)
└── README.md
```

---

## Key Findings

- **Pitch histogram is the dominant feature** (62% survival after feature 
  selection) independently confirming the music-theoretic prediction that 
  raga identity is defined by note-usage patterns
- **Timbre is irrelevant**: MFCC features survive at only 8–15%, confirming 
  raga is encoded in melody, not voice quality
- **XGBoost** achieves the best accuracy (77.8%) with perfect Todi recall 
  (F1 = 0.93) and strong Kamas detection (F1 = 0.80)
- **Unsupervised KMeans clustering** on 163 remaining tracks recovers 
  moderate note-complexity structure (63% alignment with pentatonic vs. 
  sampurna categories)

---

## Methods

| Component | Details |
|-----------|---------|
| Dataset | Saraga Carnatic v1.5 via mirdata |
| Target ragas | Todi, Kamas, Saurastram (7 tracks each) |
| Feature vector | 228-dim hybrid (pitch annotations + raw audio) |
| Augmentation | 3 deterministic segments per track (25/50/75%) |
| Validation | GroupKFold(n_splits=3) — prevents track-level leakage |
| Models | SVM, Random Forest, Soft Voting, MLP, XGBoost |
| Best model | XGBoost — 77.8% ± 0.208 |

---

## Feature Vector Composition (228 dims)

| Feature Group | Dims | Source |
|---------------|------|--------|
| Chroma mean + std | 24 | Raw audio (middle segment) |
| Chroma transition matrix | 144 | Raw audio (middle segment) |
| Tonic-normalized pitch histogram | 24 | Saraga pre-extracted annotations |
| Pitch mean + std | 2 | Saraga pre-extracted annotations |
| Gamaka (ornamentation) features | 2 | Saraga pre-extracted annotations |
| Directional velocity | 6 | Saraga pre-extracted annotations |
| MFCC mean + std | 26 | Raw audio (middle segment) |

---

## How to Reproduce

1. Open the notebook in Google Colab  
2. Run **Section 1** (Setup) — installs dependencies and downloads 
   Saraga Carnatic v1.5 (~2GB) automatically via mirdata  
3. Run all cells in order  
4. Feature extraction takes ~5–15 minutes depending on hardware  
5. All random seeds are fixed at 42 for deterministic results

---

## Dependencies
```
mirdata, librosa, scikit-learn, xgboost, numpy, scipy, 
matplotlib, seaborn, pandas, tqdm, unicodedata
```

Install via:
```bash
pip install mirdata xgboost
```
All other dependencies are pre-installed in Google Colab.

---

## References

- Koduri et al. (2014). Intonation analysis of ragas in Carnatic music. *JNMR*
- Gulati et al. (2016). Phrase-based raga recognition using vector space modeling. *ICASSP*
- Madhusudhan & Murthy (2019). Recurrent network-based automatic raga recognition. *ISMIR*
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *KDD*
- Serra (2011). A multicultural approach to music information research. *ISMIR*
