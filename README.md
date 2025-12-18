# AI-Powered Grammar Scoring Engine

This repository contains a machine learning pipeline designed to automatically assess the grammatical proficiency of spoken English from raw audio samples. The solution utilizes a **Hybrid Feature Architecture**, combining state-of-the-art Deep Learning embeddings with traditional signal processing to achieve high correlation with human annotators (MOS Likert Scale 0-5).

## Key Features

* **Hybrid Architecture:** Merges deep semantic context (Transformers) with linguistic structure (Acoustic Features).
* **Wav2Vec 2.0 Integration:** Utilizes Facebook's pre-trained `wav2vec2-base-960h` to capture latent phonetic and syntactic patterns.
* **Acoustic Engineering:** Extracts explicit prosodic features (Pitch, RMS Energy, Zero-Crossing Rate) using `librosa` to measure fluency and confidence.
* **Dimensionality Reduction:** Implements PCA (Principal Component Analysis) to compress 700+ features into robust signals, preventing overfitting on small datasets.
* **Optimized Regression:** Uses a tuned **XGBoost Regressor** for final score prediction.

## Methodology

The pipeline processes audio files (45-60s length) through the following stages:

1. **Preprocessing:**
   * Audio resampling to 16kHz.
   * Silence removal (25dB threshold) to focus on speech segments.
2. **Feature Extraction:**
   * **Deep Features:** 768-dimensional embeddings from the last hidden layer of Wav2Vec 2.0.
   * **Handcrafted Features:** Statistical moments (Mean, Std, Skew, Kurtosis) of Pitch (F0), Energy, and Speech Rate.
3. **Modeling:**
   * **PCA:** Reduces dimensionality from ~776 to 50 components.
   * **XGBoost:** Trains on the reduced feature set with heavy regularization (`subsample=0.7`, `max_depth=3`).

## Performance

* **Metric:** Pearson Correlation Coefficient & RMSE.
* **Result:** The model demonstrates strong predictive capability, effectively distinguishing between low, intermediate, and high grammatical proficiency.
* **Validation:** Rigorously tested using 5-Fold Cross-Validation to ensure stability across different data splits.

## Tech Stack

* **Python 3.10+**
* **Deep Learning:** `PyTorch`, `Transformers` (Hugging Face)
* **Audio Processing:** `Librosa`, `SoundFile`
* **Machine Learning:** `XGBoost`, `Scikit-Learn`
* **Visualization:** `Matplotlib`, `Seaborn`

## Repository Structure

```
├── dataset/
│   ├── audios/
│   │   ├── test/          # Test audio samples
│   │   └── train/         # Training audio samples
│   └── csvs/
│       ├── test.csv       # Test metadata
│       └── train.csv      # Training labels and metadata
├── .gitattributes         # Git LFS settings
├── grammar scoring engine.ipynb  # Main pipeline code
└── README.md              # Project documentation
