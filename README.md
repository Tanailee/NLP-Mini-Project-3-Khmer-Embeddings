# Khmer Temple NLP Explorer

This repository contains an NLP Mini Project 3 for the Master of Data Science program at the Institute of Technology of Cambodia. The project builds Khmer word embeddings and neural language models using a Khmer temple corpus.

## Main Notebook

The final notebook is:

```text
notebook/NLP_Mini_Project_3_Khmer_Word_Embeddings_FINAL_10_10_CLEAN.ipynb
```

The notebook includes Khmer preprocessing, tokenization, EDA, skip-gram training with negative sampling, PCA visualization, n-gram baseline modeling, neural language modeling, scratch embedding comparison, model comparison, and a report summary.

## Streamlit App

The Streamlit app is:

```text
streamlit_app/app.py
```

App title:

```text
Khmer Temple NLP Explorer
```

### Streamlit Features

- Home dashboard with project metrics.
- Default `temples.txt` corpus analysis.
- Upload new `.txt` file for preprocessing and EDA.
- Khmer text cleaning and tokenization.
- Khmer punctuation and number-only token removal.
- EDA dashboard with Plotly charts.
- Word embedding explorer using saved skip-gram embeddings.
- The Word Embedding Explorer hides `<UNK>` and common function words by default for cleaner interpretation. This is display-only filtering and does not change the trained model.
- PCA map of pretrained embeddings.
- Advanced Clustering page for optional embedding exploration.
- K-means clustering of word embeddings with silhouette score selection.
- PCA cluster visualization and hierarchical clustering dendrogram.
- Next-word prediction using the pretrained neural LM when available.
- Model comparison table and charts.
- Project report page.
- GitHub and Streamlit Cloud deployment guide.

Uploaded `.txt` files are used for preprocessing and EDA only. The pretrained embeddings, PCA map, and next-word prediction model are based on the original `temples.txt` corpus. Retraining on uploaded text can be added as future work.

## Dataset

Main corpus:

```text
data/temples.txt
```

Fallback path supported by the notebook and app:

```text
temples.txt
```

## Methods

- Khmer text cleaning and normalization.
- Khmer tokenization using `khmernltk` when available.
- Dictionary maximum matching and regex fallback.
- Khmer punctuation and number-only token removal.
- Vocabulary filtering with `MIN_FREQ = 10`.
- Rare words mapped to `<UNK>` as required by the project instruction.
- Skip-gram word embeddings with embedding dimension 50.
- Context window of `+/-4`.
- Negative sampling with `k = 2`.
- PCA visualization of embeddings.
- Optional advanced embedding exploration with K-means clustering.
- Silhouette score graph for choosing the number of clusters.
- PCA cluster visualization colored by K-means cluster.
- Hierarchical clustering dendrogram for selected frequent words.
- Add-one smoothed n-gram baseline.
- Neural language model using previous 5 words.
- Hidden layer size 512.
- Comparison between fixed skip-gram embeddings and scratch learned embeddings.

## Result Summary

The cleaned notebook run removes punctuation from useful tokens and vocabulary.

Saved model comparison result:

| Model | Test Perplexity |
|---|---:|
| N-gram baseline | 158.84 |
| Neural LM fixed skip-gram | 115.10 |
| Neural LM scratch | 229.07 |

The fixed skip-gram neural language model has the lowest test perplexity in the cleaned run. The scratch model can overfit because the corpus is small and domain-specific.

## Advanced Embedding Exploration

The project now includes an optional advanced clustering analysis beyond the core Mini Project 3 requirements:

- K-means clustering of learned skip-gram word embeddings.
- Silhouette score graph for choosing a reasonable K.
- PCA visualization colored by K-means cluster.
- Cluster summary table and top words by cluster.
- Hierarchical clustering dendrogram for frequent Khmer words.

These analyses are optional advanced embedding exploration beyond the core Mini Project 3 requirements. They help inspect contextual word groups, but they should not be interpreted as perfect semantic classes because the corpus is small and domain-specific.

## Khmer Font Rendering Note

For best Khmer text rendering in figures, install Noto Sans Khmer or Khmer OS fonts. If Khmer labels appear broken in Matplotlib figures, use the numbered dendrogram version with the word mapping table:

```text
outputs/figures/hierarchical_dendrogram_numbered.png
outputs/tables/dendrogram_word_labels.csv
```

No font files are included in this repository.

## How to Run Locally

Install requirements:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

Open the local URL shown by Streamlit.

## Streamlit Cloud Deployment

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Select this app file:

```text
streamlit_app/app.py
```

5. Deploy.

For the full app experience, commit these folders:

```text
data/
outputs/
models/
streamlit_app/
```

## Folder Structure

```text
data/                 Khmer corpus
notebook/             Final Jupyter notebooks
outputs/tables/       CSV outputs from the notebook
outputs/embeddings/   Saved NumPy embeddings and word-index files
models/               Saved PyTorch model files
reports/              Report-ready material
streamlit_app/        Streamlit UI
requirements.txt      Python dependencies
README.md             Project documentation
.gitignore            Git ignore rules
```

## Limitations

- The corpus is small and domain-specific.
- Khmer tokenization can still be imperfect.
- `MIN_FREQ = 10` maps many rare but meaningful words to `<UNK>`.
- PCA reduces 50-dimensional embeddings to 2 dimensions, so it is only an approximate visualization.
- Uploaded `.txt` files are not used to retrain deep learning models in the Streamlit app.

## Future Work

- Add optional retraining for uploaded corpora.
- Add saved PCA coordinate files for faster app loading.
- Improve Khmer tokenization with larger dictionaries.
- Add richer next-word prediction examples.
- Deploy the app publicly on Streamlit Cloud.
