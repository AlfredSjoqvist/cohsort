# CohSort: Can automatic cohesion measures improve the readability of summaries by reordering their sentences?

CohSort is a research prototype for post-processing extractive summaries. It reorders sentences to *maximize sentence-to-sentence cohesion* using Coh-Metrix–style metrics, SBERT-based LSA, and the L2 Reading Index, and evaluates how this affects human-perceived readability.

This code was developed in the context of the TextAD project at Linköping University and used in the study:

> **Can automatic cohesion measures improve the readability of summaries by reordering their sentences?**

The main finding of the project is that *simply maximizing local cohesion does not necessarily improve human-perceived readability*, even when linguistic cohesion metrics improve.

---

## Method overview

Given an extractive text summary (e.g. from ElsaSum), CohSort:

1. **Parses and annotates** sentences with Stanza (tokenization, POS, dependencies) and Benepar (constituency trees).
2. Computes sentence-level **Coh-Metrix style indices**:

   * **LSA Adjacent Sentences** (LSASS1, LSASS1d)
   * **LSA Givenness** (LSAGN, LSAGNd)
   * **L2 Reading Index components**:

     * Content Word Overlap (CRFCWO1)
     * Sentence Syntax Similarity (SYNSTRUTa)
     * NyLLex-based word frequency (WRDFRQmc; not used for reordering)
3. Uses **Sentence-BERT (SBERT)** embeddings instead of classical SVD-based LSA.
4. Aggregates the indices into a single **cohesion score** for a particular sentence ordering.
5. Uses **simulated annealing** (and alternative search baselines) to search over permutations of sentences and return a high-scoring order.

We then compare:

* **Original summary order** (ElsaSum) vs
* **CohSort-reordered summaries**

using both automatic metrics (via SAPIS) and a human survey (N = 22).

---

## Repository structure

High-level structure (non-exhaustive):

* `main.py` – Entry point; orchestrates parsing, scoring, and sentence reordering.
* `parsing.py` – Utilities for sentence segmentation and Stanza/Benepar parsing.
* `SBERT.py` / `cached_SBERT.py` – SBERT sentence embeddings (with caching for speed).
* `cosine_sim.py` – Cosine similarity helpers for embeddings.
* `lsa_adjacent_sentences.py` / `lsa_givenness.py` (`lsa_all_sentences.py`) – LSA-based Coh-Metrix indices.
* `content_word_overlap.py` – Content word overlap (L2 component).
* `syntactic_similarity.py` – Sentence syntax similarity via dependency and constituency trees.
* `word_frequencies.py` / `L2_index.py` – NyLLex-based word frequency and L2 index helpers.
* `taaco_givenness.py` – Additional givenness-style cohesion measures.
* `text_scorer.py` – Aggregates individual indices into a single cohesion score.
* `simulated_annealing.py` / `genetic_search.py` / `brute_force.py` – Search strategies over sentence permutations.
* `stanza_resources/` – Bundled Stanza models (including Swedish models).
* `Summaries/` – Example original and reordered summaries used in the study.
* `environment.yml`, `requirements.txt` – Environment and dependency specification.
* `*.pdf` – Full and short versions of the research paper.

---

## Installation

Create and activate a conda environment (recommended):

```bash
conda env create -f environment.yml
conda activate cohsort
```

Or, using `requirements.txt` and a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The repo includes pre-downloaded Stanza resources in `stanza_resources/`. If you want to regenerate them, see `parsing.py`.

---

## Quick start

Reorder the sentences in a Swedish summary string:

```python
from main import reorder_summary

summary = """...
Ditt svenska sammandrag här.
..."""

reordered = reorder_summary(summary)
print(reordered)
```

Typical pipeline inside `reorder_summary`:

1. Split input into sentences.
2. Parse sentences with Stanza/Benepar.
3. Compute LSA + L2-based indices.
4. Run simulated annealing to search over sentence permutations.
5. Return the best-scoring order as a new text.

For more fine-grained control (e.g. using a specific search strategy or index weighting), see:

* `simulated_annealing.py`
* `genetic_search.py`
* `text_scorer.py`

---

## Reproducing the study (outline)

The repo contains enough code and artefacts to roughly reproduce the experiments from the paper:

1. **Prepare data**

   * Collect Swedish news articles (we used 15 DN articles, 300–400 words).
   * Produce 8-sentence extractive summaries with ElsaSum (not included in this repo).

2. **Generate CohSort summaries**

   * Run each ElsaSum summary through `reorder_summary` to obtain a reordered version.

3. **Technical evaluation**

   * Compare ElsaSum vs CohSort summaries with **SAPIS**, extracting the most frequently changed metrics.

4. **Human evaluation**

   * Build a survey where participants rate pairs of summaries (coherence, ease of reading, ease of understanding) and perform direct comparisons.
   * Analyze results with paired-samples t-tests over aggregated readability scores.

See the included PDFs for all methodological details, statistics, and discussions.

---

## Key results (high-level)

* CohSort consistently **improves cohesion metrics** (e.g. content word overlap between adjacent sentences, LSA-based measures).
* Human participants generally **preferred the original ElsaSum order** on perceived coherence, ease of reading, and ease of understanding.
* This exposes a **gap between computational cohesion measures and perceived readability**, and suggests that maximizing local cohesion alone is not sufficient.

---

## Citation

If you use this code or ideas from the project, please cite:

> A. Sjöqvist, D. Tufvesson, I. Wanström, K. Stendahl, L. Tullstedt, L. Rammus, S. Davidsson.
> *Can automatic cohesion measures improve the readability of summaries by reordering their sentences?* (2023).

You can find the full and short versions of the paper in the repository root.