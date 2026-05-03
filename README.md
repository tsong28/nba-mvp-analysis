# NBA MVP — Causal Effect of PER on Vote Share

Course project (Foundations of Data Science). Group: Tyler Song, Hyeonung Cho.

**Question.** How does a player's Player Efficiency Rating (PER) affect his
MVP vote share in a given NBA season?

**Causal model.** Treatment: PER. Outcome: MVP vote share. Confounder:
Usage Rate. Adjustment set: `{Usage Rate}`.

```
   PER ────────► MVP share
    ▲              ▲
    │              │
    └── Usage Rate ┘
```

## Layout

```
nba-mvp-per/
├── analysis.ipynb       # everything: download, process, EDA, model framework
├── data/
│   ├── raw/             # Kaggle CSVs (gitignored)
│   └── processed/       # Built analysis table (gitignored)
├── figures/             # PNGs from the notebook (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab analysis.ipynb        # or: jupyter notebook
```

To use the in-notebook Kaggle downloader, create an API token at
https://www.kaggle.com/settings ("Create New Token"), move the file to
`~/.kaggle/kaggle.json`, and `chmod 600` it. Otherwise download
[sumitrodatta/nba-aba-baa-stats](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)
manually and drop `Advanced.csv` and `Player_Award_Shares.csv` into `data/raw/`.

## Data scope (matches proposal feedback)

- Restricted to NBA seasons from 1980 onward (3-point era).
- For traded players, the season-total ("TOT") row is used.
- Only players who **actually received MVP votes** are kept — the proposal
  grader flagged that including everyone else would force the response to be
  ~all zeros, which a Beta likelihood cannot represent.

## Modeling (scaffolded, not fit)

Two options written in the notebook:

1. **Beta regression** (proposal's choice).
   `logit(μᵢ) = α + β_P·PERᵢ + β_U·USGᵢ`,  `yᵢ ~ Beta(μᵢ·φ, (1−μᵢ)·φ)`.
   Priors per proposal: `α, β_P, β_U ~ Normal(0, 0.5)`, `φ ~ Exponential(1)`.

2. **LogitNormal** (grader's suggested fallback).
   `logit(yᵢ) ~ Normal(α + β_P·PERᵢ + β_U·USGᵢ, σ)`. Easier to fit if Beta
   gives divergences.

Predictors are standardised before fitting. Both code blocks are commented
out — uncomment when ready and `pip install pymc arviz`.
