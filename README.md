# blockchain_analysis

Open-source models that estimate how likely a blockchain transaction is to be fraudulent and,
where the on-chain evidence allows, what kind of fraud it is.

**Status: Phase 1.** We're comparing methods on public labelled datasets. Nothing here is ready
to score real transactions yet.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                                          # create .venv and install
uv run python -m blockchain_analysis.data.download elliptic      # about 150 MB download
uv run python -m blockchain_analysis.experiments.elliptic_baseline
uv run pytest
```

Results are written to [reports/elliptic_baseline.md](reports/elliptic_baseline.md).

Datasets are saved under `data/`, which git ignores. To keep them on another drive, set
`BA_DATA_DIR` to a folder there.

## Datasets

| Name | Contents | Source | Notes |
|---|---|---|---|
| `elliptic` | 203,769 Bitcoin transactions labelled licit, illicit or unknown | Weber et al. 2019, via the PyTorch Geometric mirror | Features are anonymised: useful for comparing methods, not for scoring live data |
| `bitcoinheist` | ~2.9M Bitcoin addresses labelled with ransomware family | Akcora et al. 2019, UCI ML Repository | For fraud-type classification |

This repository doesn't redistribute any dataset. The download script fetches each one from its
source and checks it against a pinned SHA-256 checksum. Check each source's terms before use.

## How models are evaluated

- **Split by time.** Train on the past, test on the future. Random splits leak information
  and overstate results; the baseline report shows by how much.
- **Use metrics suited to rare events:** PR-AUC, recall at a 1% false-positive rate, and
  calibration (Brier score, ECE). Never accuracy.
- **Beat the baselines.** A new model has to outperform the simple models in
  `models/baselines.py` on the temporal split.

## Responsible use

Scores are risk indicators, not proof of wrongdoing. Don't publish lists of addresses flagged
by these models: false positives point at real people.
