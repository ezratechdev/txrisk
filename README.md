# txrisk

Open-source models that estimate how likely a blockchain transaction is to be fraudulent and,
where the on-chain evidence allows, what kind of fraud it is.

**Status: Phase 1.** We're comparing methods on public labelled datasets. Nothing here is ready
to score real transactions yet.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                                      # create .venv and install
uv run python -m txrisk.data.download elliptic bitcoinheist  # about 270 MB download
uv run python -m txrisk.experiments.elliptic_baseline        # fraud or not (about 1 minute)
uv run python -m txrisk.experiments.bitcoinheist_typology    # which fraud (about 10 minutes)
uv run pytest
```

Results are written to [reports/elliptic_baseline.md](reports/elliptic_baseline.md) and
[reports/bitcoinheist_typology.md](reports/bitcoinheist_typology.md).

Datasets are saved under `data/`, which git ignores. To keep them on another drive, set
`TXRISK_DATA_DIR` to a folder there.

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
- **Split by actor as well as by time.** Test rows for an address seen in training are
  dropped, so a model is scored on new actors rather than on remembering old ones.
- **Learn the decision threshold** on held-out training data. At a 0.5% fraud rate an
  assumed 0.5 cut-off flags nothing and makes a working model look broken.
- **Let the model answer "unknown".** New fraud types appear that were never in the
  training data; naming a trained type for them is worse than admitting ignorance.

## Responsible use

Scores are risk indicators, not proof of wrongdoing. Don't publish lists of addresses flagged
by these models: false positives point at real people.
