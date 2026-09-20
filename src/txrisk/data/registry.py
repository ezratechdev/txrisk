"""Public datasets the project can fetch: where they come from and how to verify them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteFile:
    url: str
    sha256: str | None = None
    """Pinned after the first verified download, so later downloads can't silently change."""


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    source: str
    files: tuple[RemoteFile, ...]


_PYG_ELLIPTIC = "https://data.pyg.org/datasets/elliptic"

DATASETS = {
    "elliptic": Dataset(
        name="elliptic",
        description="203,769 Bitcoin transactions over 49 time steps, labelled licit/illicit",
        source="Weber et al. 2019 (Elliptic), mirrored by PyTorch Geometric",
        files=(
            RemoteFile(
                f"{_PYG_ELLIPTIC}/elliptic_txs_features.csv.zip",
                "d33d62159e64b5e889f1a7ea880227c612775b58d409598855e0c4400fa52b3e",
            ),
            RemoteFile(
                f"{_PYG_ELLIPTIC}/elliptic_txs_edgelist.csv.zip",
                "a2f9f6b67a39da2d8cf87fe77b9db89571ba6d880e5dd5b5991dc45c80fa34ec",
            ),
            RemoteFile(
                f"{_PYG_ELLIPTIC}/elliptic_txs_classes.csv.zip",
                "4ca957f0ceffd5dd164e255c7d5ad9ee69a6fa64ae1dd94d6f113e5ebf3b07ba",
            ),
        ),
    ),
    "bitcoinheist": Dataset(
        name="bitcoinheist",
        description="~2.9M Bitcoin addresses (2009-2018) labelled with ransomware family",
        source="Akcora et al. 2019, UCI Machine Learning Repository",
        files=(
            RemoteFile(
                "https://archive.ics.uci.edu/static/public/526/"
                "bitcoinheistransomwareaddressdataset.zip",
                "cdf3bb34199367b08285cf7f9c1db06c4a262d353aa596055e97db77f4514e5b",
            ),
        ),
    ),
}
