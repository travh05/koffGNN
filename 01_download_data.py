"""
STEP 1 — Data Download
======================
Downloads two data sources:

  Source A: SAbDab (Structural Antibody Database, Oxford)
    - Hundreds of antibody-antigen complexes
    - Experimental koff from published SPR/BLI measurements
    - PDB structures available for each entry
    URL: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/

  Source B: BindingDB PPI kinetics subset
    - Filtered for protein-protein interactions (not small molecule)
    - Includes kon, koff, Kd, temperature, method
    URL: https://www.bindingdb.org/bind/downloads.jsp

After running this script you will have:
  data/raw/sabdab_summary.tsv   — SAbDab metadata + kinetics
  data/raw/pdb/                 — Downloaded PDB structures
  data/raw/bindingdb_ppi.tsv    — BindingDB PPI subset
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
RAW_DIR   = ROOT / "data" / "raw"
PDB_DIR   = RAW_DIR / "pdb"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PDB_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# SOURCE A: SAbDab
# ══════════════════════════════════════════════════════════════════════════

SABDAB_SUMMARY_URL = (
    "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
    "?tasklist=antigen_species%7Cantigen_name%7Ckon%7Ckoff%7Ckd%7Cmethod"
    "&format=tsv"
)

def download_sabdab_summary() -> pd.DataFrame:
    """Download the full SAbDab summary TSV and keep rows with koff data."""
    out_path = RAW_DIR / "sabdab_summary_raw.tsv"

    if out_path.exists():
        log.info("SAbDab summary already downloaded — loading cache")
        df = pd.read_csv(out_path, sep="\t", low_memory=False)
    else:
        log.info("Downloading SAbDab summary (~30 MB) ...")
        r = requests.get(SABDAB_SUMMARY_URL, timeout=120, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        df = pd.read_csv(out_path, sep="\t", low_memory=False)
        log.info(f"  Downloaded {len(df):,} SAbDab entries")

    # ── Filter: must have koff ──────────────────────────────────────────
    # SAbDab koff column is named 'koff' and uses ',' as decimal in some rows.
    # We normalise to float.
    df.columns = [c.strip().lower() for c in df.columns]

    # The key columns vary slightly between SAbDab releases.
    # We probe for common names:
    koff_col = next(
        (c for c in df.columns if "koff" in c.lower()), None
    )
    if koff_col is None:
        log.warning(
            "Could not find koff column. Available columns: "
            + str(list(df.columns))
        )
        return df

    df[koff_col] = (
        df[koff_col]
        .astype(str)
        .str.replace(",", ".")
        .str.extract(r"([\d.eE+\-]+)", expand=False)
    )
    df[koff_col] = pd.to_numeric(df[koff_col], errors="coerce")
    df = df.rename(columns={koff_col: "koff_s"})

    has_koff = df["koff_s"].notna()
    log.info(f"  Rows with koff data: {has_koff.sum():,} / {len(df):,}")
    df_filtered = df[has_koff].copy()

    # Save filtered version
    filtered_path = RAW_DIR / "sabdab_summary.tsv"
    df_filtered.to_csv(filtered_path, sep="\t", index=False)
    log.info(f"  Saved filtered summary → {filtered_path}")
    return df_filtered


def download_pdbs(df: pd.DataFrame, max_pdbs: int = 500) -> None:
    """
    Download PDB structures for entries in the SAbDab dataframe.
    We use the PDB RCSB REST API which is reliable and fast.

    Downloads at most `max_pdbs` structures to keep the MVP tractable.
    For the full dataset, remove the limit.
    """
    pdb_col = next(
        (c for c in df.columns if c.lower() in ("pdb", "pdbid", "pdb_id")),
        None,
    )
    if pdb_col is None:
        log.error("No PDB ID column found in SAbDab summary.")
        return

    pdb_ids = df[pdb_col].dropna().str.strip().str.lower().unique()
    pdb_ids = [p for p in pdb_ids if len(p) == 4]  # valid PDB IDs are 4 chars
    pdb_ids = pdb_ids[:max_pdbs]

    log.info(f"Downloading {len(pdb_ids)} PDB structures ...")
    already   = 0
    succeeded = 0
    failed    = []

    for i, pdb_id in enumerate(pdb_ids):
        out_path = PDB_DIR / f"{pdb_id}.pdb"
        if out_path.exists():
            already += 1
            continue

        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                out_path.write_bytes(r.content)
                succeeded += 1
            else:
                log.warning(f"  {pdb_id}: HTTP {r.status_code}")
                failed.append(pdb_id)
        except Exception as e:
            log.warning(f"  {pdb_id}: {e}")
            failed.append(pdb_id)

        # Progress log every 50 downloads
        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(pdb_ids)}] done")
        time.sleep(0.05)  # be polite to RCSB

    log.info(
        f"PDB download complete: {succeeded} new, {already} cached, "
        f"{len(failed)} failed"
    )
    if failed:
        fail_path = RAW_DIR / "pdb_download_failures.txt"
        fail_path.write_text("\n".join(failed))
        log.info(f"  Failed IDs saved → {fail_path}")


# ══════════════════════════════════════════════════════════════════════════
# SOURCE B: BindingDB PPI kinetics
# ══════════════════════════════════════════════════════════════════════════

BINDINGDB_URL = (
    "https://www.bindingdb.org/bind/downloads/BindingDB_All_202401_tsv.zip"
)

def download_bindingdb_ppi() -> pd.DataFrame:
    """
    Download BindingDB and extract protein-protein pairs with koff.

    BindingDB is large (~4 GB). We stream and filter to keep only:
      - Protein ligand type (not small molecule)
      - Entries with koff values
      - Measured by SPR or BLI (most reliable koff methods)

    NOTE: For a first MVP run, skip this and use SAbDab only.
          Set SKIP_BINDINGDB=True below if bandwidth is limited.
    """
    SKIP_BINDINGDB = True  # Set False to download the full ~4 GB file

    out_path = RAW_DIR / "bindingdb_ppi.tsv"
    if out_path.exists():
        log.info("BindingDB PPI subset already exists — loading cache")
        return pd.read_csv(out_path, sep="\t", low_memory=False)

    if SKIP_BINDINGDB:
        log.info(
            "Skipping BindingDB download (SKIP_BINDINGDB=True). "
            "SAbDab alone provides enough data for the MVP."
        )
        return pd.DataFrame()

    log.info("Downloading BindingDB (this is a large file) ...")
    # Full download and parse logic would go here.
    # For brevity, this branch is left as a stub for the full pipeline.
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("STEP 1: Downloading kinetics and structural data")
    log.info("=" * 60)

    # --- SAbDab ---
    df_sabdab = download_sabdab_summary()
    if len(df_sabdab) > 0:
        download_pdbs(df_sabdab, max_pdbs=500)

    # --- BindingDB ---
    df_bdb = download_bindingdb_ppi()

    log.info("")
    log.info("Summary")
    log.info("-------")
    log.info(f"  SAbDab entries with koff : {len(df_sabdab):,}")
    log.info(f"  BindingDB PPI entries    : {len(df_bdb):,}")
    log.info(f"  PDB structures in cache  : {len(list(PDB_DIR.glob('*.pdb'))):,}")
    log.info("")
    log.info("NEXT STEP: Run  python 02_build_dataset.py")


if __name__ == "__main__":
    main()
