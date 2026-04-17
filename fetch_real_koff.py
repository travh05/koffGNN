"""
fetch_real_koff.py — Collect REAL measured koff values
=======================================================
Two sources:

  Source A: SAbDab HTML scraping
    Each PDB entry page still shows kon/koff/Kd in an HTML table
    even though the TSV API dropped those columns.
    We scrape all 734 PDBs we already downloaded.

  Source B: BindingDB REST API
    Query by target UniProt IDs for protein-protein entries
    with measured koff (SPR/BLI method).
    Fetches additional complexes beyond SAbDab.

Run:
    python fetch_real_koff.py

Output:
    data/raw/real_koff.tsv        — combined, deduplicated
    data/raw/sabdab_summary.tsv   — overwritten with real koff only
"""

import re
import time
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PDB_DIR = RAW_DIR / "pdb"

# ══════════════════════════════════════════════════════════════════════════
# SOURCE A: SAbDab HTML scraping
# ══════════════════════════════════════════════════════════════════════════

SABDAB_ENTRY_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{}/"

def parse_koff_from_html(html: str) -> list[dict]:
    """
    Parse the kinetics table from a SAbDab PDB entry HTML page.
    Returns list of dicts with keys: koff_s, kon, kd, method, hchain, antigen_chain
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # SAbDab renders a table with headers including Koff, Kon, Kd
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not any("koff" in h for h in headers):
            continue

        # Map column index to field name
        col_map = {}
        for i, h in enumerate(headers):
            if "koff" in h:
                col_map["koff"] = i
            elif "kon" in h:
                col_map["kon"] = i
            elif "kd" in h or "affinity" in h:
                col_map["kd"] = i
            elif "method" in h:
                col_map["method"] = i
            elif "hchain" in h or "heavy" in h:
                col_map["hchain"] = i
            elif "antigen" in h and "chain" in h:
                col_map["antigen_chain"] = i

        if "koff" not in col_map:
            continue

        for row in table.find_all("tr")[1:]:  # skip header
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells or len(cells) <= col_map["koff"]:
                continue

            koff_raw = cells[col_map["koff"]]
            koff_val = _parse_float(koff_raw)
            if koff_val is None:
                continue

            entry = {
                "koff_s":        koff_val,
                "kon":           _parse_float(cells[col_map["kon"]]) if "kon" in col_map else None,
                "kd":            _parse_float(cells[col_map["kd"]]) if "kd" in col_map else None,
                "method":        cells[col_map["method"]] if "method" in col_map else "",
                "hchain":        cells[col_map["hchain"]] if "hchain" in col_map else "",
                "antigen_chain": cells[col_map["antigen_chain"]] if "antigen_chain" in col_map else "",
            }
            results.append(entry)

    return results


def _parse_float(s: str):
    """Extract first float from a string like '1.2e-4' or '0.0012 ± 0.0001'."""
    if not s or s in ("-", "N/A", "n/a", "None", "nan"):
        return None
    # Handle scientific notation with various formats
    s = s.replace(",", ".").replace("×10", "e").replace("x10", "e")
    s = re.sub(r"\s*[±+\-]\s*[\d.eE+\-]+$", "", s)  # strip ± uncertainty
    match = re.search(r"[-+]?\d+\.?\d*[eE]?[-+]?\d*", s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def scrape_sabdab_koff(pdb_ids: list[str], delay: float = 0.15) -> pd.DataFrame:
    """
    Scrape koff from SAbDab HTML pages for a list of PDB IDs.
    Returns DataFrame with columns: pdb, koff_s, kon, kd, method, hchain, antigen_chain
    """
    rows = []
    failed = 0

    log.info(f"Scraping SAbDab HTML for {len(pdb_ids)} PDB entries...")

    for i, pdb_id in enumerate(pdb_ids):
        url = SABDAB_ENTRY_URL.format(pdb_id.lower())
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                failed += 1
                continue

            entries = parse_koff_from_html(r.text)
            for e in entries:
                e["pdb"] = pdb_id.lower()
                rows.append(e)

        except Exception as ex:
            log.debug(f"  {pdb_id}: {ex}")
            failed += 1

        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(pdb_ids)}] found {len(rows)} koff entries, {failed} failed")

        time.sleep(delay)

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pdb", "koff_s", "kon", "kd", "method", "hchain", "antigen_chain"]
    )
    log.info(f"SAbDab scrape complete: {len(df)} entries with koff")
    return df


# ══════════════════════════════════════════════════════════════════════════
# SOURCE B: BindingDB REST API
# ══════════════════════════════════════════════════════════════════════════

# Key antibody/cytokine target UniProt IDs to query
# These are well-characterised PPI targets with known SPR koff in BindingDB
UNIPROT_TARGETS = [
    "P01375",  # TNF-alpha
    "P60568",  # IL-2
    "P05106",  # Integrin beta-3
    "P00533",  # EGFR
    "Q15116",  # PD-1
    "Q9NZQ7",  # PD-L1
    "P01584",  # IL-1 beta
    "P05155",  # C1-inhibitor
    "P01579",  # IFN-gamma
    "P10145",  # IL-8
    "P05113",  # HER2/ERBB2 (partial)
    "P04626",  # HER2
    "P06213",  # Insulin receptor
    "P01308",  # Insulin
    "P35968",  # VEGFR2
    "P15692",  # VEGF-A
]

BINDINGDB_API = "https://www.bindingdb.org/axis2/services/BDBService"


def fetch_bindingdb_koff(uniprot_id: str) -> list[dict]:
    """
    Query BindingDB REST API for kinetics data for a given UniProt target.
    Filters for protein ligands (not small molecules) with koff values.
    """
    url = (
        f"{BINDINGDB_API}/getLigandsByUniprot"
        f"?uniprot={uniprot_id}&response=json"
    )
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    affinities = data.get("affinities", []) if isinstance(data, dict) else []
    rows = []

    for entry in affinities:
        koff_raw = entry.get("koff") or entry.get("koff_s") or entry.get("Koff")
        if not koff_raw or str(koff_raw).strip() in ("", "-", "N/A", "None"):
            continue

        koff_val = _parse_float(str(koff_raw))
        if koff_val is None or not (1e-7 < koff_val < 1e2):
            continue

        # Try to get PDB ID
        pdb_id = (entry.get("pdbid") or entry.get("pdb") or "").lower().strip()

        rows.append({
            "pdb":            pdb_id if len(pdb_id) == 4 else "",
            "koff_s":         koff_val,
            "kon":            _parse_float(str(entry.get("kon", "") or "")),
            "kd":             _parse_float(str(entry.get("kd", "") or entry.get("ki", "") or "")),
            "method":         str(entry.get("assay_type", "") or ""),
            "hchain":         "",
            "antigen_chain":  "",
            "source":         "bindingdb",
            "uniprot":        uniprot_id,
            "ligand_name":    str(entry.get("ligandname", "") or ""),
        })

    return rows


def fetch_all_bindingdb(uniprot_ids: list[str]) -> pd.DataFrame:
    log.info(f"Querying BindingDB for {len(uniprot_ids)} UniProt targets...")
    all_rows = []

    for i, uid in enumerate(uniprot_ids):
        rows = fetch_bindingdb_koff(uid)
        all_rows.extend(rows)
        log.info(f"  {uid}: {len(rows)} koff entries")
        time.sleep(0.3)

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    log.info(f"BindingDB total: {len(df)} entries with koff")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Download additional PDBs for BindingDB entries
# ══════════════════════════════════════════════════════════════════════════

def download_missing_pdbs(df: pd.DataFrame, max_new: int = 300) -> int:
    """Download PDB files for entries in df that we don't have yet."""
    if "pdb" not in df.columns:
        return 0

    needed = df[df["pdb"].str.len() == 4]["pdb"].str.lower().unique()
    have   = {p.stem.lower() for p in PDB_DIR.glob("*.pdb")}
    to_get = [p for p in needed if p not in have][:max_new]

    if not to_get:
        log.info("No new PDBs to download")
        return 0

    log.info(f"Downloading {len(to_get)} new PDB structures...")
    got, failed = 0, 0

    for i, pdb_id in enumerate(to_get):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                (PDB_DIR / f"{pdb_id}.pdb").write_bytes(r.content)
                got += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(to_get)}] got={got} failed={failed}")
        time.sleep(0.05)

    log.info(f"  Downloaded {got} new PDBs ({failed} failed)")
    return got


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Fetching REAL measured koff data")
    log.info("=" * 60)

    # Check BeautifulSoup is available
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        import subprocess, sys
        log.info("Installing beautifulsoup4...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])

    # ── SOURCE A: SAbDab HTML scraping ────────────────────────────────
    pdb_ids = [p.stem for p in PDB_DIR.glob("*.pdb")]
    df_sabdab = scrape_sabdab_koff(pdb_ids)

    # ── SOURCE B: BindingDB ────────────────────────────────────────────
    df_bdb = fetch_all_bindingdb(UNIPROT_TARGETS)

    # ── Combine ────────────────────────────────────────────────────────
    frames = []
    if len(df_sabdab) > 0:
        df_sabdab["source"] = "sabdab_html"
        frames.append(df_sabdab)
    if len(df_bdb) > 0:
        frames.append(df_bdb)

    if not frames:
        log.error("No real koff data found from either source.")
        log.info("Falling back to Kd-derived estimates in existing sabdab_summary.tsv")
        return

    df_combined = pd.concat(frames, ignore_index=True)

    # Filter to physical range
    df_combined["koff_s"] = pd.to_numeric(df_combined["koff_s"], errors="coerce")
    df_combined = df_combined[
        df_combined["koff_s"].notna() &
        (df_combined["koff_s"] > 1e-7) &
        (df_combined["koff_s"] < 1e2)
    ]

    # Save full combined set
    combined_path = RAW_DIR / "real_koff.tsv"
    df_combined.to_csv(combined_path, sep="\t", index=False)
    log.info(f"Saved {len(df_combined)} entries → {combined_path}")

    # ── Download new PDBs for BindingDB entries ────────────────────────
    if len(df_bdb) > 0:
        download_missing_pdbs(df_bdb, max_new=300)

    # ── Build sabdab_summary.tsv with real koff ────────────────────────
    # Match to PDBs we actually have
    have_pdbs = {p.stem.lower() for p in PDB_DIR.glob("*.pdb")}
    df_out = df_combined[df_combined["pdb"].isin(have_pdbs)].copy()

    # For entries without hchain, use the raw summary as fallback
    raw = pd.read_csv(RAW_DIR / "sabdab_summary_raw.tsv", sep="\t", low_memory=False)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["pdb"] = raw["pdb"].astype(str).str.lower().str.strip()
    chain_map = raw.set_index("pdb")[["hchain", "antigen_chain"]].to_dict("index")

    def fill_chains(row):
        if row.get("hchain") or row.get("antigen_chain"):
            return row
        info = chain_map.get(row["pdb"], {})
        row["hchain"]        = info.get("hchain", "H")
        row["antigen_chain"] = info.get("antigen_chain", "A")
        return row

    df_out = df_out.apply(fill_chains, axis=1)

    # Rename to match what 02_build_dataset.py expects
    df_final = df_out[["pdb", "hchain", "antigen_chain", "koff_s"]].copy()
    df_final.columns = ["pdb", "hchain", "antigen_chain", "koff_s"]
    df_final = df_final.drop_duplicates(subset=["pdb"])

    df_final.to_csv(RAW_DIR / "sabdab_summary.tsv", sep="\t", index=False)

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS")
    log.info(f"  SAbDab HTML entries  : {len(df_sabdab)}")
    log.info(f"  BindingDB entries    : {len(df_bdb)}")
    log.info(f"  With matching PDB    : {len(df_final)}")
    log.info(f"  log10(koff) mean     : {np.log10(df_final['koff_s']).mean():.2f}")
    log.info(f"  log10(koff) std      : {np.log10(df_final['koff_s']).std():.2f}")
    log.info("")

    if len(df_final) >= 50:
        log.info("  Enough real koff data collected.")
        log.info("  Delete the old processed dataset and rebuild:")
        log.info("    rm -rf data/processed/processed/")
        log.info("    python 02_build_dataset.py")
        log.info("    python 03_train.py --epochs 150 --hidden_dim 256")
    else:
        log.info(f"  Only {len(df_final)} entries with matching PDB.")
        log.info("  SAbDab may have changed their HTML structure.")
        log.info("  Check data/raw/real_koff.tsv for what was found.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
