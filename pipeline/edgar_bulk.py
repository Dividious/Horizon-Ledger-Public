"""
Horizon Ledger — SEC EDGAR Bulk Data Download & Parse
Downloads company facts from SEC EDGAR for point-in-time fundamentals.
Uses the SEC's public XBRL API (no authentication required).

Rate limit: SEC requests ≤ 10 req/sec.  We stay well under this.
User-Agent header is required by SEC fair access policy.
"""

import json
import logging
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

from config import BASE_DIR, EDGAR_COMPANYFACTS_URL, EDGAR_SUBMISSIONS_URL
from db.schema import get_connection
from db.queries import get_active_universe, get_stock_id, upsert_fundamental, upsert_stock

log = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data" / "edgar"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEC_HEADERS = {
    "User-Agent": "HorizonLedger research@local.dev",
    "Accept-Encoding": "gzip, deflate",
}

REQUEST_DELAY = 0.15   # seconds between requests — stay under 10/sec


def fetch_company_facts(cik: str) -> Optional[dict]:
    """
    Fetch XBRL company facts JSON for a single CIK from SEC EDGAR.
    Returns the parsed JSON dict or None on failure.
    """
    cik_padded = str(cik).zfill(10)
    url = f"{EDGAR_COMPANYFACTS_URL}CIK{cik_padded}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        if resp.status_code == 404:
            log.debug("No EDGAR facts for CIK %s", cik_padded)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("EDGAR fetch failed for CIK %s: %s", cik_padded, e)
        return None


def parse_company_facts(facts: dict) -> list[dict]:
    """
    Parse SEC company facts JSON into a list of quarterly fundamental dicts.
    Extracts the key XBRL concepts needed for scoring factors.

    SEC XBRL concepts used:
      us-gaap:Revenues, us-gaap:GrossProfit, us-gaap:OperatingIncomeLoss,
      us-gaap:NetIncomeLoss, us-gaap:Assets, us-gaap:Liabilities,
      us-gaap:StockholdersEquity, us-gaap:CashAndCashEquivalents,
      us-gaap:NetCashProvidedByUsedInOperatingActivities,
      us-gaap:PaymentsToAcquirePropertyPlantAndEquipment,
      us-gaap:CommonStockDividendsPaid, us-gaap:SharesOutstanding, etc.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return []

    def _extract(concept: str, unit: str = "USD") -> pd.DataFrame:
        """Extract all filed values for a concept, return as DataFrame."""
        data = us_gaap.get(concept, {}).get("units", {}).get(unit, [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "filed" not in df.columns:
            return pd.DataFrame()
        # Keep only 10-K and 10-Q form types
        df = df[df.get("form", pd.Series(dtype=str)).isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])]
        df = df.dropna(subset=["end", "filed", "val"])
        df["end"] = df["end"].astype(str)
        df["filed"] = df["filed"].astype(str)
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        return df[["end", "filed", "val", "form"]].drop_duplicates(subset=["end"])

    def _latest_val(df: pd.DataFrame, period_end: str) -> Optional[float]:
        """Get value for a specific period end date, preferring most recent filing."""
        if df.empty:
            return None
        row = df[df["end"] == period_end].sort_values("filed").iloc[-1:]["val"]
        return float(row.iloc[0]) if not row.empty else None

    # Extract all concepts
    revenues       = _extract("Revenues")
    if revenues.empty:
        revenues   = _extract("RevenueFromContractWithCustomerExcludingAssessedTax")
    gross_profit   = _extract("GrossProfit")
    ebit           = _extract("OperatingIncomeLoss")
    net_income     = _extract("NetIncomeLoss")
    total_assets   = _extract("Assets")
    total_liab     = _extract("Liabilities")
    equity         = _extract("StockholdersEquity")
    if equity.empty:
        equity     = _extract("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    cur_assets     = _extract("AssetsCurrent")
    cur_liab       = _extract("LiabilitiesCurrent")
    cash           = _extract("CashAndCashEquivalentsAtCarryingValue")
    ocf            = _extract("NetCashProvidedByUsedInOperatingActivities")
    capex          = _extract("PaymentsToAcquirePropertyPlantAndEquipment")
    lt_debt        = _extract("LongTermDebt")
    dividends      = _extract("PaymentsOfDividends")
    shares         = _extract("CommonStockSharesOutstanding", unit="shares")
    if shares.empty:
        shares     = _extract("EntityCommonStockSharesOutstanding", unit="shares")
    eps            = _extract("EarningsPerShareBasic", unit="USD/shares")

    # Get all unique period end dates from the most data-rich series
    anchor = revenues if not revenues.empty else (total_assets if not total_assets.empty else pd.DataFrame())
    if anchor.empty:
        return []

    rows = []
    for _, anchor_row in anchor.iterrows():
        period_end = anchor_row["end"]
        filing_date = anchor_row["filed"]

        rev = _latest_val(revenues, period_end)
        gp  = _latest_val(gross_profit, period_end)
        op  = _latest_val(ebit, period_end)
        ni  = _latest_val(net_income, period_end)
        ta  = _latest_val(total_assets, period_end)
        eq  = _latest_val(equity, period_end)
        ca  = _latest_val(cur_assets, period_end)
        cl  = _latest_val(cur_liab, period_end)
        csh = _latest_val(cash, period_end)
        cf  = _latest_val(ocf, period_end)
        cpx_raw = _latest_val(capex, period_end)
        cpx = -abs(cpx_raw) if cpx_raw is not None else None
        fcf = (cf + cpx) if (cf is not None and cpx is not None) else None
        ltd = _latest_val(lt_debt, period_end)
        div = _latest_val(dividends, period_end)
        shr = _latest_val(shares, period_end)
        ep  = _latest_val(eps, period_end)
        bvps = (eq / shr) if (eq and shr and shr > 0) else None

        # Infer total debt
        total_debt_val = ltd  # simplified — can add short-term borrowings

        # Fiscal quarter from period end month
        try:
            m = int(period_end[5:7])
            q = ((m - 1) // 3) + 1
            fy = int(period_end[:4])
        except Exception:
            q = 0
            fy = None

        rows.append({
            "report_date":         period_end,
            "filing_date":         filing_date,
            "fiscal_year":         fy,
            "fiscal_quarter":      q,
            "revenue":             rev,
            "gross_profit":        gp,
            "ebit":                op,
            "net_income":          ni,
            "total_assets":        ta,
            "total_debt":          total_debt_val,
            "total_equity":        eq,
            "current_assets":      ca,
            "current_liabilities": cl,
            "cash":                csh,
            "operating_cash_flow": cf,
            "capex":               cpx,
            "free_cash_flow":      fcf,
            "dividends_paid":      div,
            "shares_outstanding":  shr,
            "eps":                 ep,
            "book_value_per_share":bvps,
            "data_source":         "edgar_bulk",
        })

    return rows


def update_fundamentals_from_edgar(tickers: Optional[list[str]] = None) -> None:
    """
    Pull company facts from EDGAR for each ticker with a known CIK
    and upsert into the fundamentals table.
    """
    conn = get_connection()
    universe = get_active_universe(conn)
    conn.close()

    if tickers is not None:
        universe = universe[universe["ticker"].isin(tickers)]

    # Filter to rows that have a CIK
    universe = universe[universe["cik"].notna() & (universe["cik"] != "")]
    log.info("Updating EDGAR fundamentals for %d tickers with CIK...", len(universe))

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        cik    = row["cik"]
        log.info("  EDGAR: %s (CIK %s)", ticker, cik)

        facts = fetch_company_facts(cik)
        if facts is None:
            continue

        rows = parse_company_facts(facts)
        if not rows:
            log.debug("  No parsed rows for %s", ticker)
            continue

        conn = get_connection()
        stock_id = get_stock_id(conn, ticker)
        if stock_id is None:
            upsert_stock(conn, ticker, is_active=1)
            stock_id = get_stock_id(conn, ticker)
        with conn:
            stored = 0
            for r in rows:
                try:
                    upsert_fundamental(conn, stock_id, r)
                    stored += 1
                except Exception as e:
                    log.debug("  Store error %s %s: %s", ticker, r.get("report_date"), e)
            conn.commit()
        conn.close()
        log.info("  → %d fundamental rows stored for %s", stored, ticker)
        time.sleep(REQUEST_DELAY)

    log.info("EDGAR fundamentals update complete.")


def download_and_save_facts(cik: str, ticker: str) -> Optional[Path]:
    """Download and save raw company facts JSON to data/edgar/ for inspection."""
    facts = fetch_company_facts(cik)
    if facts is None:
        return None
    out_path = DATA_DIR / f"{ticker}_facts.json"
    out_path.write_text(json.dumps(facts, indent=2))
    log.info("Saved facts for %s to %s", ticker, out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from db.schema import init_db
    init_db()
    update_fundamentals_from_edgar()
