"""
Horizon Ledger — Fundamentals Pipeline
Point-in-time fundamentals from SEC EDGAR via edgartools.
Enforces: only use data where filing_date <= as_of_date - LAG_DAYS
to avoid look-ahead bias.

References:
  - Sloan (1996): Accruals and stock returns — importance of filing dates
  - Leinweber (2007): Stupid data miner tricks — point-in-time discipline
"""

import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from config import EDGAR_FUNDAMENTALS_LAG_DAYS
from db.schema import get_connection
from db.queries import get_stock_id, upsert_fundamental, upsert_stock

log = logging.getLogger(__name__)


# ─── EDGAR via edgartools ─────────────────────────────────────────────────────

def fetch_edgar_fundamentals(ticker: str, cik: Optional[str] = None) -> list[dict]:
    """
    Use edgartools to pull XBRL financial data for a single ticker.
    Returns a list of quarterly fundamental dicts with filing_date set.
    """
    try:
        from edgar import Company
        company = Company(ticker) if cik is None else Company(cik)
        filings = company.get_filings(form="10-Q").head(12)
        if filings is None or len(filings) == 0:
            filings = company.get_filings(form="10-K").head(5)

        rows = []
        for filing in filings:
            try:
                financials = filing.financials
                if financials is None:
                    continue

                inc = _safe_income(financials)
                bal = _safe_balance(financials)
                csh = _safe_cashflow(financials)

                period_end = _to_date_str(filing.period_of_report)
                filed_date = _to_date_str(filing.filing_date)

                if period_end is None or filed_date is None:
                    continue

                ocf = _v(csh, "NetCashProvidedByUsedInOperatingActivities")
                capex_raw = _v(csh, "PaymentsToAcquirePropertyPlantAndEquipment")
                capex = -abs(capex_raw) if capex_raw is not None else None
                fcf = (ocf + capex) if (ocf is not None and capex is not None) else None

                revenue = _v(inc, "Revenues") or _v(inc, "RevenueFromContractWithCustomerExcludingAssessedTax")
                gross_profit = _v(inc, "GrossProfit")
                ebit = _compute_ebit(inc)
                net_income = _v(inc, "NetIncomeLoss")

                total_assets = _v(bal, "Assets")
                current_assets = _v(bal, "AssetsCurrent")
                current_liabilities = _v(bal, "LiabilitiesCurrent")
                total_equity = (
                    _v(bal, "StockholdersEquity")
                    or _v(bal, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
                )
                total_debt = (
                    (_v(bal, "LongTermDebt") or 0) +
                    (_v(bal, "ShortTermBorrowings") or 0)
                )
                cash = (
                    _v(bal, "CashAndCashEquivalentsAtCarryingValue")
                    or _v(bal, "CashCashEquivalentsAndShortTermInvestments")
                )
                dividends = _v(csh, "PaymentsOfDividends")
                shares = (
                    _v(bal, "CommonStockSharesOutstanding")
                    or _v(inc, "WeightedAverageNumberOfSharesOutstandingBasic")
                )
                eps = _v(inc, "EarningsPerShareBasic")
                bvps = (total_equity / shares) if (total_equity and shares and shares > 0) else None

                quarter = _fiscal_quarter(period_end)
                fy = int(period_end[:4]) if period_end else None

                rows.append({
                    "report_date": period_end,
                    "filing_date": filed_date,
                    "fiscal_year": fy,
                    "fiscal_quarter": quarter,
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "ebit": ebit,
                    "net_income": net_income,
                    "total_assets": total_assets,
                    "total_debt": total_debt,
                    "total_equity": total_equity,
                    "current_assets": current_assets,
                    "current_liabilities": current_liabilities,
                    "cash": cash,
                    "operating_cash_flow": ocf,
                    "capex": capex,
                    "free_cash_flow": fcf,
                    "dividends_paid": dividends,
                    "shares_outstanding": shares,
                    "eps": eps,
                    "book_value_per_share": bvps,
                    "data_source": "edgar",
                })
            except Exception as e:
                log.debug("Error parsing filing for %s: %s", ticker, e)

        return rows
    except Exception as e:
        log.warning("edgartools failed for %s: %s — trying yfinance fallback", ticker, e)
        return fetch_yfinance_fundamentals(ticker)


def fetch_yfinance_fundamentals(ticker: str) -> list[dict]:
    """
    Fallback: pull fundamentals from yfinance quarterly financials.
    Filing dates are approximated as report_date + 45 days (conservative).
    ⚠️  These are NOT point-in-time — use only as a fallback.
    """
    try:
        t = yf.Ticker(ticker)
        inc = t.quarterly_income_stmt
        bal = t.quarterly_balance_sheet
        csh = t.quarterly_cashflow

        rows = []
        if inc is None or inc.empty:
            return []

        for col in inc.columns:
            try:
                report_date = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
                # Approximate filing date (not point-in-time — documented limitation)
                filing_date = (
                    date.fromisoformat(report_date) + timedelta(days=45)
                ).isoformat()

                def _get(df, *keys):
                    for k in keys:
                        try:
                            v = df.loc[k, col]
                            if pd.notna(v):
                                return float(v)
                        except Exception:
                            pass
                    return None

                revenue = _get(inc, "Total Revenue")
                gross_profit = _get(inc, "Gross Profit")
                ebit = _get(inc, "EBIT", "Operating Income")
                net_income = _get(inc, "Net Income")
                total_assets = _get(bal, "Total Assets")
                total_equity = _get(bal, "Stockholders Equity", "Total Stockholder Equity")
                total_debt = _get(bal, "Total Debt", "Long Term Debt")
                current_assets = _get(bal, "Current Assets")
                current_liabilities = _get(bal, "Current Liabilities")
                cash = _get(bal, "Cash And Cash Equivalents")
                ocf = _get(csh, "Operating Cash Flow", "Total Cash From Operating Activities")
                capex_raw = _get(csh, "Capital Expenditure")
                capex = -abs(capex_raw) if capex_raw is not None else None
                fcf = (ocf + capex) if (ocf and capex is not None) else None
                dividends = _get(csh, "Common Stock Dividends Paid", "Dividends Paid")
                shares = _get(bal, "Share Issued", "Ordinary Shares Number")
                eps = _get(inc, "Basic EPS")
                bvps = (total_equity / shares) if (total_equity and shares and shares > 0) else None

                quarter = _fiscal_quarter(report_date)
                fy = int(report_date[:4])

                rows.append({
                    "report_date": report_date,
                    "filing_date": filing_date,
                    "fiscal_year": fy,
                    "fiscal_quarter": quarter,
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "ebit": ebit,
                    "net_income": net_income,
                    "total_assets": total_assets,
                    "total_debt": total_debt,
                    "total_equity": total_equity,
                    "current_assets": current_assets,
                    "current_liabilities": current_liabilities,
                    "cash": cash,
                    "operating_cash_flow": ocf,
                    "capex": capex,
                    "free_cash_flow": fcf,
                    "dividends_paid": dividends,
                    "shares_outstanding": shares,
                    "eps": eps,
                    "book_value_per_share": bvps,
                    "data_source": "yfinance_approx",
                })
            except Exception as e:
                log.debug("yfinance fundamental row error for %s: %s", ticker, e)

        return rows
    except Exception as e:
        log.error("yfinance fundamentals failed for %s: %s", ticker, e)
        return []


# ─── Store ────────────────────────────────────────────────────────────────────

def store_fundamentals(ticker: str, rows: list[dict]) -> int:
    """Persist a list of fundamental dicts to the DB. Returns count stored."""
    if not rows:
        return 0
    conn = get_connection()
    stock_id = get_stock_id(conn, ticker)
    if stock_id is None:
        upsert_stock(conn, ticker, is_active=1)
        stock_id = get_stock_id(conn, ticker)
    stored = 0
    with conn:
        for row in rows:
            try:
                upsert_fundamental(conn, stock_id, row)
                stored += 1
            except Exception as e:
                log.debug("Store error for %s row %s: %s", ticker, row.get("report_date"), e)
        conn.commit()
    conn.close()
    return stored


def update_fundamentals_for_universe() -> None:
    """Update fundamentals for all active tickers."""
    from db.queries import get_active_universe
    conn = get_connection()
    universe = get_active_universe(conn)
    conn.close()

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        cik = row.get("cik")
        log.info("Fetching fundamentals for %s...", ticker)
        rows = fetch_edgar_fundamentals(ticker, cik)
        n = store_fundamentals(ticker, rows)
        log.info("  → %d rows stored for %s", n, ticker)
        time.sleep(0.3)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _v(obj, key: str) -> Optional[float]:
    """Safely extract a value from an edgar financials object."""
    if obj is None:
        return None
    try:
        val = getattr(obj, key, None)
        if val is None:
            return None
        if hasattr(val, "value"):
            return float(val.value)
        return float(val)
    except Exception:
        return None


def _safe_income(financials):
    try:
        return financials.income_statement
    except Exception:
        return None


def _safe_balance(financials):
    try:
        return financials.balance_sheet
    except Exception:
        return None


def _safe_cashflow(financials):
    try:
        return financials.cash_flow_statement
    except Exception:
        return None


def _compute_ebit(inc) -> Optional[float]:
    ebit = _v(inc, "OperatingIncomeLoss")
    if ebit is not None:
        return ebit
    revenue = _v(inc, "Revenues") or _v(inc, "RevenueFromContractWithCustomerExcludingAssessedTax")
    cogs = _v(inc, "CostOfGoodsAndServicesSold") or _v(inc, "CostOfRevenue")
    sga = _v(inc, "SellingGeneralAndAdministrativeExpense")
    if revenue and cogs:
        return revenue - cogs - (sga or 0)
    return None


def _to_date_str(d) -> Optional[str]:
    if d is None:
        return None
    try:
        if isinstance(d, str):
            return d[:10]
        return d.strftime("%Y-%m-%d")
    except Exception:
        return None


def _fiscal_quarter(report_date: str) -> int:
    """Infer fiscal quarter from period end month."""
    try:
        m = int(report_date[5:7])
        return ((m - 1) // 3) + 1
    except Exception:
        return 0
