"""
wrds_source.py - Compustat via WRDS as a survivorship-bias-free replacement for
the yfinance/Wikipedia data layer.

WHY. Yahoo serves no history for delisted tickers (11 of 11 removed S&P 500
names return nothing), so our panel covers 64% of true index membership and
only 61.5% back in 2015. Compustat Security Daily retains delisted companies
and carries an OPEN price, which is the one field the overnight/intraday
decomposition needs. `comp.idxcst_his` then gives index membership with real
from/thru dates, replacing the Wikipedia revision scrape.

WHAT THIS DOES NOT FIX. CRSP has proper delisting returns - what you actually
received when a company was removed. Compustat's daily file mostly just stops.
For a next-day study that matters less than for long horizons, but the final
observation of each delisted name is still wrong, and those are precisely the
observations that carry the bias. Treat this as most of the fix, not all of it.

CREDENTIALS. This module never handles your password. It connects with
psycopg2, which reads ~/.pgpass itself; you create that file once, yourself.
Nothing here reads, stores, prompts for or logs a credential.

    export WRDS_USERNAME=YOURUSER
    read -s "?WRDS password: " P
    P=${P//\\/\\\\}; P=${P//:/\\:}   # see the escaping note below
    printf 'wrds-pgdata.wharton.upenn.edu:9737:wrds:%s:%s\n' "$WRDS_USERNAME" "$P" >> ~/.pgpass
    chmod 600 ~/.pgpass          # postgres silently ignores it otherwise
    unset P

Two failure modes that produce identical, unhelpful errors:
  * A ~/.pgpass that is not chmod 600 is ignored WITHOUT WARNING, giving
    "fe_sendauth: no password supplied", the same message as having no file.
  * ':' and '\' are field separators in ~/.pgpass. An unescaped ':' in a
    password silently truncates it at the colon and the server reports
    "FATAL: PAM authentication failed", with nothing pointing at the file.
    Hence the two substitutions above; a letters-and-digits password avoids
    the issue entirely.

We deliberately do NOT use the official `wrds` package: it pins pandas<2.3,
which downgrades this repo's pandas 3.x and makes master_cache.pkl unreadable.
The package is a thin wrapper over psycopg2 plus read_sql, so we call those.

SCHEMA. Column and code assumptions are listed in ASSUMPTIONS below and are
CHECKED at runtime by probe() before any large query runs, because they are
written from documentation rather than from a live connection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRICES_OUT = REPO / "alpha" / "ohlc_compustat.parquet"
MEMB_OUT = REPO / "alpha" / "sp500_membership_compustat.parquet"

ASSUMPTIONS = {
    "comp.secd": ["gvkey", "iid", "datadate", "tic", "cusip", "prccd", "prcod",
                  "prchd", "prcld", "cshtrd", "ajexdi", "trfd", "curcdd", "exchg"],
    "comp.idxcst_his": ["gvkey", "iid", "gvkeyx", "from", "thru"],
    "comp.security": ["gvkey", "iid", "tic", "tpci", "exchg"],
}
# US listed exchanges in Compustat: 11 NYSE, 12 AMEX/NYSE American, 14 NASDAQ
US_EXCHANGES = (11, 12, 14)


# --------------------------------------------------------------- connection
WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"


def connect():
    """Open a WRDS connection using YOUR credentials from ~/.pgpass.

    No password is accepted, prompted for, or stored by this code: psycopg2
    resolves it from ~/.pgpass. Never called by the tests.
    """
    import os

    import psycopg2
    user = os.environ.get("WRDS_USERNAME", "").strip()
    if not user:
        raise RuntimeError(
            "set WRDS_USERNAME (and put the password in ~/.pgpass, chmod 600). "
            "See this module's docstring.")
    # 30s was too short to even surface the real failure: it reported
    # "timeout expired" where the server actually accepts TLS in 0.2s, sends
    # an auth request, then holds and drops. 120s lets the true error through.
    return psycopg2.connect(host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
                            user=user, sslmode="require", connect_timeout=120)


def raw_sql(conn, query: str) -> pd.DataFrame:
    """Run a query, return a DataFrame. Built from the cursor rather than via
    pandas.read_sql so no SQLAlchemy connectable is required."""
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def probe(db) -> pd.DataFrame:
    """Query-test each table. WRDS lists every product it SELLS in the menus,
    subscribed or not; entitlement only shows up when you actually query. This
    is that test, automated."""
    rows = []
    for table, cols in ASSUMPTIONS.items():
        lib, name = table.split(".")
        try:
            got = raw_sql(db, f"select * from {lib}.{name} limit 1")
            missing = [c for c in cols if c not in got.columns]
            rows.append({"table": table, "access": "YES",
                         "n_cols": len(got.columns),
                         "missing_expected_cols": ", ".join(missing) or "none"})
        except Exception as e:
            db.rollback()          # a failed query poisons the transaction
            msg = str(e).split("\n")[0][:90]
            rows.append({"table": table, "access": "NO", "n_cols": 0,
                         "missing_expected_cols": msg})
    # CRSP is the thing we actually want; test it too
    for table in ("crsp.dsf", "crsp.dsp500list"):
        lib, name = table.split(".")
        try:
            raw_sql(db, f"select * from {lib}.{name} limit 1")
            rows.append({"table": table, "access": "YES", "n_cols": -1,
                         "missing_expected_cols": "-"})
        except Exception as e:
            db.rollback()
            rows.append({"table": table, "access": "NO", "n_cols": 0,
                         "missing_expected_cols": str(e).split("\n")[0][:90]})
    return pd.DataFrame(rows)


def sp500_gvkeyx(db) -> str:
    """Find the S&P 500 index id by NAME rather than hardcoding it."""
    idx = raw_sql(db, "select gvkeyx, conm from comp.idx_index")
    hit = idx[idx["conm"].str.upper().str.contains("S&P 500", na=False)]
    exact = hit[hit["conm"].str.upper().str.strip().isin(
        ["S&P 500 COMP-LTD", "S&P 500 COMPOSITE", "S&P 500"])]
    chosen = (exact if len(exact) else hit).iloc[0]
    print(f"  index: gvkeyx={chosen['gvkeyx']}  {chosen['conm']}")
    return chosen["gvkeyx"]


# ------------------------------------------------------------ transformation
def to_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Compustat Security Daily -> the schema the rest of alpha/ already reads.

    Adjustment follows yfinance's auto_adjust convention so the two sources are
    interchangeable:
        adj_price = price / ajexdi * trfd
    The intraday leg is adjustment-invariant (both legs share the day's ajexdi
    and trfd); the overnight leg carries the corporate action, which is exactly
    the convention LPS adopt (p.196 n.9).
    """
    df = raw.rename(columns={"datadate": "Date", "tic": "Ticker"}).copy()
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    for c in ("prccd", "prcod", "prchd", "prcld", "cshtrd", "ajexdi", "trfd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["prccd"] > 0) & (df["prcod"] > 0) & (df["ajexdi"] > 0)]
    df["trfd"] = df["trfd"].fillna(1.0)

    f = df["trfd"] / df["ajexdi"]
    df["Close"] = df["prccd"] * f
    df["Open"] = df["prcod"] * f
    df["High"] = df["prchd"] * f
    df["Low"] = df["prcld"] * f
    df["Volume"] = df["cshtrd"]

    df = (df.sort_values(["Ticker", "Date"])
            .drop_duplicates(["Ticker", "Date"], keep="last")
            .reset_index(drop=True))
    g = df.groupby("Ticker", observed=True)
    prev_close = g["Close"].shift(1)

    cal = pd.Index(sorted(df["Date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    dpos = df["Date"].map(pos)
    contiguous = (dpos - dpos.groupby(df["Ticker"]).shift(1)) == 1

    df["r_cc"] = df["Close"] / prev_close - 1
    df["r_intraday"] = df["Close"] / df["Open"] - 1
    df["r_overnight"] = df["Open"] / prev_close - 1
    df.loc[~contiguous, ["r_cc", "r_overnight"]] = np.nan

    chk = ((1 + df["r_overnight"]) * (1 + df["r_intraday"]) - 1 - df["r_cc"]).abs()
    df.attrs["max_decomposition_error"] = float(chk.max()) if chk.notna().any() else 0.0
    keep = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
            "r_cc", "r_intraday", "r_overnight", "gvkey", "iid"]
    return df[[c for c in keep if c in df.columns]]


def membership_to_panel(raw: pd.DataFrame, freq: str = "QE") -> pd.DataFrame:
    """comp.idxcst_his (from/thru spans) -> the (asof, Ticker) snapshot schema
    that pit_analysis.py already consumes. A NULL `thru` means still a member."""
    df = raw.copy()
    df["from"] = pd.to_datetime(df["from"]).astype("datetime64[ns]")
    df["thru"] = pd.to_datetime(df["thru"]).astype("datetime64[ns]")
    end = pd.Timestamp.today().normalize()
    df["thru"] = df["thru"].fillna(end)
    dates = pd.date_range(df["from"].min().normalize(), end, freq=freq)
    out = []
    for d in dates:
        live = df[(df["from"] <= d) & (df["thru"] >= d)]
        out.append(pd.DataFrame({"asof": d, "Ticker": live["Ticker"].dropna().unique()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["asof", "Ticker"])


# ------------------------------------------------------------------- pulls
def pull_prices(db, start="2014-12-01", end=None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    q = f"""
        select d.gvkey, d.iid, d.datadate, d.tic, d.cusip,
               d.prccd, d.prcod, d.prchd, d.prcld, d.cshtrd, d.ajexdi, d.trfd
        from comp.secd as d
        where d.datadate between '{start}' and '{end}'
          and d.curcdd = 'USD'
          and d.exchg in {US_EXCHANGES}
          and d.prccd is not null and d.prcod is not null
          and d.tic is not null
    """
    print("  querying comp.secd (large; several minutes) ...", flush=True)
    return to_panel(raw_sql(db, q))


def pull_constituents(db, gvkeyx: str) -> pd.DataFrame:
    q = f"""
        select c.gvkey, c.iid, c."from", c.thru, s.tic as "Ticker"
        from comp.idxcst_his as c
        left join comp.security as s on s.gvkey = c.gvkey and s.iid = c.iid
        where c.gvkeyx = '{gvkeyx}'
    """
    return membership_to_panel(raw_sql(db, q))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "prices", "constituents", "all"])
    ap.add_argument("--start", default="2014-12-01")
    a = ap.parse_args()

    db = connect()
    try:
        if a.cmd in ("probe", "all"):
            r = probe(db)
            print("\n=== entitlement check (query-tested, not menu-read) ===")
            print(r.to_string(index=False))
            if a.cmd == "probe":
                sys.exit(0)
        if a.cmd in ("prices", "all"):
            p = pull_prices(db, a.start)
            p.to_parquet(PRICES_OUT, index=False)
            print(f"prices: rows={len(p):,} tickers={p.Ticker.nunique():,} "
                  f"decomposition error={p.attrs['max_decomposition_error']:.1e}")
            print(f"wrote {PRICES_OUT}")
        if a.cmd in ("constituents", "all"):
            m = pull_constituents(db, sp500_gvkeyx(db))
            m.to_parquet(MEMB_OUT, index=False)
            n = m.groupby("asof")["Ticker"].size()
            print(f"membership: snapshots={n.size} members {n.min()}-{n.max()} "
                  f"distinct ever={m.Ticker.nunique()}")
            print(f"wrote {MEMB_OUT}")
    finally:
        db.close()
