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


def has_crsp(db) -> bool:
    """Query-tested, not assumed: the pre-registration picks the source on
    entitlement, and WRDS lists products it does not grant."""
    try:
        raw_sql(db, "select permno from crsp.dsf limit 1")
        return True
    except Exception:
        db.rollback()
        return False


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


# -------------------------------------------------------------------- CRSP
# The pre-registration mandates CRSP where entitled, because it carries true
# delisting returns. It also lets us use LPS's OWN definition of the overnight
# leg (p.196) rather than reconstructing it from adjusted prices:
#
#     r_intraday,s  = Close_s / Open_s - 1
#     r_overnight,s = (1 + r_close_to_close,s) / (1 + r_intraday,s) - 1
#
# crsp.dsf.ret is a total return including dividends, and merged with
# crsp.dsedelist it includes the delisting return. So the identity
# (1+overnight)(1+intraday) = (1+cc) holds by construction, and dividends and
# corporate actions land in the overnight leg, which is what the paper assumes.
#
# Identity is permno, not ticker. Tickers are reused after a delisting, which
# would silently graft one company's history onto another - exactly the error
# this whole exercise exists to remove.
# NO share-code or exchange filter. The frozen universe is "point-in-time
# S&P 500 members", and membership IS the definition -- filtering it further
# deletes real constituents. The usual shrcd in (10,11) screen drops 51
# foreign-incorporated members (shrcd 12: Linde, Medtronic, Aon) and 38 REITs
# (shrcd 18: Simon Property, Prologis, American Tower), which is what failed
# the point-in-time coverage gate at 0.886 on the first attempt.


def crsp_sp500_permnos(db, start: str) -> str:
    """Permnos ever in the S&P 500 since `start`, as a SQL list. We pull their
    FULL history, not just their member-days, because the frozen feature set
    needs trailing windows before a name enters the index."""
    q = f"""select distinct permno from crsp.dsp500list
            where ending >= '{start}' or ending is null"""
    ids = raw_sql(db, q)["permno"].astype(int).tolist()
    if not ids:
        raise RuntimeError("crsp.dsp500list returned no permnos")
    print(f"  {len(ids)} permnos ever in the S&P 500 since {start}")
    return ",".join(str(i) for i in ids)


def pull_prices_crsp(db, start="2014-12-01", end=None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    permnos = crsp_sp500_permnos(db, start)
    q = f"""
        select d.permno, d.date, d.openprc, d.prc, d.askhi, d.bidlo, d.vol,
               d.ret, d.cfacpr, n.ticker, n.shrcd, n.exchcd
        from crsp.dsf as d
        join crsp.dsenames as n
          on n.permno = d.permno
         and d.date between n.namedt and coalesce(n.nameendt, date '2099-12-31')
        where d.permno in ({permnos})
          and d.date between '{start}' and '{end}'
          and d.prc is not null
          and d.openprc is not null
    """
    print("  querying crsp.dsf ...", flush=True)
    raw = raw_sql(db, q)
    dl = raw_sql(db, f"""select permno, dlstdt, dlret from crsp.dsedelist
                         where permno in ({permnos}) and dlret is not null""")
    return to_panel_crsp(raw, dl)


def to_panel_crsp(raw: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    for c in ("openprc", "prc", "askhi", "bidlo", "vol", "ret", "cfacpr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # A NEGATIVE prc is CRSP's flag for a bid/ask midpoint, i.e. no trade that
    # day. Take the magnitude (standard practice) and record how many.
    df["no_trade"] = df["prc"] < 0
    for c in ("prc", "openprc", "askhi", "bidlo"):
        df[c] = df[c].abs()
    df = df[(df["prc"] > 0) & (df["openprc"] > 0) & (df["ret"].notna())]

    # apply the delisting return on the delisting date
    if len(delist):
        d = delist.copy()
        d["Date"] = pd.to_datetime(d["dlstdt"]).astype("datetime64[ns]")
        d["dlret"] = pd.to_numeric(d["dlret"], errors="coerce")
        df = df.merge(d[["permno", "Date", "dlret"]], on=["permno", "Date"], how="left")
        hit = df["dlret"].notna()
        df.loc[hit, "ret"] = (1 + df.loc[hit, "ret"]) * (1 + df.loc[hit, "dlret"]) - 1
        print(f"  delisting returns applied to {int(hit.sum())} observations")

    df["Ticker"] = df["permno"].astype(int).astype(str)     # identity is permno
    df["ticker_label"] = df["ticker"]
    df = (df.sort_values(["Ticker", "Date"])
            .drop_duplicates(["Ticker", "Date"], keep="last").reset_index(drop=True))

    # LPS's own decomposition
    df["r_cc"] = df["ret"]
    df["r_intraday"] = df["prc"] / df["openprc"] - 1
    df["r_overnight"] = (1 + df["r_cc"]) / (1 + df["r_intraday"]) - 1

    # a return needs the previous trading day to exist for that permno
    cal = pd.Index(sorted(df["Date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    dpos = df["Date"].map(pos)
    contiguous = (dpos - dpos.groupby(df["Ticker"]).shift(1)) == 1
    df.loc[~contiguous, ["r_cc", "r_overnight"]] = np.nan

    df["Close"] = df["prc"] / df["cfacpr"]
    df["Open"] = df["openprc"] / df["cfacpr"]
    df["High"] = df["askhi"] / df["cfacpr"]
    df["Low"] = df["bidlo"] / df["cfacpr"]
    df["Volume"] = df["vol"]

    chk = ((1 + df["r_overnight"]) * (1 + df["r_intraday"]) - 1 - df["r_cc"]).abs()
    df.attrs["max_decomposition_error"] = float(chk.max()) if chk.notna().any() else 0.0
    df.attrs["no_trade_share"] = float(df["no_trade"].mean())
    keep = ["Date", "Ticker", "ticker_label", "Open", "High", "Low", "Close",
            "Volume", "r_cc", "r_intraday", "r_overnight", "permno"]
    return df[keep]


def pull_constituents_crsp(db, start: str = "2014-12-01", freq: str = "D") -> pd.DataFrame:
    """crsp.dsp500list gives exact membership spans, so point-in-time
    membership is daily and real rather than a quarterly reconstruction.

    dsp500list covers the index's whole history, so the spans are clipped to
    `start` before expanding; without that it emits ~36k daily snapshots back
    to the 1920s, none of which the analysis window can use.
    """
    q = """select permno, start as "from", ending as thru from crsp.dsp500list"""
    m = raw_sql(db, q)
    m["Ticker"] = m["permno"].astype(int).astype(str)
    m["from"] = pd.to_datetime(m["from"]).astype("datetime64[ns]")
    m["thru"] = pd.to_datetime(m["thru"]).astype("datetime64[ns]")
    lo = pd.Timestamp(start)
    m = m[m["thru"].isna() | (m["thru"] >= lo)]
    m["from"] = m["from"].clip(lower=lo)
    return membership_to_panel(m, freq=freq)


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
            if has_crsp(db):
                print("CRSP entitled -> using crsp.dsf (PREREGISTRATION.md)")
                p = pull_prices_crsp(db, a.start)
                print(f"  bid/ask-midpoint closes: {p.attrs['no_trade_share']:.2%}")
            else:
                print("CRSP not entitled -> falling back to comp.secd")
                p = pull_prices(db, a.start)
            p.to_parquet(PRICES_OUT, index=False)
            print(f"prices: rows={len(p):,} tickers={p.Ticker.nunique():,} "
                  f"decomposition error={p.attrs['max_decomposition_error']:.1e}")
            print(f"wrote {PRICES_OUT}")
        if a.cmd in ("constituents", "all"):
            m = (pull_constituents_crsp(db) if has_crsp(db)
                 else pull_constituents(db, sp500_gvkeyx(db)))
            m.to_parquet(MEMB_OUT, index=False)
            n = m.groupby("asof")["Ticker"].size()
            print(f"membership: snapshots={n.size} members {n.min()}-{n.max()} "
                  f"distinct ever={m.Ticker.nunique()}")
            print(f"wrote {MEMB_OUT}")
    finally:
        db.close()
