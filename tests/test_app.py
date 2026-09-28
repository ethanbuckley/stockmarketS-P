"""Smoke test for the Streamlit dashboard against the committed artefacts.

Runs the whole script headlessly. It would have caught the st.stop() bug
that blanked the Validation tab whenever the Monte Carlo selection was empty.
"""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

# Absolute path: newer Streamlit resolves relative paths against this test
# file rather than the working directory.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def app():
    at = st_testing.AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    return at


def test_app_renders_all_tabs_without_exceptions(app):
    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.tabs) == 3
    labels = {m.label for m in app.metric}
    assert {"Total candidates", "Long signals", "Short signals"} <= labels  # Screener
    assert {"VaR (95%)", "CVaR (95%)", "P(loss)"} <= labels  # Monte Carlo from committed prices
    assert {"ROC AUC (pooled)", "Brier score", "Precision@15 (daily mean)"} <= labels  # Validation


def test_validation_tab_survives_empty_monte_carlo_selection(app):
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any(m.label == "ROC AUC (pooled)" for m in app.metric)
    assert any("Built by" in m.value for m in app.markdown)  # footer still rendered
