"""Unit tests for FinBERT news-sentiment scoring, with yfinance mocked out."""

import pytest

import screener
from screener import NEWS_ARTICLES_PER_TICKER, get_news_sentiment


class RecordingModel:
    """Stub sentiment model that records the headlines it is asked to score."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, headlines):
        self.calls.append(list(headlines))
        return self.results[: len(headlines)]


def fake_ticker(monkeypatch, news):
    class FakeTicker:
        def __init__(self, symbol):
            self.news = news

    monkeypatch.setattr(screener.yf, "Ticker", FakeTicker)


def neutral(n):
    return [{"label": "neutral", "score": 0.9}] * n


def test_parses_new_yfinance_payload_format(monkeypatch):
    fake_ticker(monkeypatch, [{"content": {"title": "Acme beats estimates"}}])
    model = RecordingModel(neutral(1))
    assert get_news_sentiment("ACME", model) == 0.0
    assert model.calls == [["Acme beats estimates"]]


def test_parses_old_yfinance_payload_format(monkeypatch):
    fake_ticker(monkeypatch, [{"title": "Acme misses estimates"}])
    model = RecordingModel(neutral(1))
    assert get_news_sentiment("ACME", model) == 0.0
    assert model.calls == [["Acme misses estimates"]]


def test_malformed_items_are_skipped(monkeypatch):
    fake_ticker(monkeypatch, [{"uuid": "x"}, {"title": "Real headline"}])
    model = RecordingModel(neutral(1))
    assert get_news_sentiment("ACME", model) == 0.0
    assert model.calls == [["Real headline"]]


def test_no_news_returns_none_not_zero(monkeypatch):
    fake_ticker(monkeypatch, [])
    model = RecordingModel([])
    assert get_news_sentiment("ACME", model) is None
    assert model.calls == []


def test_only_malformed_items_returns_none(monkeypatch):
    fake_ticker(monkeypatch, [{"uuid": "x"}])
    model = RecordingModel([])
    assert get_news_sentiment("ACME", model) is None


def test_fetch_failure_returns_none_not_zero(monkeypatch):
    class ExplodingTicker:
        def __init__(self, symbol):
            raise ConnectionError("network down")

    monkeypatch.setattr(screener.yf, "Ticker", ExplodingTicker)
    assert get_news_sentiment("ACME", RecordingModel([])) is None


def test_signed_mean_of_positive_and_negative(monkeypatch):
    fake_ticker(monkeypatch, [{"title": "Good"}, {"title": "Bad"}])
    model = RecordingModel(
        [{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.4}]
    )
    assert get_news_sentiment("ACME", model) == pytest.approx(0.2)


def test_neutral_label_contributes_zero(monkeypatch):
    # Regression: neutral used to fall into the negative branch, so a
    # confidently neutral headline was scored as strongly bearish.
    fake_ticker(monkeypatch, [{"title": "Company holds annual meeting"}])
    model = RecordingModel([{"label": "neutral", "score": 0.95}])
    assert get_news_sentiment("ACME", model) == 0.0


def test_article_cap_is_enforced(monkeypatch):
    fake_ticker(monkeypatch, [{"title": f"Headline {i}"} for i in range(25)])
    model = RecordingModel(neutral(NEWS_ARTICLES_PER_TICKER))
    get_news_sentiment("ACME", model)
    assert len(model.calls[0]) == NEWS_ARTICLES_PER_TICKER
