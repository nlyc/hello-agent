"""HashingEmbedder 与检索分词的离线测试。"""

import math

import pytest

from agentflow.retrieval import HashingEmbedder, tokenize


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def test_hashing_embedder_dimension_and_fingerprint():
    embedder = HashingEmbedder()
    assert embedder.dimension == 512
    assert embedder.fingerprint == "hashing:char-ngram:512"
    assert len(embedder.embed_query("北京天气")) == 512


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder()
    first = embedder.embed(["故宫开放时间"])[0]
    second = HashingEmbedder().embed(["故宫开放时间"])[0]
    assert first == second


def test_hashing_embedder_normalizes():
    vector = HashingEmbedder().embed_query("北京 故宫 天气 travel")
    assert _norm(vector) == pytest.approx(1.0, abs=1e-9)


def test_hashing_embedder_different_texts_differ():
    embedder = HashingEmbedder()
    a = embedder.embed_query("北京故宫开放时间")
    b = embedder.embed_query("上海外滩夜景攻略")
    assert a != b


def test_hashing_embedder_empty_text_is_zero_vector():
    vector = HashingEmbedder().embed_query("")
    assert len(vector) == 512
    assert _norm(vector) == 0.0


def test_hashing_embedder_custom_dimension():
    embedder = HashingEmbedder(dimension=64)
    assert embedder.dimension == 64
    assert embedder.fingerprint == "hashing:char-ngram:64"
    assert len(embedder.embed_query("test")) == 64


def test_tokenize_cjk_bigram_and_english_words():
    tokens = tokenize("故宫 open hours 2026")
    assert "故宫" in tokens
    assert "open" in tokens
    assert "hours" in tokens
    assert "2026" in tokens
    assert "故" not in tokens


def test_embed_batch_matches_single():
    embedder = HashingEmbedder()
    batch = embedder.embed(["北京", "上海"])
    assert batch[0] == embedder.embed_query("北京")
    assert batch[1] == embedder.embed_query("上海")
