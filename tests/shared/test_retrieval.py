"""Retrieval tests (blueprint sec 6): exact vs semantic match types, reworded-topic
semantic recall, and an honest no-match below the similarity threshold. Mode-agnostic — the
retriever is called directly through ``recommend_study_material``'s underlying index and never
touches an LLM client, so this holds identically across all three pipeline modes."""

from src.retrieval.indexer import build_index, get_retriever
from src.utils.data_loader import load_dataset


def _retriever(config):
    repo = load_dataset("all")
    build_index(repo.materials(), config)
    return get_retriever(config, "all", materials=repo.materials())


def test_exact_match_is_labeled_exact(config):
    hits = _retriever(config).recommend("Algebra", top_k=3)
    assert hits
    assert hits[0]["material_id"] == "M101"
    assert hits[0]["match_type"] == "exact"


def test_reworded_topic_matches_semantically(config):
    # weak topic "Cell Division and Mitosis"; material topic is the reworded "Mitosis Cell Division"
    hits = _retriever(config).recommend("Cell Division and Mitosis", top_k=3)
    ids = [h["material_id"] for h in hits]
    assert "M205" in ids
    m205 = next(h for h in hits if h["material_id"] == "M205")
    assert m205["match_type"] in ("semantic", "exact")


def test_below_threshold_returns_no_match(config):
    hits = _retriever(config).recommend("Quantum Chromodynamics Renormalization", top_k=3)
    assert hits == []


def test_unknown_topic_never_fabricates(config):
    hits = _retriever(config).recommend("Probability", top_k=5)
    # SYN-08's weak topics deliberately have no material on file
    assert hits == []


def test_score_present_and_bounded(config):
    hits = _retriever(config).recommend("Fractions", top_k=2)
    assert hits
    for h in hits:
        assert 0.0 <= h["score"] <= 1.0
