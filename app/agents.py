"""
agents: display identity for each dataset-scoped module — the thing a
person actually picks in the UI. Each module in app/modules/ remains the
real "specialist agent" (its own schema knowledge, entity lists, query
parsing); this file is purely the human-facing name/tagline/dataset label
shown when choosing one, kept separate so router.py stays focused on
behavior and this stays focused on presentation.
"""
from __future__ import annotations

AGENTS: dict[str, dict] = {
    "sr_attribute": {
        "name": "SST Attribute Agent",
        "tagline": "Which sensory attributes drive a benefit, by benefit stage",
        "dataset": "sst_attribute_rank.csv",
    },
    "sd_driver": {
        "name": "SST Driver Agent",
        "tagline": "Which test stage (CLT wave) drives a benefit",
        "dataset": "sst_driver_rank.csv",
    },
    "bh_blindhut": {
        "name": "Blind Hut Agent",
        "tagline": "Consumer liking scores per product and benefit",
        "dataset": "blind_hut_benefit_score.csv",
    },
    "pc_perceptual": {
        "name": "PCA Agent",
        "tagline": "Perceptual map positioning, neighbours, clusters, whitespace",
        "dataset": "pca_coordinate.csv",
    },
    "rw_relweight": {
        "name": "RWA Agent",
        "tagline": "Hair-care platform driver importance",
        "dataset": "rwa_driver_importance.csv",
    },
    "spider_profile": {
        "name": "Spider Profile Agent",
        "tagline": "Full attribute profile per product, brand, or product type",
        "dataset": "spider_attribute_score.csv",
    },
}


def list_agents() -> list[dict]:
    return [{"id": key, **meta} for key, meta in AGENTS.items()]
