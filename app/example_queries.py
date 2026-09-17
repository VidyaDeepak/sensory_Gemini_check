"""
example_queries: curated, hand-written starter questions for each module,
grounded in what that module's CSV actually contains. Used to populate the
per-dataset "tab" view in the UI, so once a person pins a module they see
example queries that dataset can genuinely answer, instead of the generic
cross-dataset preset list.
"""
from __future__ import annotations

EXAMPLE_QUERIES: dict[str, list[str]] = {
    "sr_attribute": [
        "Show top attributes for Natural Product",
        "What attributes drive Moisturization?",
        "Top attributes for Natural Product by benefit stage",
        "What attributes drive Instant Glow in the Fragrance stage?",
        "Compare the attribute drivers of Moisturization and Instant Glow",
        "Which attributes matter least for Natural Product?",
    ],
    "sd_driver": [
        "What drives Overall Opinion?",
        "What's the top test-stage driver for Natural Product?",
        "Show test-stage drivers for Moisturization by benefit stage",
        "Compare test-stage drivers for Natural Product and Even Tone",
        "Which benefits does the Fingertip stage drive most?",
        "What's the weakest test-stage driver of Overall Opinion?",
    ],
    "bh_blindhut": [
        "Which products score highest on Instant Glow?",
        "How does GARNIER vitamin C Cream score across all benefits?",
        "Compare GARNIER vitamin C Cream and JOHNSONS baby oil blind-hut scores",
        "What's the top-scoring product for Moisturization?",
        "Show KIEHLS multi corrective cream's full benefit profile",
    ],
    "pc_perceptual": [
        "Show the perceptual map for SUNCARE",
        "Where does the CeraVe facial moisturizing lotion sit on the perceptual map?",
        "Compare two CeraVe facial moisturizing lotion variants on the PCA map",
        "Nearest neighbours of the CeraVe facial moisturizing lotion",
        "Which two SUNCARE products are positioned closest together?",
        "Show product clusters for SUNCARE",
        "Identify whitespace opportunities in SUNCARE",
    ],
    "rw_relweight": [
        "What are the strongest drivers for Dry?",
        "What are the weakest RWA drivers?",
        "How consistent is the top driver across studies for Hairfall?",
        "Compare driver importance across studies for Shine",
        "Summarize key RWA learnings across all platforms",
    ],
    "spider_profile": [
        "Show the attribute profile for Anessa",
        "What are Anessa's weakest attributes?",
        "Which attributes separate Anessa from Ahc?",
        "What makes Anessa unique?",
        "Show sensory gaps for Anessa against a benchmark product",
    ],
}


def examples_for(module_name: str) -> list[str]:
    return EXAMPLE_QUERIES.get(module_name, [])
