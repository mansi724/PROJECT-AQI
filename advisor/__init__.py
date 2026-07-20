"""
advisor — the post-attribution intelligence pipeline for the Delhi AQI platform.

Everything AFTER Source Attribution lives here:

    Context Builder -> Hybrid Retrieval (+ Knowledge Graph) -> Cross-Encoder
    Rerank -> LLM Reasoning -> Policy Validation -> Counterfactual Simulation
    -> Action Ranking -> Dashboard/API

It REUSES the frozen forecasting stack (the Graph Transformer checkpoint, SHAP /
GNNExplainer, and the LightGBM source-attribution heads) through a thin serving
layer — it never retrains or edits those modules.
"""
__all__ = ["config"]
