"""Explainability. Two mechanisms, never conflated.

The most common error in hybrid-QML write-ups is running SHAP on the classical head
and captioning the plot as an explanation of the quantum model. It is not. SHAP over
a head that consumes four expectation values explains the head; the circuit that
produced those expectations is untouched by it. Every figure this package produces
carries an explicit scope string for that reason.
"""

from qheart.explain.saliency_quantum import circuit_saliency, saliency_note
from qheart.explain.shap_classical import shap_values, shap_scope_note

__all__ = ["shap_values", "shap_scope_note", "circuit_saliency", "saliency_note"]
