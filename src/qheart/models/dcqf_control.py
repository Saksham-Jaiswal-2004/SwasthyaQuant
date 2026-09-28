"""The classical control for the DCQF arm. Without this, the DCQF result means nothing.

DCQF turns 8 features into 24: eight one-body expectations, eight chain pairs, eight
chain triples. Comparing that against the 8 raw features and reporting a win would be
measuring the *dimensionality increase*, not the quantum step. The paper does not
control for this, which is one of the two methodological gaps this project exists to
close.

Two controls are provided, and a DCQF row in a results table should carry both.

``ChainProductControl`` -- **the dimensionality-matched control.** It applies the
classical operation with the same index structure as the quantum readout: where DCQF
measures <Z_a Z_b Z_c> on a closed-chain triple, this multiplies features a, b and c.
Same subsets, same count, same ordering, no circuit. If DCQF cannot beat this, its
advantage was the extra columns.

``ScrambledDCQF`` -- **the null model.** It is the real circuit with the couplings
randomly permuted, so the quantum evolution, the coupling magnitudes and the output
dimension are all preserved while the *correspondence between couplings and variables*
is destroyed. If DCQF cannot beat this, the mutual-information encoding -- the actual
content of Eq. (2) -- is doing nothing, and the features are a fixed random nonlinear
map that happens to work.

The two controls fail in different directions on purpose. Beating ChainProduct but not
Scrambled means "quantum-ish nonlinearity helps, the MI encoding does not". Beating
Scrambled but not ChainProduct means "the correlation structure helps, but a product
captures it more cheaply". Only beating both supports the method as published.

Why ``tanh`` on the product control: raw feature products on angle-scaled inputs span
a much wider range than a bounded expectation value in [-1, 1], and an unbounded
control against a bounded treatment is a scaling artefact waiting to be reported as a
finding. ``squash=True`` (the default) maps the products into the same interval the
quantum observables live in, so the head sees comparably-conditioned inputs from both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qheart.quantum.dcqf import DCQFExtractor, hypergraph_closed_chain

__all__ = ["ChainProductControl", "ScrambledDCQF", "matched_controls_for"]


@dataclass
class ChainProductControl:
    """Classical products over the same closed-chain subsets DCQF measures.

    Deliberately *not* a full polynomial expansion. ``PolynomialFeatures(degree=3)`` on
    8 features gives 164 columns, which would hand the control a 6.8x dimensionality
    advantage and make the comparison unfair in the opposite direction. The chain
    restriction is what DCQF itself uses to keep the output linear in n, so the control
    inherits it.
    """

    orders: tuple[int, ...] = (1, 2, 3)
    squash: bool = True

    subsets_: list[tuple[int, ...]] = field(default_factory=list, init=False, repr=False)
    n_features_in_: int | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ChainProductControl":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected 2-D input, got shape {X.shape}")
        n = X.shape[1]
        self.n_features_in_ = n
        self.subsets_ = []
        for k in sorted(set(int(o) for o in self.orders)):
            if k == 1:
                self.subsets_.extend((i,) for i in range(n))
            else:
                self.subsets_.extend(hypergraph_closed_chain(n, k))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("call fit before transform")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"transform got {X.shape[1] if X.ndim == 2 else '?'} features but fit "
                f"saw {self.n_features_in_}"
            )
        out = np.empty((X.shape[0], len(self.subsets_)), dtype=float)
        for j, S in enumerate(self.subsets_):
            out[:, j] = np.prod(X[:, list(S)], axis=1)
        return np.tanh(out) if self.squash else out

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def feature_names(self) -> list[str]:
        return ["*".join(f"x{i}" for i in S) for S in self.subsets_]

    def n_output_features(self) -> int:
        if self.n_features_in_ is None:
            raise RuntimeError("call fit first")
        return len(self.subsets_)

    def n_trainable_params(self) -> int:
        """Zero, matching ``DCQFExtractor.n_trainable_params()``.

        Both transforms are fixed given the training fold, so the arms are matched on
        parameter count as well as on output dimension. The entire trainable budget of
        each arm lives in the shared head.
        """
        return 0


def ScrambledDCQF(**kwargs) -> DCQFExtractor:      # noqa: N802  (factory, reads as a class)
    """The DCQF null: identical circuit, couplings permuted away from their variables.

    A thin factory rather than a subclass, because ``DCQFExtractor`` already implements
    the permutation behind ``scramble_couplings=True``. Having it here as a named
    control makes it discoverable from the controls module, which is where someone
    building a results table will look.
    """
    if kwargs.pop("scramble_couplings", None) is not None:
        raise TypeError("ScrambledDCQF sets scramble_couplings itself; do not pass it")
    return DCQFExtractor(scramble_couplings=True, **kwargs)


def matched_controls_for(extractor: DCQFExtractor) -> dict[str, object]:
    """Build both controls with settings copied from a configured DCQF extractor.

    Use this rather than constructing the controls by hand. Hand-construction is how a
    control silently ends up with different orders or a different chain than the arm
    it is supposed to match, and a mismatched control is worse than none -- it looks
    like rigour while proving nothing.
    """
    return {
        "chain_product": ChainProductControl(orders=tuple(extractor.orders_read)),
        "scrambled_dcqf": ScrambledDCQF(
            orders_encoded=tuple(extractor.orders_encoded),
            orders_read=tuple(extractor.orders_read),
            n_bins=extractor.n_bins,
            shots=extractor.shots,
            dt=extractor.dt,
            agp_scale=extractor.agp_scale,
            include_input=extractor.include_input,
            seed=extractor.seed,
        ),
    }
