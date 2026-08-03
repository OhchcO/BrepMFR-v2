# Parameter-Surface Encoder Implementation Plan

1. Add failing pure-geometry tests for normalized surface metrics and
   trim-aware intrinsic neighbour construction.
2. Add failing model tests for the hierarchical encoder, padding invariance,
   neighbour use, gradients, and zero-initialized gated fusion.
3. Add failing BIN/dataset/config tests for new fields and switches.
4. Extend STEP geometry extraction and the versioned BIN contract.
5. Extend dataset validation, optional-field normalization, and collation.
6. Implement the parametric surface encoder and residual gated fusion.
7. Thread configuration through CLI, dataset, BrepSeg, and BrepEncoder.
8. Run focused unit tests, then the complete unit-test suite; do not run
   training.
