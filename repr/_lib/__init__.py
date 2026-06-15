"""Internal library for the graph-filtered CP reproducibility suite.

Self-contained mirror of the bits of the project's main package
needed to reproduce every datapoint of the paper's main section.  No
dependency on the project's internal code base.

Modules:

* ``core`` — split-CP quantile, safe Cholesky, shrinkage covariance,
  z-score, chronological splits, metrics, deterministic RNG seeding.
* ``graph`` — symmetric normalised shift, busiest-k subgraph,
  spectral-clustering community partition, k-NN correlation graph,
  Haversine adjacency.
* ``backbone`` — ``GraphPolyVAR(K, L)`` (the shared graph-spectral
  forecaster used by every method below).
* ``filters`` — three latent filters that drive the FCP
  (filtered-CP) wrapper: ``GraphLGSSM`` (linear graph Kalman ⇒
  ``KalmanFCP``), ``NeuralDiagGaussianFilter`` (plain GRU ⇒
  ``DiagGRU``), and ``GraphNeuralSSMFilter`` (graph-convolutional GRU
  with structured covariance ⇒ ``GNF``).
* ``methods`` — the paper's eleven main-table rows by their
  publication-facing names (``GNF``, ``GNF+ACI``, ``KalmanFCP``,
  ``StaticCGIF``, ``DiagGRU``, ``GCNrankzero``, ``FactorCGIF``,
  ``AgACIGroupCGIF``, ``ACIPerGroupFactorCGIF``, ``EWMACovCGIF``,
  ``RollingCovCGIF``).
* ``vendored_copula`` — full-fidelity CopulaCPTS (Sun & Yu, ICLR
  2022) vendored from the authors' reference implementation.
* ``vendored_spci`` — full-fidelity MultiDimSPCI (Xu & Xie, ICML
  2024) vendored from the authors' reference implementation.
* ``audit`` — empirical contraction-rate diagnostics
  (``\\widehat\\rho_{\\dG}``, ``\\widehat\\rho_{\\score}``,
  ``\\widehat\\rho_{\\DL}``) used by Theorems C / observability /
  threshold-mixing in the paper.
* ``data/`` — one loader per dataset of the paper's main section
  (METR-LA, PEMS-BAY, AQI, ELEC, ETT, Solar, Jena, Loop-Seattle,
  LargeST-GBA, plus an offline synthetic GraphPolyVAR fallback).

The package is laid out as a small flat-ish library rather than a
single file so reviewers can navigate it; every file is browsable
in a few minutes and there are no external project imports.
"""
