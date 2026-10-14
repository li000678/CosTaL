# CosTaL 1.0 — retired

**This repository is no longer maintained.** It holds CosTaL 1.0 (on PyPI as 0.0.2.4).
CosTaL continues at **https://github.com/ike12th/CosTaL** — see there to install the current
version.

CosTaL 2.0 is the same method (Li et al., *Briefings in Bioinformatics* 2023,
doi:10.1093/bib/bbad157) — exact cosine kNN, Tanimoto edge weights, Leiden — rewritten in Rust:

- the same clusters as the published method, tens to hundreds of times faster;
- every platform (macOS, Linux, Windows), no external binary, no temporary files;
- 1.0 code runs unchanged (`import CosTaL`, `CosTaL.find_communities(X, pp_method="mass")`,
  `CosTaL.clustering`, `CosTaL.return_scores`, …); new code can use `import costal`;
- a command line: `costal cluster cells.fcs --platform mass`;
- opt-in extras for rare populations (`find_communities_multiscale`).

Please cite the paper as before.
