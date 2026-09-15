# Offline source and branch provenance

This repository is an offline, self-contained curated snapshot of the h-AlN/MoS2 molecular-dynamics project prepared on September 15, 2026. No private cloud location or original working-storage link is required to inspect the packaged files.

## Final ITC/TBC branch

The reproducible ITC/TBC core is the newer September 15, 2026 20-seed analysis branch. The archive contains the matching 20 raw `itc_relax_seedXX.dat` trajectories, analysis script, baseline structure, production input, force fields, launcher, processed tables, figures, analysis manifest, and structure audit.

A separate older 20-seed analysis branch existed in the working project and produced a different numerical summary. Its processed result branch was deliberately excluded from the GitHub core to avoid mixing analyses.

## In-plane NEMD material preserved

The archive additionally preserves completed original in-plane thermal-conductivity simulations:

- bilayer h-AlN/MoS2 armchair baseline
- bilayer h-AlN/MoS2 zigzag baseline
- an additional audited 3.7 Å bilayer test case
- monolayer h-AlN armchair 300 K example
- monolayer h-AlN zigzag 300 K example
- monolayer MoS2 armchair 300 K example
- monolayer MoS2 zigzag 300 K example

Each complete example retains its original input, structure, force field, available raw profile/energy/log data, analysis script, and figures. Large monolayer `profile.langevin` files are stored losslessly as `.gz` files.

The full six-length 300 K numerical validation datasets are retained in `monolayer_validation/monolayer_validation_300K.xlsx`.

## VDOS material preserved

The VDOS workflow is retained under `vdos/` with its LAMMPS input, heterostructure, two force-field files, VACF/VDOS analysis script, archived output figure, and original command record.

## Structure library

Reusable base h-AlN, MoS2, bilayer, and older 4.05 Å heterostructure files are retained under `structures/`. The final 3.70 Å ITC structure is separately stored under `itc_tbc/run/`.

## Local integrity records

`itc_tbc/results/analysis_manifest.txt` contains the original SHA-256 checksums for the final ITC raw trajectories plus the ITC structure and analyzer. `FILE_MANIFEST.csv` records SHA-256 checksums for this packaged repository. `audit/VALIDATION_REPORT.txt` records the mechanical package checks.

## What this repository intentionally is not

This is a GitHub-oriented scientific archive, not a byte-for-byte mirror of every working-directory artifact. It intentionally does not duplicate repetitive restarts, screen outputs, duplicate potentials in every parameter-grid folder, unrelated auxiliary projects, or large collections of prepared-only cases that did not contain completed results.

The presence of NEMD and VDOS files preserves the work; it does not erase the scientific caveats documented in `KNOWN_LIMITATIONS.md`.
