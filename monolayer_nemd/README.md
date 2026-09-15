# Monolayer NEMD thermal-conductivity archive

This folder supplies actual 300 K monolayer simulation records in addition to the validation workbook.

## Complete representative runs

- `300K/AlN/armchair_25nm/`
- `300K/AlN/zigzag_30nm/`
- `300K/MoS2/armchair_25nm/`
- `300K/MoS2/zigzag_30nm/`

Each run includes its LAMMPS input, structure, force field, thermostat energy output, LAMMPS log, conductivity post-processing program, preparation code, and figures. Large spatial-temperature profiles are losslessly compressed as `profile.langevin.gz`; use `gzip -dk profile.langevin.gz` to restore the original file.

## Finite-size validation

The full 300 K finite-size numerical datasets are retained in `../monolayer_validation/monolayer_validation_300K.xlsx`.

Length sets used in the working project:

- armchair: 25, 52, 78, 104, 130, 195 nm
- zigzag: 30, 60, 90, 120, 150, 195 nm

## Audit status

The stored numerical results are provisional because the archived in-plane NEMD inputs use periodic x while heat transport is along x. MoS2 runs additionally depend on verification of the exact Jiang SW implementation in the original LAMMPS executable. See `../audit/KNOWN_LIMITATIONS.md`.
