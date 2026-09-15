# Bilayer h-AlN/MoS2 in-plane NEMD archive

This folder preserves completed in-plane thermal-conductivity simulations rather than only summary plots.

## Included completed cases

- `armchair_baseline/`: full armchair baseline record
- `zigzag_baseline/`: full zigzag baseline record
- `additional_3p7A_audited_test/`: additional completed 3.7 Å-interface test retained from the audited project tree

The baseline folders include the LAMMPS input, structures, force fields, raw temperature profile, thermostat energy data, LAMMPS log, post-processing script, preparation code, and figures.

## Audit status

These runs are **provisional archival results**. The transport direction is x while the original inputs use `boundary p p f`. This does not match the intended finite-bar interpretation with fixed ends and end reservoirs. The original files are retained unchanged for transparency. See `../audit/KNOWN_LIMITATIONS.md` before reusing the conductivity values.

The MoS2 Jiang-style SW implementation also requires verification against the exact LAMMPS executable/source used for the original runs.
