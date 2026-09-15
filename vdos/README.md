# h-AlN/MoS2 VDOS workflow

This directory preserves the original vibrational-density-of-states workflow as a first-class project component.

Contents:

- `input/vdos.lmp`: LAMMPS input used to generate velocity data
- `structure/MoS2_AlN.lmp`: heterostructure structure
- `input/h-mos2.sw`, `input/XN.tersoff`: force-field files
- `analysis/vdos.py`: mass-weighted VACF/VDOS post-processing
- `figures/vdos.png`: archived spectrum
- `commands.txt`: original command record

## Audit status

The archived program multiplies a one-sided VACF by a full `np.hanning(n)` window. Because a full Hann window is zero at the first sample, this suppresses the lag-zero VACF value. The archived spectrum is therefore retained for provenance but should be regenerated with a validated one-sided taper or alternative PSD/VACF method before publication-level mechanistic interpretation.

The VDOS force-field/geometry choice should also be standardized to the final physical model used for the comparison of interest.
