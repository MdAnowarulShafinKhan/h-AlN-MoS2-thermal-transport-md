# Known limitations identified during the project audit

This file intentionally separates the final/reproducible ITC branch from NEMD and VDOS workflows that are preserved for transparency but still require correction or verification before publication-level reuse.

## 1. In-plane NEMD transport-direction boundary condition — critical

The archived monolayer and bilayer NEMD inputs impose heat transport along x but use `boundary p p f`. Periodic x permits interactions across the x boundary and does not represent the same finite bar implied by fixed end slabs and hot/cold reservoirs near the two ends.

Affected material is preserved under `bilayer_nemd/` and `monolayer_nemd/` as the original simulation record.

Recommended repair: equilibrate lattice/box dimensions in a separate periodic calculation, write the relaxed structure, then start a finite transport calculation with nonperiodic x (for example `f p f`) and no pressure/barostat control along the nonperiodic heat-flow dimension. A periodic NEMD method can also be valid, but it requires a deliberately periodic reservoir geometry and corresponding flux treatment rather than the current finite-bar interpretation.

**Consequence:** existing in-plane kappa values and inverse-length extrapolations should be treated as provisional until rerun with a consistent transport geometry.

## 2. MoS2 Stillinger-Weber implementation — critical verification item

The project uses the 12-type Jiang-style `h-mos2.sw` parameter file with ordinary `pair_style sw`. The parameter file alone cannot prove that the LAMMPS executable contained the corrected/custom Jiang source behavior required by that formulation.

Before treating MoS2-containing NEMD, VDOS, or ITC results as publication-final, identify or preserve the exact LAMMPS source/executable used for production and verify the intended cutoff/angle treatment. If a vanilla stock implementation was used where a source modification was required, the MoS2-containing simulations require re-evaluation with a validated implementation.

Do not blindly replace the archived parameterization with another SW file without first benchmarking energies, forces, phonons, and known material properties.

## 3. Archived VDOS VACF taper — major

`vdos/analysis/vdos.py` multiplies a one-sided VACF by `np.hanning(n)`. A full symmetric Hann window is zero at the first sample, so it suppresses the physically important lag-zero VACF value.

**Repository decision:** the original script, input, structure, command record, and figure are retained as provenance. Regenerate a final spectrum after adopting a validated one-sided taper (starting at unity and decaying to zero) or another validated VACF/PSD procedure.

The VDOS geometry and force-field choice should also be standardized to the physical model being compared with the final ITC calculation.

## 4. NEMD production-time comments — major provenance issue

The timestep is 0.0005 ps = 0.5 fs. Several original inputs contain comments such as “8 ns” that do not match `number_of_steps × 0.5 fs`. Examples found during the audit included actual production durations of 2.0 ns, 6.0 ns, and 6.5 ns.

Always calculate physical duration from the timestep and actual run steps rather than from the comments.

## 5. NEMD post-processing geometry/thickness — major

The archived `all.txt` scripts hard-code x/y/z dimensions, use those values to convert reduced chunk coordinates, and use a fixed effective thickness convention in the cross-sectional area. This can create a mismatch if the production box dimensions differ from the hard-coded values.

For a corrected workflow, output actual production `lx`/`ly`, region positions, timestep, and the chosen effective thickness/sheet-conductance convention from LAMMPS, then read those metadata directly in the analysis program.

For 2D materials, explicitly justify any effective thickness used to report W m^-1 K^-1. Consider also reporting thickness-independent sheet conductance.

## 6. Monolayer validation regression precision

The workbook `monolayer_validation/monolayer_validation_300K.xlsx` contains six finite-length points for each material/direction. The displayed infinite-length values were based on reciprocals of rounded chart intercepts. Re-fitting the stored data during the audit gave approximately:

- h-AlN armchair: 115.45 W m^-1 K^-1, R^2 ≈ 0.925
- h-AlN zigzag: 116.81 W m^-1 K^-1, R^2 ≈ 0.937
- MoS2 armchair: 66.14 W m^-1 K^-1, R^2 ≈ 0.994
- MoS2 zigzag: 75.82 W m^-1 K^-1, R^2 ≈ 0.998

The rounding difference itself is minor. The larger issue is that these extrapolations inherit the NEMD boundary-condition problem, and the h-AlN fits have noticeably weaker linearity than the MoS2 fits.

## 7. Bilayer force-field/interface-model consistency

Older in-plane NEMD/VDOS work and the final ITC branch do not use exactly the same interface geometry/LJ baseline. The archive intentionally preserves these original choices rather than rewriting historical inputs.

For a final unified study, choose and document one baseline geometry and one LJ mixing/parameter convention; treat alternate gap/mixing choices as explicit sensitivity cases rather than silently combining them.

## 8. Vacancy edge exclusion

Several vacancy scripts use a small default x-edge exclusion. For production NEMD defect studies, candidate vacancies must be excluded from the entire fixed-end and hot/cold thermostat regions plus a buffer, and multiple random spatial realizations should be used at each concentration to quantify configurational uncertainty.

## 9. ITC QC sensitivity

The final September 15 ITC branch reports:

- all positive runs: n = 20, G = 7.3186 ± 1.1745 MW m^-2 K^-1 (mean ± SD)
- strict QC-pass subset: n = 5, G = 7.9237 ± 0.7508 MW m^-2 K^-1

The all-independent-run estimator is the primary statistical result in the supplied workflow; the QC-only value is a sensitivity/diagnostic subset. Do not report only the QC subset without disclosing its selection criterion and the all-run result.
