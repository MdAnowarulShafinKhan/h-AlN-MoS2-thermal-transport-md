Corrected h-AlN/h-MoS2 ITC/ITR production files
================================================

Primary LAMMPS input:
  AlN_MoS2_ITC_ITR_FINAL_CORRECTED.in

Launcher-compatible alias (same contents):
  AlN_MoS2_ITC_FINAL.in

Updated analyzer:
  analyze_AlN_MoS2_ITC_ITR_FINAL_CORRECTED.py

Analyzer-compatible alias (same contents):
  analyze_AlN_MoS2_ITC_FINAL.py

Key corrections relative to the uploaded draft:
  1. Defines hot_code for both heat-flow directions.
  2. Defines independent positive Langevin seeds seed_mos and seed_aln.
  3. Retains atom-count/heat-capacity-weighted temperature preparation.
  4. Records actual T_MoS2/T_AlN targets and layer atom counts in production output.
  5. Records actual targets in prep_RUNID.dat.
  6. Uses reset_timestep 0 time 0.0 so production time starts at zero explicitly.
  7. Keeps KOKKOS-compatible cg + min_modify line quadratic minimization.
  8. Corrects stale comments referring to symmetric 325/275 K preparation and forcezero.

Python changes:
  1. Reads both the new 27-column output and prior 23-column output.
  2. Uses global LAMMPS T_all, not (T_hot+T_cold)/2, for nominal-temperature QC.
  3. Keeps arithmetic layer-temperature midpoint as a separate descriptive quantity.
  4. Plots the actual weighted preparation target temperatures when available.
  5. Remains backward-compatible with the already completed seed01 output.
  6. Primary G/R estimator, 0.5 ps non-overlapping blocks, 1 ps fit start,
     0.25*DeltaT0 automatic fit end, sensitivity analysis, and 1200-dpi/vector figures are unchanged.
