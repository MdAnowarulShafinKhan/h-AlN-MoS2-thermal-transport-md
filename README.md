# h-AlN/MoS2 Thermal Transport Molecular Dynamics

This repository is a curated, self-contained research snapshot of molecular-dynamics work on thermal transport in monolayer h-AlN, monolayer MoS2, and h-AlN/MoS2 heterostructures. It includes the final 20-seed interfacial thermal conductance (ITC/TBC) workflow, in-plane NEMD thermal-conductivity simulations, monolayer validation data, VDOS analysis, structures, force fields, vacancy utilities, tables, figures, and audit notes.

**No private cloud folder or external project-storage link is required.** The files needed to inspect and reproduce the curated workflows are stored inside this repository.

## Scientific status at a glance

### Final reproducible core: ITC/TBC
The strongest and most publication-ready branch in this archive is the September 15, 2026 20-seed ITC/TBC analysis. For all 20 positive independent trajectories, the bundled analysis reports:

- mean G = **7.3186 MW m^-2 K^-1**
- SD = **1.1745 MW m^-2 K^-1**
- SEM = **0.2626 MW m^-2 K^-1**
- bootstrap 95% CI = **6.8071-7.8189 MW m^-2 K^-1**
- mean R = **1.4027e-7 m^2 K W^-1**

The strict QC subset is also retained as a sensitivity result; the all-independent-run estimator is the primary statistical result.

### In-plane NEMD, monolayer validation, and VDOS
These project components are preserved as full scientific records, including simulation inputs and available raw/processed outputs. They are **audited/provisional**, not silently corrected. Before reusing their numerical results in a manuscript, read `audit/KNOWN_LIMITATIONS.md`, especially the in-plane periodic-x boundary-condition issue, the MoS2 Stillinger-Weber implementation verification item, and the VDOS taper issue.

## Repository map

```text
.
├── README.md
├── FILE_MANIFEST.csv
├── requirements.txt
├── audit/
│   ├── KNOWN_LIMITATIONS.md
│   ├── SOURCE_AND_BRANCH_PROVENANCE.md
│   ├── OFFLINE_ARCHIVE_CHECK.txt
│   └── VALIDATION_REPORT.txt
├── structures/
│   ├── AlN.lmp
│   ├── MoS2.lmp
│   ├── MoS2_AlN.lmp
│   └── AlN_bottom_MoS2_top_bilayer_4.05A.lmp
├── itc_tbc/
│   ├── run/
│   ├── analysis/
│   ├── raw_data/          # all 20 final trajectories
│   ├── results/
│   └── figures/
├── bilayer_nemd/
│   ├── armchair_baseline/ # completed baseline run
│   ├── zigzag_baseline/   # completed baseline run
│   └── additional_3p7A_audited_test/
├── monolayer_nemd/
│   └── 300K/
│       ├── AlN/armchair_25nm/
│       ├── AlN/zigzag_30nm/
│       ├── MoS2/armchair_25nm/
│       └── MoS2/zigzag_30nm/
├── monolayer_validation/
│   └── monolayer_validation_300K.xlsx
├── vdos/
│   ├── input/
│   ├── structure/
│   ├── analysis/
│   ├── figures/
│   └── commands.txt
└── vacancies/
```

## ITC/TBC

`itc_tbc/run/` contains the final production input, baseline structure, force fields, and 20-seed MPI/OpenMP launcher. `itc_tbc/raw_data/` contains every one of the 20 final relaxation trajectories used by the reported analysis.

Example run command:

```bash
cd itc_tbc/run
LMP=lmp_gpu bash run_20_ITC_MPI_OMP.sh
```

Example reanalysis from the repository root:

```bash
python3 itc_tbc/analysis/analyze_AlN_MoS2_ITC_FINAL.py \
  itc_tbc/raw_data/itc_relax_seed*.dat \
  --hot auto \
  --structure-data itc_tbc/run/AlN_MoS2.lmp \
  --block-ps 0.50 \
  --fit-start-ps 1.0 \
  --end-fraction 0.25 \
  --dpi 1200 \
  --formats png,pdf \
  --outdir ITC_reanalysis
```

## Bilayer in-plane thermal conductivity (NEMD)

`bilayer_nemd/armchair_baseline/` and `bilayer_nemd/zigzag_baseline/` each contain a completed simulation record:

- `new.langevin.lmp` - LAMMPS input
- `MoS2_AlN.lmp` and `MoS2_AlN_big.lmp` - structures
- `h-mos2.sw`, `XN.tersoff` - force-field files
- `profile.langevin` - raw spatial temperature profile output
- `Ener_equ.dat` - thermostat energy exchange data
- `log.lammps` - LAMMPS log
- `all.txt` - conductivity post-processing program
- `code.txt` - associated structure/preparation code preserved from the working project
- `temp_profile_midregion.png`, `energy_vs_time.png` - resulting figures

`additional_3p7A_audited_test/` preserves an additional completed 3.7 Å-interface test case.

These are intentionally original simulation records. They retain the methodological issues identified in the audit rather than being retroactively altered.

## Monolayer thermal conductivity and 300 K validation

`monolayer_nemd/300K/` contains four complete representative NEMD calculations:

- h-AlN armchair: 25 nm
- h-AlN zigzag: 30 nm
- MoS2 armchair: 25 nm
- MoS2 zigzag: 30 nm

Each folder contains the original LAMMPS input, material structure, potential, energy-exchange data, LAMMPS log, conductivity analysis script, and result figures. The large raw `profile.langevin` files are stored losslessly as `profile.langevin.gz` to keep individual GitHub files manageable. Recover the exact raw text file with:

```bash
gzip -dk profile.langevin.gz
```

The accompanying workbook `monolayer_validation/monolayer_validation_300K.xlsx` preserves the six-length inverse-length validation datasets. The archived length sets are:

- armchair: 25, 52, 78, 104, 130, 195 nm
- zigzag: 30, 60, 90, 120, 150, 195 nm

The workbook contains h-AlN and MoS2 armchair/zigzag results used for the finite-size extrapolations. See the audit before treating those extrapolated values as final because the underlying NEMD setup inherits the transport-direction boundary-condition issue.

## VDOS

`vdos/` is now a first-class project component and contains:

- `input/vdos.lmp` - LAMMPS velocity-trajectory generation input
- `structure/MoS2_AlN.lmp` - heterostructure model used for VDOS
- `input/h-mos2.sw`, `input/XN.tersoff` - force fields
- `analysis/vdos.py` - mass-weighted VACF/VDOS analysis
- `figures/vdos.png` - archived VDOS result
- `commands.txt` - original execution-command record

The archived VDOS result is preserved for provenance. The audit found that the one-sided VACF is multiplied by a full Hann window, so regenerate a final publication spectrum with a validated one-sided taper/PSD procedure before making mechanistic claims from the spectrum.

## Structures

`structures/` keeps the small/base monolayer and bilayer structures independently of the run folders, including the older 4.05 Å h-AlN/MoS2 geometry. The final ITC 3.70 Å structure is separately preserved under `itc_tbc/run/AlN_MoS2.lmp` and documented by `itc_tbc/results/structure_audit.txt`.

## Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

The final ITC analysis uses NumPy and Matplotlib. The archived NEMD post-processing additionally uses pandas.

## Critical audit notes

Read `audit/KNOWN_LIMITATIONS.md` before reusing NEMD or VDOS values. The most important items are:

1. in-plane NEMD transports heat along x while the archived inputs use periodic x (`boundary p p f`);
2. the exact corrected/custom Jiang MoS2 SW implementation used by the original LAMMPS executable must be verified;
3. the archived VDOS analysis applies a full Hann window directly to a one-sided VACF;
4. some NEMD comments report incorrect physical run durations for a 0.5 fs timestep;
5. the NEMD post-processing scripts hard-code geometry/effective thickness values.

The archive preserves these original files so the research history is transparent. Their presence does not mean the identified issues have been resolved.

## Archive scope

This is a **GitHub-oriented scientific repository**, not a byte-for-byte backup of every repetitive working-directory file. It preserves the final ITC dataset, completed bilayer NEMD baselines, representative complete monolayer NEMD runs, the full 300 K validation workbook, VDOS workflow, structures, force fields, vacancy tools, figures, tables, and audit/provenance records. Repetitive restart files, screen dumps, duplicate potentials, and large sets of prepared-but-unexecuted input folders are intentionally not duplicated.

## Integrity and offline use

`FILE_MANIFEST.csv` records SHA-256 hashes of the packaged files. The original ITC analysis manifest is retained in `itc_tbc/results/analysis_manifest.txt`. This repository is designed to remain understandable and usable after the original working storage is removed; do not rely on an external private-storage URL when sharing it.

## License

No software/data license was present in the source project. Add a `LICENSE` only after choosing the terms under which you want others to reuse the code and data.
