# Structure library

This folder preserves reusable base structures from the project:

- `AlN.lmp`: base h-AlN structure
- `MoS2.lmp`: base MoS2 structure
- `MoS2_AlN.lmp`: archived bilayer base structure used in the in-plane/VDOS workflow
- `AlN_bottom_MoS2_top_bilayer_4.05A.lmp`: explicitly named older 4.05 Å bilayer geometry

The final ITC baseline uses a different 3.70 Å surface separation and is preserved separately as `../itc_tbc/run/AlN_MoS2.lmp`. Its dimensions and layer positions are documented in `../itc_tbc/results/structure_audit.txt`.
