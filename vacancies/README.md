# Vacancy-generation utilities

These scripts were preserved from the audited project and use the project's 14-type h-AlN/MoS2 mapping.

- `V_Al.py`: Al vacancies in h-AlN
- `N_vac.py`: N vacancies in h-AlN
- `S_vac_random.py`: sulfur-vacancy utility
- `V_S_interface.py`: interface-side S vacancies in MoS2
- `MoS2_AlN.lmp`: base bilayer structure for the supplied mapping

**Important:** for production NEMD cells, do not leave the default 5 Å edge exclusion unchanged. Set the exclusion so vacancies cannot occur inside the fixed slabs or hot/cold thermostat regions, and retain a buffer between defects and those regions.
