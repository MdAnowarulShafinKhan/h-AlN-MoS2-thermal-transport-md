import random
import math
from pathlib import Path

# =========================
# Correct type mapping
# =========================
N_TYPES = {14}

# =========================
# User settings
# =========================
input_file  = "MoS2_AlN.lmp"
output_file = "bilayer_VN_AlN_1pct_seed01_unrelaxed.data"

vacancy_fraction = 0.01
random_seed = 1
min_vacancy_distance = 8.0 # Angstrom

# For real 104 nm NEMD system, use 50 Å to avoid hot/cold/fixed regions.
# For your small sample file, use 5 Å or 0 Å.
edge_exclusion = 5.0

random.seed(random_seed)

def read_lammps_atomic_data(filename):
    lines = Path(filename).read_text().splitlines()

    atoms_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_start = i
            break

    if atoms_start is None:
        raise RuntimeError("Atoms section not found.")

    atom_lines_start = None
    for i in range(atoms_start + 1, len(lines)):
        if lines[i].strip():
            atom_lines_start = i
            break

    atoms = []
    for line in lines[atom_lines_start:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if not parts[0].isdigit():
            continue

        atoms.append({
            "id": int(parts[0]),
            "type": int(parts[1]),
            "x": float(parts[2]),
            "y": float(parts[3]),
            "z": float(parts[4]),
        })

    return lines, atoms

def get_box_bounds(lines):
    bounds = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 4 and parts[2] == "xlo" and parts[3] == "xhi":
            bounds["xlo"], bounds["xhi"] = float(parts[0]), float(parts[1])
    return bounds

def distance(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def write_lammps_atomic_data(original_lines, atoms, remove_ids, output_file):
    new_atom_count = len(atoms) - len(remove_ids)
    output = []

    for line in original_lines:
        stripped = line.strip()

        if stripped.endswith("atoms"):
            output.append(f"{new_atom_count:12d}  atoms")
            continue

        if stripped.startswith("Atoms"):
            output.append(line)
            output.append("")
            break

        output.append(line)

    new_id = 1
    for atom in atoms:
        if atom["id"] in remove_ids:
            continue
        output.append(
            f"{new_id:10d} {atom['type']:4d} "
            f"{atom['x']:20.12f} {atom['y']:20.12f} {atom['z']:20.12f}"
        )
        new_id += 1

    Path(output_file).write_text("\n".join(output) + "\n")

lines, atoms = read_lammps_atomic_data(input_file)
bounds = get_box_bounds(lines)

xlo = bounds["xlo"]
xhi = bounds["xhi"]

def away_from_x_edges(atom):
    return (atom["x"] > xlo + edge_exclusion) and (atom["x"] < xhi - edge_exclusion)

n_atoms = [
    atom for atom in atoms
    if atom["type"] in N_TYPES and away_from_x_edges(atom)
]

all_n_atoms = [
    atom for atom in atoms
    if atom["type"] in N_TYPES
]

n_remove = round(vacancy_fraction * len(all_n_atoms))

random.shuffle(n_atoms)
selected = []

for atom in n_atoms:
    ok = True
    for old in selected:
        if distance(atom, old) < min_vacancy_distance:
            ok = False
            break

    if ok:
        selected.append(atom)

    if len(selected) == n_remove:
        break

if len(selected) < n_remove:
    raise RuntimeError(
        f"Could only select {len(selected)} N atoms, requested {n_remove}. "
        "Reduce min_vacancy_distance or edge_exclusion."
    )

remove_ids = {atom["id"] for atom in selected}

write_lammps_atomic_data(lines, atoms, remove_ids, output_file)

print("Created:", output_file)
print("Total N atoms:", len(all_n_atoms))
print("Removed N atoms:", len(remove_ids))
print("Removed atom IDs:", sorted(remove_ids))