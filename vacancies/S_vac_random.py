import random
import math
from pathlib import Path

# =========================
# Correct type mapping
# =========================
AL_TYPES = {13}
N_TYPES  = {14}

MO_TYPES = {2, 5, 8, 11}

S_TOP_TYPES    = {1, 4, 7, 10}
S_BOTTOM_TYPES = {3, 6, 9, 12}

S_TYPES = S_TOP_TYPES | S_BOTTOM_TYPES

# =========================
# User settings
# =========================
input_file  = "MoS2_AlN.lmp"
output_file = "bilayer_VS_MoS2_1pct_seed01_unrelaxed.data"

vacancy_fraction = 0.01
random_seed = 1
min_vacancy_distance = 8.0   # Angstrom

# For real 104 nm NEMD system, use 50 Å to avoid hot/cold/fixed regions.
# For your small sample file, use 5 Å or 0 Å.
edge_exclusion = 5.0

random.seed(random_seed)

# =========================
# Basic LAMMPS data parser
# Works for Atoms # atomic format:
# atom-ID atom-type x y z
# =========================
def read_lammps_atomic_data(filename):
    lines = Path(filename).read_text().splitlines()

    atoms_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_start = i
            break

    if atoms_start is None:
        raise RuntimeError("Atoms section not found.")

    header_lines = lines[:atoms_start + 1]
    atoms = []
    atom_lines_start = None

    for i in range(atoms_start + 1, len(lines)):
        if lines[i].strip():
            atom_lines_start = i
            break

    for line in lines[atom_lines_start:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if not parts[0].isdigit():
            continue

        atom_id = int(parts[0])
        atom_type = int(parts[1])
        x, y, z = map(float, parts[2:5])
        atoms.append({
            "id": atom_id,
            "type": atom_type,
            "x": x,
            "y": y,
            "z": z,
            "line": line,
        })

    return lines, header_lines, atoms

def get_box_bounds(lines):
    bounds = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 4 and parts[2] == "xlo" and parts[3] == "xhi":
            bounds["xlo"], bounds["xhi"] = float(parts[0]), float(parts[1])
        if len(parts) == 4 and parts[2] == "ylo" and parts[3] == "yhi":
            bounds["ylo"], bounds["yhi"] = float(parts[0]), float(parts[1])
        if len(parts) == 4 and parts[2] == "zlo" and parts[3] == "zhi":
            bounds["zlo"], bounds["zhi"] = float(parts[0]), float(parts[1])
    return bounds

def distance(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def far_from_selected(candidate, selected, min_dist):
    for atom in selected:
        if distance(candidate, atom) < min_dist:
            return False
    return True

def choose_with_min_distance(candidates, nchoose, selected_global):
    candidates = candidates[:]
    random.shuffle(candidates)

    selected_local = []
    for atom in candidates:
        if far_from_selected(atom, selected_global + selected_local, min_vacancy_distance):
            selected_local.append(atom)
        if len(selected_local) == nchoose:
            break

    if len(selected_local) < nchoose:
        raise RuntimeError(
            f"Could only select {len(selected_local)} atoms, requested {nchoose}. "
            "Reduce min_vacancy_distance or edge_exclusion."
        )

    return selected_local

def write_lammps_atomic_data(original_lines, atoms, remove_ids, output_file):
    new_atom_count = len(atoms) - len(remove_ids)

    output = []
    in_atoms_section = False
    atoms_header_written = False

    for line in original_lines:
        stripped = line.strip()

        # Update atom count line
        if stripped.endswith("atoms"):
            output.append(f"{new_atom_count:12d}  atoms")
            continue

        # Stop before old atom coordinates
        if stripped.startswith("Atoms"):
            output.append(line)
            output.append("")
            in_atoms_section = True
            atoms_header_written = True
            break

        output.append(line)

    # Write remaining atoms with new sequential IDs
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

# =========================
# Main sulfur-vacancy creation
# =========================
lines, header, atoms = read_lammps_atomic_data(input_file)
bounds = get_box_bounds(lines)

xlo = bounds["xlo"]
xhi = bounds["xhi"]

def away_from_x_edges(atom):
    return (atom["x"] > xlo + edge_exclusion) and (atom["x"] < xhi - edge_exclusion)

s_top = [
    atom for atom in atoms
    if atom["type"] in S_TOP_TYPES and away_from_x_edges(atom)
]

s_bottom = [
    atom for atom in atoms
    if atom["type"] in S_BOTTOM_TYPES and away_from_x_edges(atom)
]

all_s_atoms = [atom for atom in atoms if atom["type"] in S_TYPES]
n_remove_total = round(vacancy_fraction * len(all_s_atoms))

n_remove_top = n_remove_total // 2
n_remove_bottom = n_remove_total - n_remove_top

selected = []
selected += choose_with_min_distance(s_top, n_remove_top, selected)
selected += choose_with_min_distance(s_bottom, n_remove_bottom, selected)

remove_ids = {atom["id"] for atom in selected}

write_lammps_atomic_data(lines, atoms, remove_ids, output_file)

print("Created:", output_file)
print("Total S atoms:", len(all_s_atoms))
print("Removed S atoms:", len(remove_ids))
print("Removed upper S:", n_remove_top)
print("Removed lower S:", n_remove_bottom)
print("Removed atom IDs:", sorted(remove_ids))