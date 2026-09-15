#!/usr/bin/env python3
"""
Create aluminum vacancies in the h-AlN layer of an h-AlN/MoS2 bilayer.

Default target for the uploaded structure:
    Al atom type = {13}

Default concentration:
    1.0% of all Al atoms.

LAMMPS data format expected:
    Atoms # atomic
    atom-ID atom-type x y z

Run:
    python create_VAl_AlN.py

For a production NEMD cell with hot/cold/fixed edge regions, edit edge_exclusion,
or use the command-line options at the bottom of this script.
"""

import argparse
import math
import random
from collections import Counter
from pathlib import Path

# =========================
# Atom-type mapping for your uploaded MoS2_AlN.lmp
# =========================
AL_TYPES = {13}
N_TYPES = {14}
MO_TYPES = {2, 5, 8, 11}
S_TYPES = {1, 3, 4, 6, 7, 9, 10, 12}


# =========================
# LAMMPS data reader/writer
# =========================
def read_lammps_atomic_data(filename):
    lines = Path(filename).read_text().splitlines()

    atoms_header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_header_index = i
            break

    if atoms_header_index is None:
        raise RuntimeError("Atoms section not found. Expected a LAMMPS data file with 'Atoms # atomic'.")

    atom_lines_start = None
    for i in range(atoms_header_index + 1, len(lines)):
        if not lines[i].strip():
            continue
        parts = lines[i].split()
        if parts and parts[0].isdigit():
            atom_lines_start = i
            break

    if atom_lines_start is None:
        raise RuntimeError("No atom-coordinate lines found after the Atoms section.")

    atoms = []
    for line in lines[atom_lines_start:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if not parts[0].isdigit():
            break

        atoms.append({
            "id": int(parts[0]),
            "type": int(parts[1]),
            "x": float(parts[2]),
            "y": float(parts[3]),
            "z": float(parts[4]),
        })

    return lines, atoms_header_index, atoms


def get_box_bounds(lines):
    bounds = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 4 and parts[2] == "xlo" and parts[3] == "xhi":
            bounds["xlo"], bounds["xhi"] = float(parts[0]), float(parts[1])
        elif len(parts) == 4 and parts[2] == "ylo" and parts[3] == "yhi":
            bounds["ylo"], bounds["yhi"] = float(parts[0]), float(parts[1])
        elif len(parts) == 4 and parts[2] == "zlo" and parts[3] == "zhi":
            bounds["zlo"], bounds["zhi"] = float(parts[0]), float(parts[1])

    needed = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"}
    missing = needed - set(bounds)
    if missing:
        raise RuntimeError(f"Missing box bounds: {sorted(missing)}")

    return bounds


def update_atom_count_line(line, new_atom_count):
    stripped = line.strip()
    parts = stripped.split()
    if len(parts) >= 2 and parts[0].isdigit() and parts[1] == "atoms":
        return f"{new_atom_count:12d}  atoms"
    return line


def write_lammps_atomic_data(original_lines, atoms_header_index, atoms, remove_ids, output_file):
    new_atom_count = len(atoms) - len(remove_ids)
    output = []

    # Copy everything up to and including the Atoms header.
    for i in range(atoms_header_index + 1):
        output.append(update_atom_count_line(original_lines[i], new_atom_count))

    output.append("")

    # Write remaining atoms with new sequential IDs. This is safe for atom_style atomic.
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
# Vacancy-selection tools
# =========================
def minimum_image_delta(d, box_length):
    return d - round(d / box_length) * box_length


def distance(a, b, bounds, periodic_x=False, periodic_y=True, periodic_z=False):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]

    if periodic_x:
        dx = minimum_image_delta(dx, bounds["xhi"] - bounds["xlo"])
    if periodic_y:
        dy = minimum_image_delta(dy, bounds["yhi"] - bounds["ylo"])
    if periodic_z:
        dz = minimum_image_delta(dz, bounds["zhi"] - bounds["zlo"])

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def away_from_x_edges(atom, bounds, edge_exclusion):
    return (atom["x"] > bounds["xlo"] + edge_exclusion) and (atom["x"] < bounds["xhi"] - edge_exclusion)


def choose_with_min_distance(candidates, nchoose, bounds, min_dist, seed):
    rng = random.Random(seed)
    candidates = candidates[:]
    rng.shuffle(candidates)

    selected = []
    for atom in candidates:
        ok = True
        for old in selected:
            if distance(atom, old, bounds, periodic_x=False, periodic_y=True, periodic_z=False) < min_dist:
                ok = False
                break
        if ok:
            selected.append(atom)
        if len(selected) == nchoose:
            break

    if len(selected) < nchoose:
        raise RuntimeError(
            f"Could only select {len(selected)} vacancies, requested {nchoose}. "
            "Reduce min_vacancy_distance or edge_exclusion."
        )

    return selected


def write_id_file(selected, id_file):
    with open(id_file, "w") as f:
        f.write("# Original atom IDs removed from the input data file\n")
        for atom in sorted(selected, key=lambda a: a["id"]):
            f.write(f"{atom['id']}\n")


def write_summary(summary_file, input_file, output_file, id_file, atoms, selected, bounds, vacancy_fraction, seed, min_dist, edge_exclusion):
    counts_before = Counter(atom["type"] for atom in atoms)
    counts_after = Counter(atom["type"] for atom in atoms if atom["id"] not in {a["id"] for a in selected})

    target_total = sum(counts_before[t] for t in AL_TYPES)
    actual_fraction = len(selected) / target_total if target_total else 0.0
    area_A2 = (bounds["xhi"] - bounds["xlo"]) * (bounds["yhi"] - bounds["ylo"])
    vacancy_density_cm2 = len(selected) / (area_A2 * 1.0e-16)

    z_al = [a["z"] for a in atoms if a["type"] in AL_TYPES]

    lines = []
    lines.append("Aluminum vacancy summary")
    lines.append(f"Input file: {input_file}")
    lines.append(f"Output file: {output_file}")
    lines.append(f"Removed-ID file: {id_file}")
    lines.append(f"Random seed: {seed}")
    lines.append(f"Requested vacancy fraction: {vacancy_fraction:.8f}")
    lines.append(f"Minimum vacancy distance: {min_dist:.6f} Angstrom")
    lines.append(f"x-edge exclusion: {edge_exclusion:.6f} Angstrom")
    lines.append("")
    lines.append(f"Target atom types: {sorted(AL_TYPES)}")
    lines.append(f"Total Al atoms before deletion: {target_total}")
    lines.append(f"Removed Al atoms: {len(selected)}")
    lines.append(f"Actual vacancy fraction of Al atoms: {actual_fraction:.8f}")
    lines.append(f"Actual vacancy percent of Al atoms: {100.0 * actual_fraction:.6f}%")
    lines.append(f"Vacancy density using full xy cell area: {vacancy_density_cm2:.6e} cm^-2")
    lines.append("")
    lines.append(f"Mean z of Al atoms before deletion: {sum(z_al) / len(z_al):.12f} Angstrom")
    lines.append("")
    lines.append("Atom counts before deletion:")
    for t in sorted(counts_before):
        lines.append(f"  type {t:2d}: {counts_before[t]}")
    lines.append("Atom counts after deletion:")
    for t in sorted(counts_after):
        lines.append(f"  type {t:2d}: {counts_after[t]}")
    lines.append("")
    lines.append("Removed original atom IDs:")
    lines.append(" ".join(str(a["id"]) for a in sorted(selected, key=lambda a: a["id"])))

    Path(summary_file).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create Al vacancies in h-AlN of h-AlN/MoS2.")
    parser.add_argument("--input", default="MoS2_AlN.lmp", help="Input LAMMPS data file")
    parser.add_argument("--output", default="bilayer_VAl_AlN_1pct_seed01_unrelaxed.data", help="Output LAMMPS data file")
    parser.add_argument("--ids", default="ids_VAl_AlN_1pct_seed01.txt", help="Output file containing removed original atom IDs")
    parser.add_argument("--summary", default="summary_VAl_AlN_1pct_seed01.txt", help="Verification summary file")
    parser.add_argument("--fraction", type=float, default=0.01, help="Vacancy fraction of Al atoms")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--min-distance", type=float, default=8.0, help="Minimum vacancy-vacancy distance in Angstrom")
    parser.add_argument("--edge-exclusion", type=float, default=5.0, help="Exclude atoms within this x-distance from both x edges, in Angstrom")
    args = parser.parse_args()

    lines, atoms_header_index, atoms = read_lammps_atomic_data(args.input)
    bounds = get_box_bounds(lines)

    all_target_atoms = [atom for atom in atoms if atom["type"] in AL_TYPES]
    candidate_atoms = [atom for atom in all_target_atoms if away_from_x_edges(atom, bounds, args.edge_exclusion)]

    n_remove = round(args.fraction * len(all_target_atoms))
    if n_remove < 1:
        raise RuntimeError("Vacancy fraction is too small; it selects zero atoms.")
    if n_remove > len(candidate_atoms):
        raise RuntimeError("Requested vacancies exceed available candidates after x-edge exclusion.")

    selected = choose_with_min_distance(candidate_atoms, n_remove, bounds, args.min_distance, args.seed)
    remove_ids = {atom["id"] for atom in selected}

    write_lammps_atomic_data(lines, atoms_header_index, atoms, remove_ids, args.output)
    write_id_file(selected, args.ids)
    write_summary(args.summary, args.input, args.output, args.ids, atoms, selected, bounds, args.fraction, args.seed, args.min_distance, args.edge_exclusion)

    print("Created:", args.output)
    print("Removed-ID file:", args.ids)
    print("Summary file:", args.summary)
    print("Total Al atoms:", len(all_target_atoms))
    print("Candidate Al atoms after edge exclusion:", len(candidate_atoms))
    print("Removed Al atoms:", len(selected))
    print("Actual Al vacancy percent:", 100.0 * len(selected) / len(all_target_atoms))
    print("Removed original atom IDs:", sorted(remove_ids))


if __name__ == "__main__":
    main()
