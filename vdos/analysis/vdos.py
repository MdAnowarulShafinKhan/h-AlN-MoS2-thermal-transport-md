#!/usr/bin/env python3
"""
Additive mass-weighted VDOS from a LAMMPS velocity dump.

This script is designed for the Atomsk h-AlN/MoS2 bilayer type mapping:
    S  = types 1, 3, 4, 6, 7, 9, 10, 12
    Mo = types 2, 5, 8, 11
    Al = type 13
    N  = type 14

It outputs publication-ready data and figures:
    <prefix>_species_Mo_S_Al_N_vdos.png
    <prefix>_species_Mo_S_Al_N_vdos.pdf
    <prefix>_species_Mo_S_Al_N_vdos.dat
    <prefix>_layers_Total_MoS2_AlN_vdos.png
    <prefix>_layers_Total_MoS2_AlN_vdos.pdf
    <prefix>_layers_Total_MoS2_AlN_vdos.dat
    <prefix>_metadata.txt

Important normalization rule:
    Total_VDOS = VDOS_Mo + VDOS_S + VDOS_Al + VDOS_N
               = VDOS_MoS2 + VDOS_AlN

The script DOES NOT normalize each partial curve independently. Instead, it uses one common
normalization: the area under the total VDOS. This gives additive, common-normalized,
MD-derived mass-weighted VDOS/PDOS curves from VACF/FFT analysis.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Your uploaded Atomsk mapping
# -----------------------------
S_TYPES = {1, 3, 4, 6, 7, 9, 10, 12}
MO_TYPES = {2, 5, 8, 11}
AL_TYPES = {13}
N_TYPES = {14}
MOS2_TYPES = S_TYPES | MO_TYPES
ALN_TYPES = AL_TYPES | N_TYPES

TYPE_TO_ELEMENT = {
    1: "S", 2: "Mo", 3: "S", 4: "S", 5: "Mo", 6: "S",
    7: "S", 8: "Mo", 9: "S", 10: "S", 11: "Mo", 12: "S",
    13: "Al", 14: "N",
}

MASS_BY_ELEMENT = {
    "Al": 26.98153860,
    "N": 14.00700000,
    "Mo": 95.96000000,
    "S": 32.06000000,
}

GROUPS_BY_TYPE = {
    "Mo": MO_TYPES,
    "S": S_TYPES,
    "Al": AL_TYPES,
    "N": N_TYPES,
}

# -----------------------------
# Reading LAMMPS dump
# -----------------------------
def read_lammps_velocity_dump(filename: str, max_frames: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read LAMMPS dump custom file containing: id type vx vy vz

    Returns:
        timesteps: shape (nframes,)
        atom_types: shape (natoms,), ordered by atom ID
        velocities: shape (nframes, natoms, 3), ordered by atom ID
    """
    timesteps: List[int] = []
    frames: List[np.ndarray] = []
    atom_types_ref = None
    natoms_ref = None

    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep_line = f.readline()
            if not timestep_line:
                break
            timestep = int(timestep_line.strip())

            line = f.readline().strip()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Expected ITEM: NUMBER OF ATOMS after timestep.")
            natoms = int(f.readline().strip())
            if natoms_ref is None:
                natoms_ref = natoms
            elif natoms != natoms_ref:
                raise RuntimeError("Number of atoms changed between frames. Check dump file.")

            line = f.readline().strip()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Expected ITEM: BOX BOUNDS.")
            # Skip three box lines
            f.readline(); f.readline(); f.readline()

            header = f.readline().strip()
            if not header.startswith("ITEM: ATOMS"):
                raise RuntimeError("Expected ITEM: ATOMS.")

            columns = header.split()[2:]
            required = ["id", "type", "vx", "vy", "vz"]
            missing = [c for c in required if c not in columns]
            if missing:
                raise RuntimeError(f"Dump is missing required columns: {missing}. Need id type vx vy vz.")

            col = {name: columns.index(name) for name in columns}

            ids = np.empty(natoms, dtype=np.int64)
            types = np.empty(natoms, dtype=np.int32)
            vel = np.empty((natoms, 3), dtype=np.float64)

            for i in range(natoms):
                parts = f.readline().split()
                ids[i] = int(parts[col["id"]])
                types[i] = int(parts[col["type"]])
                vel[i, 0] = float(parts[col["vx"]])
                vel[i, 1] = float(parts[col["vy"]])
                vel[i, 2] = float(parts[col["vz"]])

            order = np.argsort(ids)
            ids = ids[order]
            types = types[order]
            vel = vel[order]

            if atom_types_ref is None:
                atom_types_ref = types.copy()
            else:
                if not np.array_equal(atom_types_ref, types):
                    raise RuntimeError("Atom ordering/types changed between frames. Check dump sorting and file.")

            timesteps.append(timestep)
            frames.append(vel)

            if max_frames is not None and len(frames) >= max_frames:
                break

    if not frames:
        raise RuntimeError("No frames found in dump file.")

    velocities = np.stack(frames, axis=0)
    return np.array(timesteps, dtype=np.int64), atom_types_ref, velocities

# -----------------------------
# VACF and VDOS calculation
# -----------------------------
def remove_center_of_mass_velocity(velocities: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
    """Remove mass-weighted COM velocity from each frame."""
    masses = masses_from_types(atom_types)
    total_mass = np.sum(masses)
    vcom = np.sum(velocities * masses[None, :, None], axis=1) / total_mass
    return velocities - vcom[:, None, :]


def masses_from_types(atom_types: np.ndarray) -> np.ndarray:
    masses = np.empty(len(atom_types), dtype=np.float64)
    for i, t in enumerate(atom_types):
        element = TYPE_TO_ELEMENT.get(int(t))
        if element is None:
            raise RuntimeError(f"Unknown atom type {t}. Update TYPE_TO_ELEMENT mapping.")
        masses[i] = MASS_BY_ELEMENT[element]
    return masses


def fft_autocorr_1d(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation using FFT for one 1D signal."""
    n = len(x)
    x = x - np.mean(x)
    nfft = 1 << ((2 * n - 1).bit_length())
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    norm = np.arange(n, 0, -1, dtype=np.float64)
    return acf / norm


def mass_weighted_vacf_for_indices(
    velocities: np.ndarray,
    masses: np.ndarray,
    indices: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """
    Raw mass-weighted VACF contribution for a group:
        C_G(t) = sum_{i in G} m_i sum_{alpha=x,y,z} <v_i,alpha(t0+t) v_i,alpha(t0)>

    No per-group normalization is used. This is necessary so that:
        C_total = C_Mo + C_S + C_Al + C_N
    and therefore VDOS_total = sum of partial VDOS on the same scale.
    """
    nframes = velocities.shape[0]
    max_lag = min(max_lag, nframes)
    vacf = np.zeros(max_lag, dtype=np.float64)

    for idx in indices:
        mi = masses[idx]
        for comp in range(3):
            acf = fft_autocorr_1d(velocities[:, idx, comp])[:max_lag]
            vacf += mi * acf
    return vacf


def gaussian_smooth(y: np.ndarray, sigma_points: float) -> np.ndarray:
    """Gaussian smoothing in frequency-index space. Pure NumPy implementation."""
    if sigma_points <= 0:
        return y.copy()
    radius = max(1, int(math.ceil(4.0 * sigma_points)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_points) ** 2)
    kernel /= np.sum(kernel)
    ypad = np.pad(y, radius, mode="edge")
    return np.convolve(ypad, kernel, mode="same")[radius:-radius]


def vacf_to_vdos(
    vacf: np.ndarray,
    sample_dt_ps: float,
    freq_smooth_thz: float,
    max_freq_thz: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert VACF to non-negative, smoothed VDOS."""
    n = len(vacf)

    # Hanning window reduces spectral ringing from finite VACF length.
    window = np.hanning(n)
    signal = vacf * window

    # The frequency unit is 1/ps = THz.
    freq = np.fft.rfftfreq(n, d=sample_dt_ps)
    spec = np.real(np.fft.rfft(signal))

    # Numerical VACF spectra can show tiny negative ringing. VDOS is non-negative.
    spec = np.maximum(spec, 0.0)

    if len(freq) > 1 and freq_smooth_thz > 0:
        df = freq[1] - freq[0]
        sigma_points = freq_smooth_thz / df
        spec = gaussian_smooth(spec, sigma_points)
        spec = np.maximum(spec, 0.0)

    if max_freq_thz is not None:
        mask = freq <= max_freq_thz
        freq = freq[mask]
        spec = spec[mask]

    return freq, spec


def select_indices(atom_types: np.ndarray, type_set: set) -> np.ndarray:
    mask = np.array([int(t) in type_set for t in atom_types], dtype=bool)
    return np.where(mask)[0]


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """
    Integrate y(x) using NumPy's composite trapezoidal rule.

    np.trapezoid is the current NumPy API. The fallback to np.trapz keeps the
    script usable on older NumPy installations that do not yet provide
    np.trapezoid.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def save_png_and_pdf(filename: str, dpi: int = 600) -> None:
    """
    Save the current Matplotlib figure as both high-resolution PNG and vector PDF.
    This is better for journal submission than PNG only.
    """
    out = Path(filename)
    plt.savefig(out, dpi=dpi)
    plt.savefig(out.with_suffix(".pdf"))


def write_dat_species(filename: str, freq: np.ndarray, species: Dict[str, np.ndarray], total_area: float) -> None:
    mo = species["Mo"] / total_area
    s = species["S"] / total_area
    al = species["Al"] / total_area
    n = species["N"] / total_area
    total = mo + s + al + n

    data = np.column_stack([freq, total, mo, s, al, n, total - (mo + s + al + n)])
    header = (
        "frequency_THz Total Mo S Al N check_Total_minus_sum_species\n"
        "# All columns after frequency are common-normalized by area(Total).\n"
        "# By construction: Total = Mo + S + Al + N."
    )
    np.savetxt(filename, data, header=header, comments="# ", fmt="%.10e")


def write_dat_layers(filename: str, freq: np.ndarray, species: Dict[str, np.ndarray], total_area: float) -> None:
    mo = species["Mo"] / total_area
    s = species["S"] / total_area
    al = species["Al"] / total_area
    n = species["N"] / total_area
    mos2 = mo + s
    aln = al + n
    total = mos2 + aln

    data = np.column_stack([freq, total, mos2, aln, total - (mos2 + aln)])
    header = (
        "frequency_THz Total MoS2 AlN check_Total_minus_sum_layers\n"
        "# All columns after frequency are common-normalized by area(Total).\n"
        "# By construction: Total = MoS2 + AlN."
    )
    np.savetxt(filename, data, header=header, comments="# ", fmt="%.10e")


def plot_species(filename: str, freq: np.ndarray, species: Dict[str, np.ndarray], total_area: float, xmax: float) -> None:
    mo = species["Mo"] / total_area
    s = species["S"] / total_area
    al = species["Al"] / total_area
    n = species["N"] / total_area
    total = mo + s + al + n

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(freq, total, label="Total", linewidth=2.2)
    plt.plot(freq, mo, label="Mo", linewidth=1.6)
    plt.plot(freq, s, label="S", linewidth=1.6)
    plt.plot(freq, al, label="Al", linewidth=1.6)
    plt.plot(freq, n, label="N", linewidth=1.6)
    plt.xlim(0, xmax)
    plt.ylim(bottom=0)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Mass-weighted VDOS, common normalized (a.u.)")
    plt.legend(frameon=False)
    plt.tight_layout()
    save_png_and_pdf(filename, dpi=600)
    plt.close()


def plot_layers(filename: str, freq: np.ndarray, species: Dict[str, np.ndarray], total_area: float, xmax: float) -> None:
    mo = species["Mo"] / total_area
    s = species["S"] / total_area
    al = species["Al"] / total_area
    n = species["N"] / total_area
    mos2 = mo + s
    aln = al + n
    total = mos2 + aln

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(freq, total, label="Total bilayer", linewidth=2.2)
    plt.plot(freq, mos2, label="MoS2 in bilayer", linewidth=1.8)
    plt.plot(freq, aln, label="AlN in bilayer", linewidth=1.8)
    plt.xlim(0, xmax)
    plt.ylim(bottom=0)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Mass-weighted VDOS, common normalized (a.u.)")
    plt.legend(frameon=False)
    plt.tight_layout()
    save_png_and_pdf(filename, dpi=600)
    plt.close()


def write_metadata(
    filename: str,
    args: argparse.Namespace,
    nframes: int,
    natoms: int,
    sample_dt_ps: float,
    max_lag_frames: int,
    atom_types: np.ndarray,
    total_area: float,
    err_species: float,
    err_layers: float,
) -> None:
    """
    Save analysis settings for reproducibility.
    """
    freq_resolution_thz = 1.0 / (max_lag_frames * sample_dt_ps)
    nyquist_thz = 1.0 / (2.0 * sample_dt_ps)

    with open(filename, "w") as f:
        f.write("MD-derived mass-weighted VDOS/PDOS metadata\n")
        f.write("================================================\n")
        f.write(f"Input dump: {args.input}\n")
        f.write(f"Output prefix: {args.output_prefix}\n")
        f.write(f"Frames read: {nframes}\n")
        f.write(f"Atoms per frame: {natoms}\n")
        f.write(f"MD timestep used in analysis (ps): {args.md_timestep_ps:.10f}\n")
        f.write(f"Velocity sampling interval (ps): {sample_dt_ps:.10f}\n")
        f.write(f"Velocity sampling interval (fs): {sample_dt_ps * 1000.0:.6f}\n")
        f.write(f"VACF max lag frames: {max_lag_frames}\n")
        f.write(f"VACF max lag time (ps): {max_lag_frames * sample_dt_ps:.6f}\n")
        f.write(f"Approximate frequency resolution (THz): {freq_resolution_thz:.8f}\n")
        f.write(f"Nyquist frequency (THz): {nyquist_thz:.8f}\n")
        f.write(f"Gaussian smoothing sigma (THz): {args.smooth_sigma_thz:.6f}\n")
        f.write(f"Maximum output frequency (THz): {args.max_freq_thz:.6f}\n")
        f.write("Mass weighting: yes, m_i * VACF_i contribution\n")
        f.write("COM removal: mass-weighted COM velocity removed from all atoms in each frame\n")
        f.write("Normalization: common normalization by integrated total VDOS area\n")
        f.write(f"Total VDOS raw area before normalization: {total_area:.10e}\n")
        f.write(f"Mo atoms: {len(select_indices(atom_types, MO_TYPES))}\n")
        f.write(f"S atoms: {len(select_indices(atom_types, S_TYPES))}\n")
        f.write(f"Al atoms: {len(select_indices(atom_types, AL_TYPES))}\n")
        f.write(f"N atoms: {len(select_indices(atom_types, N_TYPES))}\n")
        f.write(f"Additivity error, species: {err_species:.10e}\n")
        f.write(f"Additivity error, layers: {err_layers:.10e}\n")
        f.write("\nRecommended citation wording:\n")
        f.write(
            "The VDOS/PDOS was obtained from equilibrium NVE velocity trajectories "
            "using a mass-weighted velocity autocorrelation function followed by FFT. "
            "Projected curves were commonly normalized by the integrated total VDOS area.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute additive mass-weighted VDOS from a LAMMPS velocity dump."
    )
    parser.add_argument("-i", "--input", required=True, help="LAMMPS dump file with id type vx vy vz")
    parser.add_argument("-o", "--output-prefix", required=True, help="Output prefix")
    parser.add_argument("--md-timestep-ps", "--timestep-ps", dest="md_timestep_ps", type=float, default=0.0005,
                        help="MD timestep in ps. For units metal and timestep 0.0005, use 0.0005 ps.")
    parser.add_argument("--sample-dt-ps", type=float, default=None,
                        help="Sampling interval in ps. If not given, inferred from dump timesteps times md timestep.")
    parser.add_argument("--max-lag-ps", type=float, default=50.0,
                        help="Maximum VACF correlation time in ps. Default 50 ps.")
    parser.add_argument("--smooth-sigma-thz", type=float, default=0.25,
                        help="Gaussian smoothing sigma in THz. Default 0.25 THz, similar to smearing.")
    parser.add_argument("--max-freq-thz", type=float, default=50.0,
                        help="Maximum plotted/output frequency in THz. Default 50 THz.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit on number of frames read, for testing.")
    args = parser.parse_args()

    print(f"Reading dump: {args.input}")
    timesteps, atom_types, velocities = read_lammps_velocity_dump(args.input, max_frames=args.max_frames)
    nframes, natoms, _ = velocities.shape
    print(f"Frames: {nframes}")
    print(f"Atoms:  {natoms}")

    if args.md_timestep_ps <= 0:
        raise RuntimeError("--md-timestep-ps must be positive.")

    if args.sample_dt_ps is not None:
        if args.sample_dt_ps <= 0:
            raise RuntimeError("--sample-dt-ps must be positive.")
        sample_dt_ps = args.sample_dt_ps
        print("Sampling interval was provided manually with --sample-dt-ps.")
    else:
        if len(timesteps) < 2:
            raise RuntimeError("Need at least two frames to infer sampling interval. Use --sample-dt-ps.")

        dt_steps_all = np.diff(timesteps)
        if not np.all(dt_steps_all == dt_steps_all[0]):
            raise RuntimeError(
                "Dump timestep interval is not constant. "
                "VDOS requires uniformly sampled velocity data."
            )

        step_interval = int(dt_steps_all[0])
        if step_interval <= 0:
            raise RuntimeError("Could not infer positive dump step interval.")

        sample_dt_ps = step_interval * args.md_timestep_ps
        print(f"Dump interval: {step_interval} MD steps")

    print(f"Sampling interval: {sample_dt_ps:.8f} ps")

    print("Removing mass-weighted center-of-mass velocity from each frame...")
    velocities = remove_center_of_mass_velocity(velocities, atom_types)

    masses = masses_from_types(atom_types)
    max_lag_frames = min(nframes, int(round(args.max_lag_ps / sample_dt_ps)))
    if max_lag_frames < 32:
        raise RuntimeError("max_lag is too short. Increase trajectory length or --max-lag-ps.")
    print(f"VACF max lag: {max_lag_frames} frames = {max_lag_frames * sample_dt_ps:.3f} ps")

    # Report atom counts
    for name, type_set in GROUPS_BY_TYPE.items():
        print(f"{name:>2s} atoms: {len(select_indices(atom_types, type_set))}")

    species_vacf: Dict[str, np.ndarray] = {}
    species_spec: Dict[str, np.ndarray] = {}
    freq_ref = None

    for name in ["Mo", "S", "Al", "N"]:
        print(f"Computing mass-weighted VACF for {name}...")
        idx = select_indices(atom_types, GROUPS_BY_TYPE[name])
        vacf = mass_weighted_vacf_for_indices(velocities, masses, idx, max_lag=max_lag_frames)
        species_vacf[name] = vacf

        freq, spec = vacf_to_vdos(
            vacf,
            sample_dt_ps=sample_dt_ps,
            freq_smooth_thz=args.smooth_sigma_thz,
            max_freq_thz=args.max_freq_thz,
        )
        if freq_ref is None:
            freq_ref = freq
        else:
            if len(freq) != len(freq_ref) or not np.allclose(freq, freq_ref):
                raise RuntimeError("Frequency grids differ unexpectedly.")
        species_spec[name] = spec

    # Additive construction required by the user:
    # Total = Mo + S + Al + N; MoS2 = Mo + S; AlN = Al + N
    total_spec_raw = species_spec["Mo"] + species_spec["S"] + species_spec["Al"] + species_spec["N"]
    total_area = integrate_trapezoid(total_spec_raw, freq_ref)
    if total_area <= 0:
        raise RuntimeError("Total VDOS area is not positive. Check trajectory and VACF.")

    species_dat = f"{args.output_prefix}_species_Mo_S_Al_N_vdos.dat"
    species_png = f"{args.output_prefix}_species_Mo_S_Al_N_vdos.png"
    layers_dat = f"{args.output_prefix}_layers_Total_MoS2_AlN_vdos.dat"
    layers_png = f"{args.output_prefix}_layers_Total_MoS2_AlN_vdos.png"
    metadata_file = f"{args.output_prefix}_metadata.txt"

    write_dat_species(species_dat, freq_ref, species_spec, total_area)
    write_dat_layers(layers_dat, freq_ref, species_spec, total_area)
    plot_species(species_png, freq_ref, species_spec, total_area, xmax=args.max_freq_thz)
    plot_layers(layers_png, freq_ref, species_spec, total_area, xmax=args.max_freq_thz)

    # Check additivity from saved normalized arrays
    mo = species_spec["Mo"] / total_area
    s = species_spec["S"] / total_area
    al = species_spec["Al"] / total_area
    n = species_spec["N"] / total_area
    total = mo + s + al + n
    mos2 = mo + s
    aln = al + n
    err_species = np.max(np.abs(total - (mo + s + al + n)))
    err_layers = np.max(np.abs(total - (mos2 + aln)))

    write_metadata(
        metadata_file,
        args,
        nframes,
        natoms,
        sample_dt_ps,
        max_lag_frames,
        atom_types,
        total_area,
        err_species,
        err_layers,
    )

    print("\nDone. Files written:")
    print(f"  {species_png}")
    print(f"  {Path(species_png).with_suffix('.pdf')}")
    print(f"  {species_dat}")
    print(f"  {layers_png}")
    print(f"  {Path(layers_png).with_suffix('.pdf')}")
    print(f"  {layers_dat}")
    print(f"  {metadata_file}")
    print("\nAdditivity check:")
    print(f"  max|Total - (Mo+S+Al+N)|  = {err_species:.3e}")
    print(f"  max|Total - (MoS2+AlN)|    = {err_layers:.3e}")
    print("\nUse the common-normalized columns in the .dat files for publication plots.")


if __name__ == "__main__":
    main()
