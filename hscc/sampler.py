from typing import Tuple

import argparse
import numpy as np
from rdkit import Chem

from .conf_calc.pyxtb_conf_calc import ConfCalc
from .conf_calc.angles import Conformation
from .nutpie_hmc.nuts_sampler import MassMatrixAdaptation, NutsSampler


def build_calculator(mol_file: str) -> Tuple[ConfCalc, int]:
    ref_conf = Chem.MolFromMolFile(mol_file, removeHs=False)
    all_dih_angles = Conformation.find_unique_dihedral_angles(ref_conf)
    nonring_dih_angles = all_dih_angles[0]
    rotatable_dih_idx = [list(dih_angle[0]) for dih_angle in nonring_dih_angles]
    calculator = ConfCalc(
        mol=ref_conf,
        rotable_dihedral_idxs=rotatable_dih_idx,
    )
    dim = len(rotatable_dih_idx)
    return calculator, dim


def make_log_prob_sphere(calculator: ConfCalc):
    def log_prob_sphere(z: np.ndarray) -> Tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=float)
        # transform R -> (-pi, pi)
        phi = np.pi * np.tanh(z)
        
        kB_au = 3.1668105e-6
        T = 300.0
        beta = 1.0 / (kB_au * T)
        result = calculator.get_energy(
            phi.tolist(),
            req_opt=False,
            req_grad=True,
        )
        E_phi = result["energy"]
        grad_E_phi = np.array([g[1] for g in result["grads"]])
        grad_E_phi = np.asarray(grad_E_phi, dtype=float)
        
        dim_z = z.size
        log_abs_detJ = np.log(np.pi) * dim_z - 2.0 * np.log(np.cosh(z)).sum()
        
        tanh_z = np.tanh(z)
        dphi_dz = np.pi * (1.0 - tanh_z ** 2)
        
        log_p = -beta * float(E_phi) + log_abs_detJ
        grad_logp = -beta * grad_E_phi * dphi_dz
        grad_logp += -2.0 * np.tanh(z)
        
        return log_p, grad_logp
    return log_prob_sphere


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mol",
        type=str,
        default="chlor_butan.mol",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--stats",
        type=str,
        default="sampling_stats.nc",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(42)

    calculator, dim = build_calculator(args.mol)
    log_prob_sphere = make_log_prob_sphere(calculator)

    nuts = NutsSampler(
        lambda x: log_prob_sphere(x),
        dim,
    )

    sampler_stats = nuts.create_sample(
        mass_matrix_mode=MassMatrixAdaptation.GRADIENT_BASED,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=0.6,
        maxdepth=4,
        max_energy_error=1000.0,
    )

    nuts.statistics.save_to_nc(args.stats)

    def circular_mean(angles):
        x = np.mean(np.cos(angles), axis=0)
        y = np.mean(np.sin(angles), axis=0)
        return np.arctan2(y, x)

    samples = sampler_stats.values
    phi_samples = np.pi * np.tanh(samples)
    
    for chain in range(phi_samples.shape[0]):
        conformations = []

        for draw in range(phi_samples.shape[1]):
            phi = phi_samples[chain, draw, :]
            mol_with_conf = calculator.get_conformation(phi.tolist())
            conformations.append(mol_with_conf)

        phi_chain = phi_samples[chain]
        phi_mean = circular_mean(phi_chain)
        mol_with_conf_mean = calculator.get_conformation(phi_mean.tolist())
        conformations.append(mol_with_conf_mean)

        output_sdf = args.mol.replace(".mol", f"_conformations_{chain}.sdf")
        with Chem.SDWriter(output_sdf) as writer:
            for i, mol in enumerate(conformations):
                mol.SetProp("ConformationIndex", str(i))
                writer.write(mol)
    


if __name__ == "__main__":
    main()