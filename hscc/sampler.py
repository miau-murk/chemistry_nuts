from typing import Tuple

import argparse
import numpy as np
from rdkit import Chem

from .conf_calc.conf_calc import ConfCalc
from .conf_calc.angles import Conformation
from .nutpie_hmc.nuts_sampler import MassMatrixAdaptation, NutsSampler


def build_calculator(mol_file: str) -> Tuple[ConfCalc, int]:
    ref_conf = Chem.MolFromMolFile(mol_file, removeHs=False)
    all_dih_angles = Conformation.find_unique_dihedral_angles(ref_conf)
    nonring_dih_angles = all_dih_angles[0]
    rotatable_dih_idx = [list(dih_angle[0]) for dih_angle in nonring_dih_angles]
    calculator = ConfCalc(
        mol=ref_conf,
        dir_to_xyzs="xtb_calcs/",
        rotable_dihedral_idxs=rotatable_dih_idx,
    )
    dim = len(rotatable_dih_idx)
    return calculator, dim


def make_log_prob_sphere(calculator: ConfCalc):
    def log_prob_sphere(z: np.ndarray) -> Tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=float)

        # transform R -> (-pi, pi)
        phi = np.pi * np.tanh(z)
        dphi_dz = np.pi * (1.0 / np.cosh(z) ** 2)  # sech^2(z)

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

        log_abs_detJ = np.log(np.abs(dphi_dz)).sum()

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
        default="test.mol",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=3,
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

    _ = nuts.create_sample(
        mass_matrix_mode=MassMatrixAdaptation.GRADIENT_BASED,
        draws=args.draws,
        tune=args.tune,
        chains=1,
        target_accept=0.8,
        maxdepth=8,
        max_energy_error=1000.0,
    )

    nuts.statistics.save_to_nc(args.stats)


if __name__ == "__main__":
    main()
