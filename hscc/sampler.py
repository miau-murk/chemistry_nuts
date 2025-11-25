from typing import Callable, Dict, List, Tuple
import numpy as np
from rdkit import Chem
from .conf_calc.conf_calc import ConfCalc
from .conf_calc.angles import Conformation
from .nutpie_hmc.nuts_sampler import MassMatrixAdaptation, NutsSampler


MOL_FILE = "test.mol"
ref_conf = Chem.MolFromMolFile(MOL_FILE, removeHs=False)


all_dih_angles = Conformation.find_unique_dihedral_angles(ref_conf)
nonring_dih_angles = all_dih_angles[0]
rotatable_dih_idx = [list(dih_angle[0]) for dih_angle in nonring_dih_angles]

calculator = ConfCalc(mol=ref_conf,
                    dir_to_xyzs="xtb_calcs/",
                    rotable_dihedral_idxs=rotatable_dih_idx)


def log_prob(x: np.ndarray) -> Tuple[float, np.ndarray]:
    values = x.astype(float).tolist()

    result = calculator.get_energy(
            values,
            req_opt=False,
            req_grad=True,
        )
    
    E = result["energy"]
    grad_E = np.array([g[1] for g in result["grads"]])
    grad_E = np.asarray(grad_E, dtype=float)


    log_p = -float(E)
    grad_log_p = -grad_E
    return log_p, grad_log_p


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


if __name__ == "__main__":
    np.random.seed(42)
    
    dim = len(rotatable_dih_idx)    
    
    nuts = NutsSampler(
        lambda x: log_prob_sphere(x),
        dim
    )
    
    result = nuts.create_sample(
        mass_matrix_mode=MassMatrixAdaptation.STANDARD,
        draws=30,
        tune=30,
    )

    nuts.statistics.save_to_log()


