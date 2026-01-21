import numpy as np
from rdkit import Chem
from .conf_calc import ConfCalc
from .angles import Conformation

mol_file = "bibenzil.mol"

ref_conf = Chem.MolFromMolFile(mol_file, removeHs=False)

all_dih_angles = Conformation.find_unique_dihedral_angles(ref_conf) # ([((1, 2, 3, 4), angle_value)], [ ring ])
nonring_dih_angles = all_dih_angles[0] # only non-ring angles

rotatable_dih_idx = []
for dih_angle in nonring_dih_angles:
    rotatable_dih_idx.append(list(dih_angle[0]))


# Usage example
calculator = ConfCalc(mol=ref_conf,
                      dir_to_xyzs="xtb_calcs/",
                      rotable_dihedral_idxs=rotatable_dih_idx)

dim = len(rotatable_dih_idx)
phi_random = np.random.uniform(0, 2 * np.pi, dim)

mol = calculator.get_conformation(phi_random)
Chem.MolToMolFile(mol, "bibenzil.mol")

# print(calculator.get_energy(np.random.random(len(nonring_dih_angles)), req_opt=False,
#                 req_grad=False))