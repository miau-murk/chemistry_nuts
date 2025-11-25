import numpy as np
from rdkit import Chem
from .conf_calc import ConfCalc
from .angles import Conformation

# mol_file = "test0.mol"
mol_file = "test.mol"

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

print(calculator.get_energy(np.array([1.0]), req_grad=True, req_opt=False)) # for sampling
