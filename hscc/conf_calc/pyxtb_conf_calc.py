import copy
import glob
import os
import shutil
import time
from typing import Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from .wilson_b_matrix import Dihedral, get_current_derivative

# xtb-python imports
from xtb.interface import Calculator, Param, XTBException
from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MUTED

ANGSTROM_TO_BOHR = 1.8897259886


class ConfCalc:
    def __init__(
        self,
        mol_file_name: str = None,
        mol: Chem.Mol = None,
        rotable_dihedral_idxs: list[list[int]] = None,
        charge: int = 0,
        gfn_method: int = 2,
        timeout: int = 250,
        norm_en: float = 0.0,
    ):

        assert (mol_file_name is not None or mol is not None), "No mol selected!"
        assert rotable_dihedral_idxs is not None, "No idxs to rotate have been selected!"

        if mol_file_name is None:
            self.mol = mol
        elif mol is None:
            self.mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)

        self.rotable_dihedral_idxs = rotable_dihedral_idxs

        self.charge = charge
        self.gfn_method = gfn_method
        self.timeout = timeout
        self.norm_en = norm_en

        self.current_id = 0

    def set_norm_en(self, norm_en: float = 0.0):
        """
        Updates norm energy
        """
        self.norm_en = norm_en

    def __setup_dihedrals(self, values: list[float]) -> Chem.Mol:
        """
        Private function that returns a molecule with
        selected dihedral angles

        values - list of angles in radians
        """
        assert len(values) == len(
            self.rotable_dihedral_idxs
        ), "Number of values must be equal to the number of dihedral angles"

        new_mol = copy.deepcopy(self.mol)

        for idxs, value in zip(self.rotable_dihedral_idxs, values):
            rdMolTransforms.SetDihedralRad(new_mol.GetConformer(), *idxs, value)

        return new_mol


    def __get_xtb_param(self) -> Param:
        mapping = {
            0: Param.GFN0xTB,
            1: Param.GFN1xTB,
            2: Param.GFN2xTB,
            3: Param.GFNFF,
            5: Param.IPEAxTB,
        }
        return mapping.get(self.gfn_method, Param.GFN2xTB)

    def __calc_energy(
        self,
        mol: Chem.Mol,
        req_grad: bool = True,
    ) -> Tuple[Union[float, None], Union[list[tuple[list[int], float]], None]]:
        """
        Calculates energy of given molecule via python-xtb.
        returns tuple of energy and gradient
        """
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        numbers = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=int
        )

        positions_ang = np.zeros((num_atoms, 3), dtype=float)
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            positions_ang[i, 0] = pos.x
            positions_ang[i, 1] = pos.y
            positions_ang[i, 2] = pos.z

        positions_bohr = positions_ang * ANGSTROM_TO_BOHR

        param = self.__get_xtb_param()

        calc = Calculator(
            param,
            numbers,
            positions_bohr,
            charge=float(self.charge),
            uhf=None,
        )
        calc.set_verbosity(VERBOSITY_MUTED)

        res = calc.singlepoint()

        energy = res.get_energy()  # Hartree
        irc_grad = None

        if req_grad:
            cart_grads = res.get_gradient()  # shape: (N_atoms, 3), Hartree/Bohr
            irc_grad = []
            flat_grad = cart_grads.flatten()
            for rotable_idx in self.rotable_dihedral_idxs:
                d = get_current_derivative(
                    mol,
                    flat_grad,
                    Dihedral(*rotable_idx),
                )
                irc_grad.append((rotable_idx, d))

        return energy, irc_grad



    def get_energy(
        self,
        values: list[float],
        req_opt: bool = True,
        req_grad: bool = True,
    ) -> dict:
        """
        Returns dict with fields:
        'energy' - energy in this point
        'grads' - list of tuples, consists of
                  pairs of dihedral angle atom indexes and
                  gradients of energy with respect to this angle
        """


        mol = self.__setup_dihedrals(values)

        energy, grad = self.__calc_energy(
            mol,
            req_grad=req_grad,
        )

        if energy:
            energy -= self.norm_en

        return {
            "energy": energy,
            "grads": grad,
            "mol": mol
        }
    
    def get_conformation(
        self,
        values: list[float],
    ) -> dict:
        
        mol = self.__setup_dihedrals(values)
        return mol
