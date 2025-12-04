import copy
from typing import Tuple, Union, List, Optional, Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy import constants                                                      

from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MINIMAL, VERBOSITY_FULL, VERBOSITY_MUTED
from xtb.utils import get_method, Solvent

from .wilson_b_matrix import (
    Dihedral,
    get_current_derivative,
)

ANGSTROM_IN_BOHR = constants.physical_constants['Bohr radius'][0] * 1.0e10

class ConfCalc:
    def __init__(
        self,
        mol_file_name: str = None,
        mol: Chem.Mol = None,
        rotable_dihedral_idxs: List[List[int]] = None,
        dir_to_xyzs: str = "",
        charge: int = 0,
        gfn_method: int = 2,
        timeout: int = 250,
        norm_en: float = 0.0,
    ):
        """
        Class that calculates energy of current molecule
        with selected values of dihedral angles.

        Parameters
        ----------
        mol_file_name : str, optional
            Path to .mol file.
        mol : Chem.Mol, optional
            RDKit molecule object.
        rotable_dihedral_idxs : list[list[int]]
            List of 4-element lists with zero-based atom indices
            for dihedral angles.
        dir_to_xyzs : str
            Kept for API compatibility, not used by xtb-python backend.
        charge : int
            Molecular charge for xTB.
        gfn_method : int
            GFN method index for xTB (0, 1, 2).
        timeout : int
            Kept for API compatibility, not used by xtb-python backend.
        norm_en : float
            Reference energy to subtract from raw xTB energy.
        """

        assert (mol_file_name is not None or mol is not None), "No mol selected!"
        assert rotable_dihedral_idxs is not None, "No idxs to rotate have been selected!"

        if mol_file_name is None:
            self.mol = mol
        elif mol is None:
            self.mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)

        self.rotable_dihedral_idxs = rotable_dihedral_idxs

        # dir_to_xyzs и timeout сохраняем как атрибуты для обратной совместимости,
        # но в реализации через xtb-python они не используются
        self.dir_to_xyzs = dir_to_xyzs
        self.charge = charge
        self.gfn_method = gfn_method
        self.timeout = timeout
        self.norm_en = norm_en

        # Id for potential future use (сохраняем для совместимости, но не используем)
        self.current_id = 0

    def set_norm_en(self, norm_en: float = 0.0) -> None:
        """
        Update reference (normalization) energy.
        """
        self.norm_en = norm_en

    def __setup_dihedrals(self, values: List[float]) -> Chem.Mol:
        """
        Return a new molecule with selected dihedral angles.

        Parameters
        ----------
        values : list[float]
            List of angles in radians, one per dihedral in rotable_dihedral_idxs.
        """
        assert len(values) == len(self.rotable_dihedral_idxs), (
            "Number of values must be equal to the number of dihedral angles"
        )

        new_mol = copy.deepcopy(self.mol)
        conf = new_mol.GetConformer()

        for idxs, value in zip(self.rotable_dihedral_idxs, values):
            rdMolTransforms.SetDihedralRad(conf, *idxs, value)

        return new_mol

    def __rdkit_to_xtb_arrays(
        self,
        mol: Chem.Mol,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert RDKit molecule to xTB arrays:
        atomic numbers and Cartesian coordinates.
        """
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        numbers = np.empty(num_atoms, dtype=int)
        positions = np.empty((num_atoms, 3), dtype=float)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            numbers[i] = atom.GetAtomicNum()
            positions[i, 0] = pos.x
            positions[i, 1] = pos.y
            positions[i, 2] = pos.z

        return numbers, positions

    def __get_xtb_method(self):
        """
        Map integer gfn_method to xtb-python method descriptor.
        """

        if self.gfn_method == 0:
            name = "GFN0-xTB"
        elif self.gfn_method == 1:
            name = "GFN1-xTB"
        else:
            name = "GFN2-xTB"

        method = get_method(name)
        if method is None:
            raise ValueError(f"Unsupported GFN method name for gfn_method={self.gfn_method}")
        return method

    def __run_xtb(
        self,
        mol: Chem.Mol,
        req_grad: bool = True,
    ) -> tuple[float | None, np.ndarray | None]:
        
        numbers, positions = self.__rdkit_to_xtb_arrays(mol)
        method = self.__get_xtb_method()

        calc = Calculator(method, numbers, positions / ANGSTROM_IN_BOHR, charge=self.charge)
        calc.set_verbosity(VERBOSITY_MUTED)
        calc.set_solvent()

        res = calc.singlepoint()

        energy = float(res.get_energy())

        gradient = None
        if req_grad:
            gradient = np.asarray(res.get_gradient(), dtype=float)

        return energy, gradient


        

    def __calc_energy(
        self,
        mol: Chem.Mol,
        req_opt: bool = True,
        req_grad: bool = True,
    ) -> Tuple[Optional[float], Optional[List[Tuple[List[int], float]]]]:
        """
        Calculate energy (and optionally gradients) of given molecule via xtb-python.

        Parameters
        ----------
        mol : Chem.Mol
            Molecule to evaluate.
        req_opt : bool
            Kept for API compatibility; ignored (single-point calculation is always used).
        req_grad : bool
            Request gradients from xTB.

        Returns
        -------
        (energy, grads) : (float or None, list or None)
            energy : float or None
                Total energy, or None if xTB failed.
            grads : list[ (rotable_idx, dE/dphi) ] or None
                Gradient with respect to each dihedral angle, or None.
        """
        # req_opt игнорируем: в старом коде тоже фактически использовался только single-point.
        energy, cart_grads = self.__run_xtb(mol, req_grad=req_grad)

        if energy is None:
            return None, None

        irc_grad: Optional[List[Tuple[List[int], float]]] = None

        if req_grad and cart_grads is not None:
            irc_grad = []
            flat_grads = cart_grads.flatten()
            for rotable_idx in self.rotable_dihedral_idxs:
                dval = get_current_derivative(
                    mol,
                    flat_grads,
                    Dihedral(*rotable_idx),
                )
                irc_grad.append((rotable_idx, dval))

        return energy, irc_grad

    def get_energy(
        self,
        values: List[float],
        req_opt: bool = True,
        req_grad: bool = True,
    ) -> Dict[str, Union[float, List[Tuple[List[int], float]], None]]:
        """
        Public API: calculate energy (and optionally gradients)
        at given dihedral angles.

        Parameters
        ----------
        values : list[float]
            Dihedral angle values in radians.
        req_opt : bool
            Kept for API compatibility; single-point calculation is always used.
        req_grad : bool
            If True, calculate gradients with respect to dihedral angles.

        Returns
        -------
        dict
            {
                'energy': float or None,
                'grads': list[ (rotable_idx, dE/dphi) ] or None
            }
        """
        # 1. Устанавливаем диэдральные углы в RDKit-молекуле
        mol = self.__setup_dihedrals(values)

        # 2. Вызываем xTB через библиотечный интерфейс
        energy, grad = self.__calc_energy(
            mol,
            req_opt=req_opt,
            req_grad=req_grad,
        )

        if energy is None:
            return {
                "energy": None,
                "grads": None,
            }

        return {
            "energy": energy,
            "grads": grad,
        }
