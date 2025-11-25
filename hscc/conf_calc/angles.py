from rdkit import Chem
from rdkit.Chem import TorsionFingerprints, rdMolTransforms
from typing import List, Tuple

# Type: one unique four and its angle
Dihedral = Tuple[Tuple[int, int, int, int], float]


class Conformation:

    @staticmethod
    def find_unique_dihedral_angles(
        mol: Chem.Mol, conf_id: int = -1
    ) -> tuple[List[Dihedral], List[Dihedral]]:
        """
        Returns the unique dihedral angles of the molecule.

        For each torsion bar (one central connection), the following is selected
        one representative equivalent four atoms and
        Its dihedral angle is calculated (in degrees).

        Returns:
            (nonring_dihedrals, ring_dihedrals), where
            nonring_dihedrals: list [(i,j,k,l), angle_deg] for non-cyclic torsion bars
            ring_dihedrals: list [(i,j,k,l), angle_deg] for torsion bars in rings

        !!! Return angles also for hard fragments such as CON !!! rewrite?
        """

        # Lists of groups of equivalent torsion bars: non-ring and ring
        tors_nonring, tors_ring = TorsionFingerprints.CalculateTorsionLists(mol)
        conf = mol.GetConformer(conf_id)

        def _representative_dihedrals(
            torsion_groups
        ) -> List[Dihedral]:
            result: List[Dihedral] = []

            for quad_group in torsion_groups:
                # We take one representative four from the group
                rep_quad = quad_group[0][0]
                # Counting the dihedral angle (in degrees)
                angle = rdMolTransforms.GetDihedralDeg(
                    conf, rep_quad[0], rep_quad[1], rep_quad[2], rep_quad[3]
                )
                # Save the tuple (four atoms, angle)
                result.append((tuple(rep_quad), float(angle)))
            return result

        nonring_unique = _representative_dihedrals(tors_nonring)
        ring_unique = _representative_dihedrals(tors_ring)

        return nonring_unique, ring_unique
    
    @staticmethod
    def generate_conformation_with_dihedrals(
        mol: Chem.Mol,
        dihedral_targets: List[Dihedral],
        conf_id: int = -1,
        use_degrees: bool = False,
    ) -> Chem.Mol:
        """
        Creates a new conformation of the molecule by setting the specified dihedral angles.

        dihedral_targets: list of tuples ((i,j,k,l), value),
            where (i,j,k,l) are the indices of atoms, and value is the desired angle
            (in degrees if use_degrees=True, otherwise in radians).

        conf_id: The conformer's ID, the current one is taken by default (usually 0 / -1).

        Returns: a new molecule (a copy of the original one) with modified dihedral angles.
        """

        new_mol = Chem.Mol(mol)
        conf = new_mol.GetConformer(conf_id)

        for quad, value in dihedral_targets:
            i, j, k, l = quad
            if use_degrees:
                rdMolTransforms.SetDihedralDeg(conf, int(i), int(j), int(k), int(l), float(value))
            else:
                rdMolTransforms.SetDihedralRad(conf, int(i), int(j), int(k), int(l), float(value))

        return new_mol
