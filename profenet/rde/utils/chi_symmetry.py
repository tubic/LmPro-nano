"""Side-chain torsion symmetries using the project's amino-acid vocabulary."""

import math

from rde.utils.protein.constants import AA, chi_pi_periodic


# Chi indices are one-based here to match the notation used in reports.
CHI_SYMMETRY_PERIODS = {
    int(restype): {
        chi_index + 1: math.pi
        for chi_index, is_periodic in enumerate(periodic_flags)
        if is_periodic
    }
    for restype, periodic_flags in chi_pi_periodic.items()
    if any(periodic_flags)
}


def validate_chi_symmetry_periods():
    expected = {
        int(AA.ASP): {2: math.pi},
        int(AA.GLU): {3: math.pi},
        int(AA.PHE): {2: math.pi},
        int(AA.TYR): {2: math.pi},
    }
    if CHI_SYMMETRY_PERIODS != expected:
        raise RuntimeError(
            "Project chi periodicity metadata does not match the expected vocabulary"
        )


validate_chi_symmetry_periods()
