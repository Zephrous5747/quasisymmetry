from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent
_EXTERNAL = _REPO_ROOT / "external" / "QuasiSymmetries" / "src"
_QS_PKG = _EXTERNAL / "quasisymmetries"

if not _QS_PKG.is_dir():
    raise ModuleNotFoundError(
        "Missing external/QuasiSymmetries (Python package 'quasisymmetries').\n"
        "  On a git checkout:\n"
        "    git submodule update --init --recursive\n"
        "  Or clone directly (Alliance copies without .git):\n"
        "    git clone https://github.com/Praveen91299/QuasiSymmetries "
        "external/QuasiSymmetries\n"
        f"  Expected package at: {_QS_PKG}"
    )

if str(_EXTERNAL) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL))

from quasisymmetries import Clifford, taper_hamiltonian
from quasisymmetries.state_utils import get_cisd_gs, get_hf_occ, get_hf_wfn
from quasisymmetries.bs.beam import beam_search_symmetries, BeamSearch_Symmetries
from quasisymmetries.bs.utils import mask_to_qubit_operator
from quasisymmetries.metrics import variance
def molecular_data_from_fcidump(*args, **kwargs):
    """Lazily import the PySCF-backed FCIDUMP adapter.

    Clifford-only workflows should not require PySCF merely because they use
    this external-package bootstrap module.
    """
    from fcidump_openfermion import molecular_data_from_fcidump as _load

    return _load(*args, **kwargs)
