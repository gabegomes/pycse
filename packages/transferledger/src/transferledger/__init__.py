"""transferledger: readable equations from tabular data, and which parts of a model still
hold under a named shift."""

from .curves import KANAdditive, ModelCurves, SplineAdditive
from .detection import Detection, detectable
from .forms import FORMS, Form, Piece, match_curve, register_form
from .learning import LearningCurve, backtest, learning_curve
from .ledger import Ledger, transfer_ledger
from .model import PieceModel
from .preflight import Preflight, preflight
from .shifts import BeyondRange, ForwardInTime, NewGroups, Reference, Shift, Splitter
from .truth import scorecard

__version__ = "0.1.0"
__all__ = [
    "transfer_ledger", "Ledger", "learning_curve", "LearningCurve", "backtest", "detectable", "Detection", "preflight", "Preflight", "PieceModel", "SplineAdditive", "KANAdditive", "ModelCurves", "Form", "Piece", "FORMS",
    "register_form", "match_curve", "Shift", "Reference", "NewGroups", "ForwardInTime",
    "BeyondRange", "Splitter", "scorecard",
]  # fmt: skip
