"""Wasserstein--Fisher--Rao Mean Flow Matching."""

from .inference import rollout
from .models import MeanFlowModel

__all__ = ["MeanFlowModel", "rollout"]
__version__ = "0.1.0"
