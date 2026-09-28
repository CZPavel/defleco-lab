from .base import MethodInfo, ProcessingMethod, ProcessingResult
from .history import FrameHistory
from .multi_frame import *
from .registry import registry
from .single_frame import *

for _method in (
    Gradient,
    Laplacian,
    DifferenceOfGaussians,
    LocalBackgroundResidual,
    StructureTensor,
    GaborBank,
    DirectionalResidual,
    FringeLineGeometry,
    FrameDifference,
    TemporalStatistics,
    TemporalMedianResidual,
    FarnebackFlow,
    DISFlow,
    LocalPhaseCorrelation,
    TemporalFusion,
):
    if _method.info.id not in {i.id for i in registry.infos()}:
        registry.register(_method)


def load_builtin_methods():
    """Import-time registration already occurred; return the shared registry."""
    return registry
