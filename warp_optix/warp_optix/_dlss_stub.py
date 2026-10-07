"""Harmless DLSS API surface for builds compiled without the DLSS SDK."""

from dataclasses import dataclass
from enum import IntEnum


class DlssPerfQuality(IntEnum):
    MAX_PERF = 0
    BALANCED = 1
    MAX_QUALITY = 2
    ULTRA_PERFORMANCE = 3
    ULTRA_QUALITY = 4
    DLAA = 5


class RayReconstructionHintRenderPreset(IntEnum):
    DEFAULT = 0
    D = 1
    E = 2
    F = 3


class DlssRRResource(IntEnum):
    RESOURCE_COLOR_IN = 0
    RESOURCE_COLOR_OUT = 1
    RESOURCE_DIFFUSE_ALBEDO = 2
    RESOURCE_SPECULAR_ALBEDO = 3
    RESOURCE_NORMALROUGHNESS = 4
    RESOURCE_MOTIONVECTOR = 5
    RESOURCE_LINEARDEPTH = 6
    RESOURCE_SPECULAR_HITDISTANCE = 7


@dataclass
class DlssRRInitInfo:
    inputWidth: int = 0
    inputHeight: int = 0
    outputWidth: int = 0
    outputHeight: int = 0
    quality: DlssPerfQuality = DlssPerfQuality.MAX_QUALITY
    preset: RayReconstructionHintRenderPreset = RayReconstructionHintRenderPreset.F
    mvJittered: bool = False
    lowResolutionMotionVectors: bool = True
    isContentHDR: bool = True
    depthInverted: bool = False
    autoExposure: bool = False
    useHWDepth: bool = False


@dataclass
class DlssRRSupportedSizes:
    minWidth: int = 0
    minHeight: int = 0
    maxWidth: int = 0
    maxHeight: int = 0
    optimalWidth: int = 0
    optimalHeight: int = 0


class DlssRRDenoiser:
    def setResource(self, *args, **kwargs):
        pass

    def resetResource(self, *args, **kwargs):
        pass

    def denoise(self, *args, **kwargs):
        pass

    def deinit(self):
        pass


class DlssRRContext:
    def init(self, *args, **kwargs):
        pass

    def deinit(self):
        pass

    def isDlssRRAvailable(self):
        return False

    def getMinDriverVersion(self):
        return (0, 0)

    def querySupportedDlssInputSizes(self, *args, **kwargs):
        return DlssRRSupportedSizes()

    def initDlssRR(self, *args, **kwargs):
        return DlssRRDenoiser()


def dlssRRGetResultString(ngxResultCode):
    return "DLSS support is not compiled into this build"


__all__ = [
    "DlssPerfQuality",
    "RayReconstructionHintRenderPreset",
    "DlssRRResource",
    "DlssRRInitInfo",
    "DlssRRSupportedSizes",
    "DlssRRDenoiser",
    "DlssRRContext",
    "dlssRRGetResultString",
]
