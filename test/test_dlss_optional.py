import optix
import pytest


def test_dlss_support_query():
    assert isinstance(optix.dlss_support_available(), bool)


def test_dlss_free_build_has_harmless_api():
    if optix.dlss_support_available():
        pytest.skip("DLSS bindings are compiled into this wheel")

    assert not optix.dlss_rr_available()
    context = optix.DlssRRContext()
    context.init()
    assert not context.isDlssRRAvailable()
    assert context.getMinDriverVersion() == (0, 0)
    sizes = context.querySupportedDlssInputSizes(1920, 1080)
    assert sizes.optimalWidth == 0

    denoiser = context.initDlssRR(optix.DlssRRInitInfo())
    denoiser.setResource(optix.DlssRRResource.RESOURCE_COLOR_IN, 0)
    denoiser.resetResource(optix.DlssRRResource.RESOURCE_COLOR_IN)
    denoiser.denoise(0, 0, 0, 0, [], [])
    denoiser.deinit()
    context.deinit()
