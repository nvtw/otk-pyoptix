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


def test_dlss_contexts_share_sdk_lifetime():
    if not optix.dlss_support_available():
        pytest.skip("DLSS bindings are not compiled into this wheel")

    first = optix.DlssRRContext()
    second = optix.DlssRRContext()
    try:
        try:
            first.init()
        except RuntimeError as error:
            pytest.skip(f"DLSS runtime is unavailable: {error}")
        second.init()
        first.deinit()
        assert isinstance(second.isDlssRRAvailable(), bool)
    finally:
        second.deinit()
        first.deinit()
