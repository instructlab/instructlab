# Standard
from unittest import mock
import functools
import os

# First Party
from instructlab.feature_gates import FeatureGating, FeatureScopes, GatedFeatures


# decorator for tests that are in dev preview scope
def dev_preview(func):
    @mock.patch.dict(
        os.environ,
        {FeatureGating.env_var_name: FeatureScopes.DevPreviewNoUpgrade.value},
    )
    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        return func(*arg, **kwargs)

    return wrapper


@dev_preview
def test_dev_preview_features():
    assert FeatureGating.feature_available(GatedFeatures.RAG)


def test_experimental_features_off_when_environment_variable_not_set():
    assert not FeatureGating.feature_available(GatedFeatures.RAG)


@mock.patch.dict(
    os.environ,
    {FeatureGating.env_var_name: FeatureScopes.Default.value},
)
def test_experimental_features_off_in_default_feature_scope():
    assert not FeatureGating.feature_available(GatedFeatures.RAG)
