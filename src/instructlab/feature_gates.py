# Standard
from enum import Enum
import os


class FeatureScopes(str, Enum):
    Default = "Default"
    DevPreviewNoUpgrade = "DevPreviewNoUpgrade"
    TechPreviewNoUpgrade = "TechPreviewNoUpgrade"
    CustomNoUpgrade = "CustomNoUpgrade"


class GatedFeatures(str, Enum):
    RAG = "RAG"


class FeatureGating:
    env_var_name = "ILAB_FEATURE_SCOPE"
    _feature_scopes = {GatedFeatures.RAG: FeatureScopes.DevPreviewNoUpgrade}

    @staticmethod
    def available_scopes():
        return [f.value for f in FeatureScopes]

    @staticmethod
    def feature_available(feature: GatedFeatures):
        return FeatureGating._feature_scopes.get(feature, None) == os.environ.get(
            FeatureGating.env_var_name, FeatureScopes.Default
        )
