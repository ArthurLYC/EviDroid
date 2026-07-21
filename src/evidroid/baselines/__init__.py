from evidroid.baselines.static import (
    abstract_api,
    build_drebin_features,
    build_droidapiminer_features,
    build_mamadroid_features_from_apk,
    build_mamadroid_features_from_evidence,
    markov_transition_features,
)
from evidroid.baselines.runner import (
    convert_mamadroid_package_cache_to_family,
    decision_scores,
    evaluate_predictions,
    iter_jsonl,
    load_mamadroid_cache,
    load_sample_index,
    run_prebuilt_sklearn_method,
    run_streamed_sklearn_method,
)
from evidroid.baselines.deep import (
    build_deep_inputs,
    require_torch,
    run_api_transformer,
    run_apppoet_like,
)

METHODS = ("drebin", "droidapiminer", "mamadroid", "apppoet", "api_transformer", "evidroid")
TORCH_METHODS = {"apppoet", "api_transformer"}

__all__ = [
    "METHODS",
    "TORCH_METHODS",
    "abstract_api",
    "build_deep_inputs",
    "build_drebin_features",
    "build_droidapiminer_features",
    "build_mamadroid_features_from_apk",
    "build_mamadroid_features_from_evidence",
    "convert_mamadroid_package_cache_to_family",
    "decision_scores",
    "evaluate_predictions",
    "iter_jsonl",
    "load_mamadroid_cache",
    "load_sample_index",
    "markov_transition_features",
    "require_torch",
    "run_api_transformer",
    "run_apppoet_like",
    "run_prebuilt_sklearn_method",
    "run_streamed_sklearn_method",
]
