"""Loss package for PRISM-DTI."""

from src.losses.evidential_loss import (
    SUPPORTED_EVIDENTIAL_LOSSES,
    dirichlet_kl_to_uniform,
    evidential_loss,
    remove_true_class_evidence,
)
from src.losses.poer import (
    DEFAULT_CONFIDENCE_FEATURES,
    attach_target_prior_groups,
    build_confidence_features,
    poer_total_loss,
    ppce_proxy_loss,
    prior_adversary_loss,
    target_prior_groups_to_tensor,
)

__all__ = [
    "DEFAULT_CONFIDENCE_FEATURES",
    "SUPPORTED_EVIDENTIAL_LOSSES",
    "attach_target_prior_groups",
    "build_confidence_features",
    "dirichlet_kl_to_uniform",
    "evidential_loss",
    "poer_total_loss",
    "ppce_proxy_loss",
    "prior_adversary_loss",
    "remove_true_class_evidence",
    "target_prior_groups_to_tensor",
]
