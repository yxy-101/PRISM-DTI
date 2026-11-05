"""Top-level EviDTI-style backbone with TAPB-inspired debiasing.

The raw branch remains unchanged. The causal branch adds a lightweight
target-prior confounder alignment path and evidence pooling posterior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from src.models.cam import CausalAttentionModule
from src.models.confounder_dictionary import ConfounderDictionary
from src.models.drug2d_encoder import Drug2DEncoder
from src.models.drug3d_encoder import Drug3DEncoder
from src.models.drug_conditioned_protein import DrugConditionedProteinAttention
from src.models.evidential_head import EvidentialHead
from src.models.fusion import FeatureFusion
from src.models.protein_encoder import ProteinEncoder


class PRISMDTI(nn.Module):
    """Raw backbone plus target-prior causal evidential branch."""

    raw_outputs = ("alpha_raw", "p_raw", "u_raw")
    causal_outputs = ("T_ci", "alpha_ci", "alpha_do", "p_do", "u_do")

    def __init__(
        self,
        protein_input_dim: int = 1024,
        protein_hidden_dim: int = 128,
        drug2d_input_dim: int = 768,
        drug2d_hidden_dim: int = 128,
        drug3d_input_dim: int = 300,
        drug3d_hidden_dim: int = 128,
        use_drug3d: bool = True,
        fusion_hidden_dim: int = 256,
        evidential_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        protein_num_conv_blocks: int = 2,
        protein_conv_kernel_size: int = 9,
        protein_use_stats_pool: bool = True,
        protein_feature_token_count: int = 1,
        protein_transformer_layers: int = 0,
        protein_transformer_heads: int = 4,
        protein_pooling_mode: str = "attention",
        drug2d_num_conv_blocks: int = 2,
        drug2d_conv_kernel_size: int = 5,
        drug2d_use_attention_pool: bool = True,
        drug2d_feature_token_count: int = 1,
        drug2d_transformer_layers: int = 0,
        drug2d_transformer_heads: int = 4,
        drug3d_num_conv_blocks: int = 2,
        drug3d_conv_kernel_size: int = 5,
        drug3d_use_attention_pool: bool = True,
        fusion_mode: str = "concat",
        use_drug_conditioned_protein_attention: bool = False,
        drug_conditioned_protein_update_mode: str = "gated_blend",
        encoder_architecture: str = "enhanced",
        confounder_dim: Optional[int] = None,
        cam_hidden_dim: Optional[int] = None,
        cam_temperature: float = 1.0,
        cluster_centers_path: Optional[str] = None,
        p_ci_path: Optional[str] = None,
        num_fallback_confounders: int = 1,
        backdoor_intervention_strength: float = 1.0,
        use_sample_residual_backdoor: bool = False,
        sample_residual_backdoor_scale: Optional[float] = None,
        backdoor_confounder_layernorm: Optional[bool] = None,
        cam_mode: str = "causal_attention",
        backdoor_pooling_mode: str = "prior_weighted",
        backdoor_aggregation_space: str = "evidence",
        enable_causal_branch: bool = True,
        enable_randomization: bool = True,
        use_cam: bool = True,
        use_backdoor: bool = True,
        sample_residual_mode: str = "centered",
    ) -> None:
        """Create raw and causal-branch model components."""
        super().__init__()
        self.num_classes = num_classes
        self.protein_input_dim = protein_input_dim
        self.drug2d_input_dim = drug2d_input_dim
        self.drug3d_input_dim = drug3d_input_dim
        self.use_drug3d = bool(use_drug3d)
        self.enable_causal_branch = enable_causal_branch
        self.enable_randomization = enable_randomization
        self.use_cam = use_cam
        self.use_backdoor = use_backdoor
        self.use_drug_conditioned_protein_attention = bool(
            use_drug_conditioned_protein_attention
        )
        self.encoder_architecture = _validate_mode(
            encoder_architecture,
            allowed={"enhanced", "legacy"},
            name="encoder_architecture",
        )
        self.backdoor_intervention_strength = _clamp_unit_interval(
            backdoor_intervention_strength
        )
        if sample_residual_backdoor_scale is None:
            residual_scale = 1.0 if use_sample_residual_backdoor else 0.0
        else:
            residual_scale = _clamp_unit_interval(
                sample_residual_backdoor_scale
            )
        self.sample_residual_backdoor_scale = residual_scale
        self.sample_residual_mode = _validate_mode(
            sample_residual_mode,
            allowed={"centered", "uncentered", "prior_mean"},
            name="sample_residual_mode",
        )
        self.use_sample_residual_backdoor = residual_scale > 0.0
        # Preserve historical behavior when the new switch is omitted: the
        # confounder path was normalized only when a sample residual existed.
        # An explicit value makes residual and normalization independently
        # controllable for the submission attribution experiment.
        self.backdoor_confounder_layernorm = (
            self.use_sample_residual_backdoor
            if backdoor_confounder_layernorm is None
            else bool(backdoor_confounder_layernorm)
        )
        self.cam_mode = _validate_mode(
            cam_mode,
            allowed={"causal_attention", "standard_attention"},
            name="cam_mode",
        )
        self.backdoor_pooling_mode = _validate_mode(
            backdoor_pooling_mode,
            allowed={"prior_weighted", "uniform"},
            name="backdoor_pooling_mode",
        )
        self.backdoor_aggregation_space = _validate_mode(
            backdoor_aggregation_space,
            allowed={"evidence", "alpha", "probability"},
            name="backdoor_aggregation_space",
        )
        self.protein_encoder = ProteinEncoder(
            input_dim=protein_input_dim,
            hidden_dim=protein_hidden_dim,
            kernel_size=protein_conv_kernel_size,
            dropout=dropout,
            num_conv_blocks=protein_num_conv_blocks,
            use_stats_pool=protein_use_stats_pool,
            feature_token_count=protein_feature_token_count,
            transformer_layers=protein_transformer_layers,
            transformer_heads=protein_transformer_heads,
            architecture=self.encoder_architecture,
            pooling_mode=protein_pooling_mode,
        )
        self.drug2d_encoder = Drug2DEncoder(
            input_dim=drug2d_input_dim,
            hidden_dim=drug2d_hidden_dim,
            dropout=dropout,
            num_conv_blocks=drug2d_num_conv_blocks,
            conv_kernel_size=drug2d_conv_kernel_size,
            use_attention_pool=drug2d_use_attention_pool,
            feature_token_count=drug2d_feature_token_count,
            transformer_layers=drug2d_transformer_layers,
            transformer_heads=drug2d_transformer_heads,
            architecture=self.encoder_architecture,
        )
        self.drug3d_encoder = None
        if self.use_drug3d:
            self.drug3d_encoder = Drug3DEncoder(
                input_dim=drug3d_input_dim,
                hidden_dim=drug3d_hidden_dim,
                dropout=dropout,
                num_conv_blocks=drug3d_num_conv_blocks,
                conv_kernel_size=drug3d_conv_kernel_size,
                use_attention_pool=drug3d_use_attention_pool,
                architecture=self.encoder_architecture,
            )
        self.drug_conditioned_protein_attention = None
        if self.use_drug_conditioned_protein_attention:
            self.drug_conditioned_protein_attention = (
                DrugConditionedProteinAttention(
                    protein_dim=protein_hidden_dim,
                    drug2d_dim=drug2d_hidden_dim,
                    drug3d_dim=(drug3d_hidden_dim if self.use_drug3d else 0),
                    dropout=dropout,
                    update_mode=drug_conditioned_protein_update_mode,
                )
            )
        fusion_input_dims = [protein_hidden_dim, drug2d_hidden_dim]
        if self.use_drug3d:
            fusion_input_dims.append(drug3d_hidden_dim)
        self.fusion = FeatureFusion(
            input_dims=tuple(fusion_input_dims),
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            mode=fusion_mode,
        )
        self.evidential_head = EvidentialHead(
            input_dim=fusion_hidden_dim,
            hidden_dim=evidential_hidden_dim,
            num_classes=num_classes,
        )
        self.cam = CausalAttentionModule(
            feature_dim=protein_hidden_dim,
            confounder_dim=confounder_dim or protein_hidden_dim,
            hidden_dim=cam_hidden_dim or protein_hidden_dim,
            temperature=cam_temperature,
        )
        self.confounder_dictionary = ConfounderDictionary(
            cluster_centers_path=cluster_centers_path,
            p_ci_path=p_ci_path,
            num_fallback_confounders=num_fallback_confounders,
        )
        self.causal_fusion = FeatureFusion(
            input_dims=tuple(fusion_input_dims),
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            mode=fusion_mode,
        )
        self.causal_evidential_head = EvidentialHead(
            input_dim=fusion_hidden_dim,
            hidden_dim=evidential_hidden_dim,
            num_classes=num_classes,
        )

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "PRISMDTI":
        """Build the raw-vs-causal model from a nested config dictionary."""
        config = config or {}
        model_config = config.get("model", {})
        ablation_config = config.get("ablation", {})
        debiasing_config = config.get("debiasing", {})
        confounder_config = config.get("confounder", {})
        randomization_config = config.get("randomization", {})
        feature_config = config.get("features", {})
        use_drug3d = _as_bool(
            model_config.get(
                "use_drug3d",
                feature_config.get(
                    "use_drug3d",
                    config.get("use_drug3d", True),
                ),
            )
        )
        causal_enabled = _as_bool(
            config.get(
                "use_causal_branch",
                ablation_config.get(
                    "causal_branch",
                    debiasing_config.get("enabled", True),
                ),
            )
        )
        randomization_enabled = _as_bool(
            config.get(
                "use_randomization",
                ablation_config.get(
                    "target_randomization", randomization_config.get("enabled", True)
                ),
            )
        )
        cam_enabled = causal_enabled and _as_bool(
            config.get(
                "use_cam",
                ablation_config.get("cam", debiasing_config.get("use_cam", True)),
            )
        )
        backdoor_enabled = causal_enabled and _as_bool(
            config.get(
                "use_backdoor",
                ablation_config.get(
                    "backdoor", debiasing_config.get("use_backdoor", True)
                ),
            )
        )
        return cls(
            protein_input_dim=int(model_config.get("protein_input_dim", 1024)),
            protein_hidden_dim=int(model_config.get("protein_hidden_dim", 128)),
            drug2d_input_dim=int(model_config.get("drug2d_input_dim", 768)),
            drug2d_hidden_dim=int(model_config.get("drug2d_hidden_dim", 128)),
            drug3d_input_dim=int(model_config.get("drug3d_input_dim", 300)),
            drug3d_hidden_dim=int(model_config.get("drug3d_hidden_dim", 128)),
            use_drug3d=use_drug3d,
            fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", 256)),
            evidential_hidden_dim=int(
                model_config.get("evidential_hidden_dim", 128)
            ),
            num_classes=int(model_config.get("num_classes", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
            protein_num_conv_blocks=int(
                model_config.get("protein_num_conv_blocks", 2)
            ),
            protein_conv_kernel_size=int(
                model_config.get("protein_conv_kernel_size", 9)
            ),
            protein_use_stats_pool=_as_bool(
                model_config.get("protein_use_stats_pool", True)
            ),
            protein_feature_token_count=int(
                model_config.get("protein_feature_token_count", 1)
            ),
            protein_transformer_layers=int(
                model_config.get("protein_transformer_layers", 0)
            ),
            protein_transformer_heads=int(
                model_config.get("protein_transformer_heads", 4)
            ),
            protein_pooling_mode=str(
                model_config.get("protein_pooling_mode", "attention")
            ),
            drug2d_num_conv_blocks=int(
                model_config.get("drug2d_num_conv_blocks", 2)
            ),
            drug2d_conv_kernel_size=int(
                model_config.get("drug2d_conv_kernel_size", 5)
            ),
            drug2d_use_attention_pool=_as_bool(
                model_config.get("drug2d_use_attention_pool", True)
            ),
            drug2d_feature_token_count=int(
                model_config.get("drug2d_feature_token_count", 1)
            ),
            drug2d_transformer_layers=int(
                model_config.get("drug2d_transformer_layers", 0)
            ),
            drug2d_transformer_heads=int(
                model_config.get("drug2d_transformer_heads", 4)
            ),
            drug3d_num_conv_blocks=int(
                model_config.get("drug3d_num_conv_blocks", 2)
            ),
            drug3d_conv_kernel_size=int(
                model_config.get("drug3d_conv_kernel_size", 5)
            ),
            drug3d_use_attention_pool=_as_bool(
                model_config.get("drug3d_use_attention_pool", True)
            ),
            fusion_mode=str(model_config.get("fusion_mode", "concat")),
            use_drug_conditioned_protein_attention=_as_bool(
                model_config.get(
                    "use_drug_conditioned_protein_attention",
                    False,
                )
            ),
            drug_conditioned_protein_update_mode=str(
                model_config.get(
                    "drug_conditioned_protein_update_mode",
                    "gated_blend",
                )
            ),
            encoder_architecture=str(
                model_config.get("encoder_architecture", "enhanced")
            ),
            confounder_dim=int(
                debiasing_config.get(
                    "confounder_dim",
                    confounder_config.get(
                        "confounder_dim", model_config.get("protein_hidden_dim", 128)
                    ),
                )
            ),
            cam_hidden_dim=int(
                debiasing_config.get(
                    "cam_hidden_dim", model_config.get("protein_hidden_dim", 128)
                )
            ),
            cam_temperature=float(debiasing_config.get("cam_temperature", 1.0)),
            cluster_centers_path=debiasing_config.get(
                "cluster_centers_path",
                confounder_config.get("cluster_centers_path"),
            ),
            p_ci_path=debiasing_config.get(
                "p_ci_path",
                confounder_config.get("p_ci_path"),
            ),
            num_fallback_confounders=int(
                debiasing_config.get("num_fallback_confounders", 1)
            ),
            backdoor_intervention_strength=float(
                debiasing_config.get("backdoor_intervention_strength", 1.0)
            ),
            use_sample_residual_backdoor=_as_bool(
                debiasing_config.get("use_sample_residual_backdoor", False)
            ),
            sample_residual_backdoor_scale=(
                float(debiasing_config["sample_residual_backdoor_scale"])
                if "sample_residual_backdoor_scale" in debiasing_config
                else None
            ),
            sample_residual_mode=str(debiasing_config.get("sample_residual_mode", "centered")),
            backdoor_confounder_layernorm=(
                _as_bool(debiasing_config["backdoor_confounder_layernorm"])
                if debiasing_config.get("backdoor_confounder_layernorm")
                is not None
                else None
            ),
            cam_mode=str(debiasing_config.get("cam_mode", "causal_attention")),
            backdoor_pooling_mode=str(
                debiasing_config.get(
                    "backdoor_pooling_mode",
                    "prior_weighted",
                )
            ),
            backdoor_aggregation_space=str(
                debiasing_config.get(
                    "backdoor_aggregation_space",
                    config.get("backdoor_aggregation_space", "evidence"),
                )
            ),
            enable_causal_branch=causal_enabled,
            enable_randomization=randomization_enabled,
            use_cam=cam_enabled,
            use_backdoor=backdoor_enabled,
        )

    def module_switches(self) -> Dict[str, Any]:
        """Return active ablation switches for logging and reproducibility."""
        return {
            "drug3d": self.use_drug3d,
            "causal_branch": self.enable_causal_branch,
            "target_randomization": self.enable_randomization,
            "cam": self.use_cam and self.enable_causal_branch,
            "backdoor": self.use_backdoor and self.enable_causal_branch,
            "backdoor_aggregation_space": self.backdoor_aggregation_space,
            "sample_residual_backdoor": (
                self.use_sample_residual_backdoor
                and self.use_backdoor
                and self.enable_causal_branch
            ),
            "backdoor_confounder_layernorm": (
                self.backdoor_confounder_layernorm
                and self.use_backdoor
                and self.enable_causal_branch
            ),
            "drug_conditioned_protein_attention": (
                self.use_drug_conditioned_protein_attention
            ),
        }

    def required_external_features(self) -> List[Dict[str, str]]:
        """Describe external feature files required by real branch runs."""
        features = [
            {
                "batch_key": "protein_feature_path",
                "source": "ProtTrans protein embeddings",
                "formats": ".pt/.pth/.npy",
                "shape": "[seq_len, protein_input_dim] or [batch, seq_len, protein_input_dim]",
            },
            {
                "batch_key": "drug2d_feature_path",
                "source": "MG-BERT drug token embeddings",
                "formats": ".pt/.pth/.npy",
                "shape": "[seq_len, drug2d_input_dim] or [batch, seq_len, drug2d_input_dim]",
            },
        ]
        if self.use_drug3d:
            features.append(
                {
                    "batch_key": "drug3d_feature_path",
                    "source": "drug 3D feature bank",
                    "formats": ".pt/.pth/.npy",
                    "shape": "[num_nodes, drug3d_input_dim], [batch, num_nodes, drug3d_input_dim], or pooled [batch, drug3d_input_dim]",
                }
            )
        if not self.enable_causal_branch:
            return features
        features.extend(
            [
                {
                    "batch_key": "cluster_centers_path",
                    "source": "KMeans target-prior confounder centers",
                    "formats": ".pt/.pth",
                    "shape": "[num_clusters, confounder_dim]",
                },
                {
                    "batch_key": "p_ci_path",
                    "source": "Cluster prior weights from training targets",
                    "formats": ".pt/.pth",
                    "shape": "[num_clusters]",
                },
            ]
        )
        return features

    def forward(
        self, batch: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """Run raw and causal branches while preserving raw branch outputs."""
        if not batch:
            batch = self._build_dummy_batch(batch_size=1)
        drug2d_repr = self.drug2d_encoder(batch)
        drug3d_repr = None
        if self.use_drug3d:
            drug3d_repr = self.drug3d_encoder(batch)
        if self.use_drug_conditioned_protein_attention:
            residue_tokens, residue_mask = self.protein_encoder.encode_sequence(batch)
            pooled_protein = self.protein_encoder.pool_sequence(
                residue_tokens,
                mask=residue_mask,
            )
            protein_repr, _ = self.drug_conditioned_protein_attention(
                residue_tokens=residue_tokens,
                pooled_protein=pooled_protein,
                drug2d=drug2d_repr,
                drug3d=self._drug3d_or_empty(drug2d_repr, drug3d_repr),
                residue_mask=residue_mask,
            )
        else:
            protein_repr = self.protein_encoder(batch)
        fused = self.fusion(
            *self._fusion_features(protein_repr, drug2d_repr, drug3d_repr)
        )
        outputs = self.evidential_head(fused)
        if not self.enable_causal_branch:
            return outputs
        causal_outputs = self._run_causal_branch(
            protein_repr=protein_repr,
            drug2d_repr=drug2d_repr,
            drug3d_repr=drug3d_repr,
        )
        outputs.update(causal_outputs)
        if "p_do" in outputs:
            outputs["pcs"] = outputs["p_raw"][:, -1] - outputs["p_do"][:, -1]
        return outputs

    def _run_causal_branch(
        self,
        protein_repr: torch.Tensor,
        drug2d_repr: torch.Tensor,
        drug3d_repr: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run confounder alignment and backdoor evidence pooling."""
        centers, p_ci = self.confounder_dictionary.get_bank(protein_repr)
        if self.use_cam:
            attention_prior = (
                p_ci if self.cam_mode == "causal_attention" else None
            )
            cam_outputs = self.cam(
                target_features=protein_repr,
                confounder_centers=centers,
                p_ci=attention_prior,
            )
            t_ci = cam_outputs["T_ci"]
            confounder_context = cam_outputs["confounder_context"]
            confounder_values = cam_outputs["confounder_values"]
        else:
            t_ci = protein_repr
            confounder_values = self._confounder_values_without_cam(
                centers=centers,
                reference=protein_repr,
            )
            confounder_context = torch.sum(
                confounder_values * p_ci.reshape(-1, 1),
                dim=0,
                keepdim=True,
            ).expand_as(protein_repr)
        alpha_ci_raw = self.causal_evidential_head(
            self.causal_fusion(
                *self._fusion_features(t_ci, drug2d_repr, drug3d_repr)
            )
        )["alpha_raw"]

        if self.use_backdoor:
            sample_residual = None
            if self.use_sample_residual_backdoor:
                sample_residual = self._sample_residual(
                    protein_repr, confounder_context, confounder_values, p_ci
                )
            alpha_by_confounder = self._alpha_for_confounders(
                confounder_values=confounder_values,
                drug2d_repr=drug2d_repr,
                drug3d_repr=drug3d_repr,
                sample_residual=sample_residual,
            )
            pooling_weights = self._pooling_weights(p_ci)
            if self.backdoor_aggregation_space == "evidence":
                alpha_do = self.confounder_dictionary.backdoor_evidence_pooling(
                    alpha_by_confounder=alpha_by_confounder,
                    p_ci=pooling_weights,
                )
            elif self.backdoor_aggregation_space == "alpha":
                alpha_do = self.confounder_dictionary.backdoor_alpha_pooling(
                    alpha_by_confounder=alpha_by_confounder,
                    p_ci=pooling_weights,
                )
            else:
                alpha_do = self.confounder_dictionary.backdoor_probability_pooling(
                    alpha_by_confounder=alpha_by_confounder,
                    p_ci=pooling_weights,
                )
            if self.backdoor_intervention_strength < 1.0:
                strength = self.backdoor_intervention_strength
                alpha_do = strength * alpha_do + (1.0 - strength) * alpha_ci_raw
        else:
            alpha_do = alpha_ci_raw
        p_do, u_do = self._probability_and_uncertainty(alpha_do)
        return {
            "T_ci": t_ci,
            "alpha_ci": alpha_ci_raw,
            "alpha_do": alpha_do,
            "p_do": p_do,
            "u_do": u_do,
        }

    def _sample_residual(self, protein_repr, context, values, prior):
        """Matched, parameter-free alternatives; no batch-dependent shuffling."""
        if self.sample_residual_mode == "uncentered":
            residual = protein_repr
        elif self.sample_residual_mode == "prior_mean":
            weights = prior / prior.sum()
            global_context = (values * weights.reshape(-1, 1)).sum(0, keepdim=True)
            residual = protein_repr - global_context
        else:
            residual = protein_repr - context
        return self.sample_residual_backdoor_scale * residual

    def _alpha_for_confounders(
        self,
        confounder_values: torch.Tensor,
        drug2d_repr: torch.Tensor,
        drug3d_repr: Optional[torch.Tensor],
        sample_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-confounder evidential parameters for evidence pooling."""
        batch_size = drug2d_repr.size(0)
        num_confounders = confounder_values.size(0)
        expanded_confounders = confounder_values.unsqueeze(0).expand(
            batch_size, num_confounders, -1
        )
        if sample_residual is not None:
            if sample_residual.shape != (batch_size, confounder_values.size(-1)):
                raise ValueError(
                    "sample_residual must be [batch, protein_hidden_dim]."
                )
            expanded_confounders = (
                expanded_confounders + sample_residual.unsqueeze(1)
            )
        if self.backdoor_confounder_layernorm:
            expanded_confounders = self.cam.output_norm(expanded_confounders)
        expanded_drug2d = drug2d_repr.unsqueeze(1).expand(
            batch_size, num_confounders, -1
        )
        flat_confounders = expanded_confounders.reshape(batch_size * num_confounders, -1)
        flat_drug2d = expanded_drug2d.reshape(batch_size * num_confounders, -1)
        flat_drug3d = None
        if self.use_drug3d:
            if drug3d_repr is None:
                raise ValueError("drug3d_repr is required when use_drug3d=True.")
            expanded_drug3d = drug3d_repr.unsqueeze(1).expand(
                batch_size, num_confounders, -1
            )
            flat_drug3d = expanded_drug3d.reshape(
                batch_size * num_confounders, -1
            )
        flat_fused = self.causal_fusion(
            *self._fusion_features(
                flat_confounders,
                flat_drug2d,
                flat_drug3d,
            )
        )
        flat_alpha = self.causal_evidential_head(flat_fused)["alpha_raw"]
        return flat_alpha.reshape(batch_size, num_confounders, self.num_classes)

    def _confounder_values_without_cam(
        self,
        centers: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return confounder values for the no-CAM ablation."""
        if centers.size(-1) == reference.size(-1):
            return centers.float()
        return self.cam.project_confounders(centers)

    def _fusion_features(
        self,
        protein_repr: torch.Tensor,
        drug2d_repr: torch.Tensor,
        drug3d_repr: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Return the active modalities without a learnable zero placeholder."""
        features = [protein_repr, drug2d_repr]
        if self.use_drug3d:
            if drug3d_repr is None:
                raise ValueError("drug3d_repr is required when use_drug3d=True.")
            features.append(drug3d_repr)
        return tuple(features)

    @staticmethod
    def _drug3d_or_empty(
        drug2d_repr: torch.Tensor,
        drug3d_repr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return a zero-width tensor for the optional attention query input."""
        if drug3d_repr is not None:
            return drug3d_repr
        return drug2d_repr.new_empty((drug2d_repr.size(0), 0))

    def _probability_and_uncertainty(
        self, alpha: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive class probabilities and evidential uncertainty from alpha."""
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True).clamp_min(1e-8)
        probabilities = alpha / alpha_sum
        uncertainty = float(self.num_classes) / alpha_sum
        return probabilities, uncertainty

    def _pooling_weights(self, p_ci: torch.Tensor) -> torch.Tensor:
        """Return prior-weighted or ordinary uniform backdoor weights."""
        if self.backdoor_pooling_mode == "prior_weighted":
            return p_ci
        return torch.ones_like(p_ci)

    def _build_dummy_batch(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Create synthetic feature tensors for smoke tests and dry runs."""
        device = next(self.parameters()).device
        batch = {
            "protein_features": torch.randn(
                batch_size, 32, self.protein_input_dim, device=device
            ),
            "drug2d_features": torch.randn(
                batch_size, 24, self.drug2d_input_dim, device=device
            ),
        }
        if self.use_drug3d:
            batch["drug3d_features"] = torch.randn(
                batch_size, 16, self.drug3d_input_dim, device=device
            )
        return batch


def _as_bool(value: Any) -> bool:
    """Parse booleans from YAML/JSON scalar values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clamp_unit_interval(value: float) -> float:
    """Clamp a scalar config value to [0, 1]."""
    return min(1.0, max(0.0, float(value)))


def _validate_mode(value: str, allowed: set[str], name: str) -> str:
    """Normalize and validate a finite configuration mode."""
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(
            "%s must be one of %s, got %r."
            % (name, sorted(allowed), value)
        )
    return normalized
