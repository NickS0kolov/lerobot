"""Patch alignment without downloading model weights."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, alignment_loss
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class TinyVLM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.visual = nn.Linear(3, 8)
        self.context = nn.Linear(8, 8)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.expert_hidden_size = 8
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(fake_image_token_id=1, global_image_token_id=2)
        )
        self.vlm = SimpleNamespace(device=torch.device("cpu"))

    def embed_language_tokens(self, tokens):
        return self.embedding(tokens)

    def embed_image(self, image):
        return self.visual(image.flatten(2).transpose(1, 2))

    def forward(self, inputs_embeds, alignment_layer_index=None, **kwargs):
        prefix, suffix = inputs_embeds
        outputs = (self.context(prefix), suffix + prefix.mean(1, keepdim=True))
        if alignment_layer_index is not None:
            return outputs, None, prefix
        return outputs, None


def make_policy(special_tokens=False, enabled=True):
    config = SmolVLAConfig(
        alignment_enabled=enabled,
        train_expert_only=False,
        alignment_teacher_dim=6,
        chunk_size=2,
        n_action_steps=2,
        max_action_dim=2,
        max_state_dim=2,
        resize_imgs_with_padding=(8, 8),
        add_image_special_tokens=special_tokens,
    )
    # Deliberately reverse camera order relative to the teacher mapping.
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    with patch("lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", TinyVLM):
        return SmolVLAPolicy(config)


def make_batch():
    return {
        OBS_STATE: torch.randn(2, 2),
        ACTION: torch.randn(2, 2, 2),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        "observation.images.image": torch.rand(2, 3, 8, 8),
        "observation.images.wrist_image": torch.rand(2, 3, 8, 8),
        "observation.teacher.image": torch.randn(2, 64, 6),
        "observation.teacher.wrist_image": torch.randn(2, 64, 6),
    }


@pytest.mark.parametrize("special_tokens", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_alignment_selects_post_vlm_patches_and_backpropagates(special_tokens, reduction):
    policy = make_policy(special_tokens)
    batch = make_batch()
    captured = []
    context_output = []
    hook = policy.model.vlm_with_expert.context.register_forward_hook(
        lambda module, args, output: context_output.append(output)
    )
    real_loss = alignment_loss

    def capture(h, targets, projector, mask, reduction):
        captured.append((h, targets, mask))
        return real_loss(h, targets, projector, mask, reduction)

    with patch("lerobot.policies.smolvla.modeling_smolvla.alignment_loss", side_effect=capture):
        loss, logs = policy(batch, reduction=reduction)
    hook.remove()
    h, target, mask = captured[0]
    start = 2 if special_tokens else 0
    stride = 67 if special_tokens else 64
    # Mapping order is external, wrist; prefix order is wrist, external.
    torch.testing.assert_close(h[:, 0], context_output[0][:, stride + start : stride + start + 64])
    torch.testing.assert_close(h[:, 1], context_output[0][:, start : start + 64])
    torch.testing.assert_close(target[:, 0], batch["observation.teacher.image"])
    assert mask.all()
    assert logs["loss"] == pytest.approx(logs["action_loss"] + 0.2 * logs["alignment_loss"])
    loss.mean().backward()
    assert policy.model.vlm_with_expert.context.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in policy.model.alignment_projector.parameters())
    restored = make_policy(special_tokens)
    restored.load_state_dict(policy.state_dict())
    for original, loaded in zip(
        policy.model.alignment_projector.parameters(),
        restored.model.alignment_projector.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, loaded)


def test_alignment_mask_and_target_detach():
    projector = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 6)).requires_grad_(False)
    h = torch.randn(2, 2, 3, 4, requires_grad=True)
    targets = torch.randn(2, 2, 3, 6, requires_grad=True)
    mask = torch.tensor([[True, False], [False, False]])
    loss = alignment_loss(h, targets, projector, mask)
    expected = alignment_loss(h[:1, :1], targets[:1, :1], projector, mask[:1, :1])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert targets.grad is None
    assert h.grad[0, 0].abs().sum() > 0
    assert h.grad[0, 1].count_nonzero() == 0
    assert h.grad[1].count_nonzero() == 0
    assert alignment_loss(h, targets, projector, torch.zeros_like(mask)).item() == 0


def test_alignment_requires_trainable_vlm():
    with pytest.raises(ValueError, match="train_expert_only=False"):
        SmolVLAConfig(alignment_enabled=True)


def test_alignment_missing_target_and_wrong_patch_count():
    policy = make_policy()
    batch = make_batch()
    batch["observation.teacher.image"] = torch.randn(2, 63, 6)
    with pytest.raises(ValueError, match="expected"):
        policy(batch)
    del batch["observation.teacher.image"]
    with pytest.raises(ValueError, match="Missing alignment target"):
        policy(batch)


def test_alignment_disabled_accepts_batches_without_teacher():
    policy = make_policy(enabled=False)
    batch = {key: value for key, value in make_batch().items() if not key.startswith("observation.teacher.")}
    loss, logs = policy(batch)
    assert loss.isfinite()
    assert "alignment_loss" not in logs
    assert not any("alignment_projector" in key for key in policy.state_dict())


def test_alignment_partial_camera_and_bfloat16_projector():
    policy = make_policy()
    batch = make_batch()
    del batch["observation.images.image"]
    batch["observation.images.wrist_image_padding_mask"] = torch.tensor([True, False])
    batch["observation.teacher.wrist_image"][1] = float("nan")
    policy.model.alignment_projector.bfloat16()
    loss, logs = policy(batch)
    loss.backward()
    assert loss.isfinite()
    assert logs["alignment_loss"] > 0
    grad = policy.model.vlm_with_expert.context.weight.grad
    assert grad.isfinite().all() and grad.abs().sum() > 0


def test_lora_all_linear_preserves_frozen_projector():
    from peft import LoraConfig, get_peft_model_state_dict

    policy = make_policy()
    policy.config.pretrained_path = "lerobot/smolvla_base"
    adapted = policy.wrap_with_peft(LoraConfig(r=2, lora_alpha=2, target_modules="all-linear"))
    projector = policy.model.alignment_projector
    assert all(not parameter.requires_grad for parameter in projector.parameters())
    assert not any("lora_" in name for name, _ in projector.named_parameters())
    assert any("alignment_projector" in key for key in get_peft_model_state_dict(adapted))
    loss, _ = adapted(make_batch())
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in adapted.named_parameters()
        if "context" in name and "lora_" in name
    )


class LayerSelectionVLM(SmolVLMWithExpertModel):
    """Exercise the real forward loop using small blocks and deterministic attention inputs."""

    def __init__(self):
        nn.Module.__init__(self)
        self.num_vlm_layers = 3
        self.num_expert_layers = 3
        self.attention_mode = "self_attn"
        self.self_attn_every_n_layers = 1

        def make_text_model():
            model = nn.Module()
            model.layers = nn.ModuleList()
            model.norm = nn.LayerNorm(8)
            for _ in range(3):
                block = nn.Module()
                block.self_attn = nn.Module()
                block.self_attn.o_proj = nn.Linear(8, 8)
                block.post_attention_layernorm = nn.LayerNorm(8)
                block.mlp = nn.Linear(8, 8)
                model.layers.append(block)
            return model

        self.vlm = nn.Module()
        self.vlm.config = SimpleNamespace(text_config=SimpleNamespace(head_dim=8))
        self.vlm.model = nn.Module()
        self.vlm.model.text_model = make_text_model()
        self.lm_expert = make_text_model()

    def forward_attn_layer(self, model_layers, inputs_embeds, *args, **kwargs):
        return inputs_embeds, None


@pytest.mark.parametrize("index", [0, 1, 2])
def test_selected_layer_is_full_block_before_norm_and_preserves_final_outputs(index):
    model = LayerSelectionVLM()
    inputs = [torch.randn(2, 4, 8), torch.randn(2, 2, 8)]
    baseline, _ = model(inputs_embeds=inputs)
    outputs, _, selected = model(inputs_embeds=inputs, alignment_layer_index=index)
    torch.testing.assert_close(outputs, baseline)
    with torch.no_grad():
        expected = inputs[0]
        for block in model.get_vlm_model().text_model.layers[: index + 1]:
            residual = expected + block.self_attn.o_proj(expected)
            expected = residual + block.mlp(block.post_attention_layernorm(residual))
    torch.testing.assert_close(selected, expected)
    if index == 2:
        torch.testing.assert_close(outputs[0], model.get_vlm_model().text_model.norm(selected))
    selected.square().mean().backward()
    for i, block in enumerate(model.get_vlm_model().text_model.layers):
        grad = block.mlp.weight.grad
        if i <= index:
            assert grad is not None and grad.abs().sum() > 0
        else:
            assert grad is None
    assert all(parameter.grad is None for parameter in model.lm_expert.parameters())


@pytest.mark.parametrize("index", [-1, 3])
def test_selected_layer_rejects_invalid_actual_depth(index):
    model = LayerSelectionVLM()
    with pytest.raises(ValueError, match="alignment_layer_index"):
        model(inputs_embeds=[torch.randn(1, 2, 8), None], alignment_layer_index=index)


def test_selected_layer_rejects_missing_prefix():
    model = LayerSelectionVLM()
    with pytest.raises(ValueError, match="prefix inputs"):
        model(inputs_embeds=[None, torch.randn(1, 2, 8)], alignment_layer_index=1)


def test_alignment_routes_selected_layer_and_default_does_not_change_action_loss():
    policy = make_policy(special_tokens=True)
    batch = make_batch()
    model = policy.model
    images, masks = policy.prepare_images(batch)
    args = (
        images,
        masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        policy.prepare_state(batch),
        policy.prepare_action(batch),
    )
    noise = torch.randn_like(args[-1])
    time = torch.full((2,), 0.5)
    baseline_loss, _ = model(*args, noise=noise, time=time, return_image_features=True)
    policy.config.alignment_layer_index = 0
    selected_loss, features = model(*args, noise=noise, time=time, return_image_features=True)
    torch.testing.assert_close(selected_loss, baseline_loss)
    for feature, image in zip(features, images, strict=True):
        raw = model.vlm_with_expert.embed_image(image) * (8**0.5)
        torch.testing.assert_close(feature, raw)
    # Inference-style calls do not request intermediate features even with the option configured.
    plain_loss = model(*args, noise=noise, time=time)
    torch.testing.assert_close(plain_loss, baseline_loss)


@pytest.mark.parametrize("index", [None, 14])
def test_alignment_layer_config_roundtrip(tmp_path, index):
    config = SmolVLAConfig(alignment_layer_index=index)
    config.save_pretrained(tmp_path)
    restored = SmolVLAConfig.from_pretrained(tmp_path)
    assert restored.alignment_layer_index == index


def test_alignment_layer_config_rejects_negative_index():
    with pytest.raises(ValueError, match="alignment_layer_index"):
        SmolVLAConfig(alignment_layer_index=-1)
