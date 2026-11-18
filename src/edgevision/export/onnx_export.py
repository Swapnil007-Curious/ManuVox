"""ONNX export of a full model, FeatureBlock included."""

import itertools
import warnings
from pathlib import Path

import onnx
import torch

from edgevision.layout import EMBEDDING_NAME, INPUT_NAME, OUTPUT_NAME, V1_LAYOUT, InputLayout
from edgevision.models.features import SignClassifier, StandardisedFeatures

DEFAULT_OPSET = 17
# A batch of 1 could let the tracer fold the batch size into a constant; 2 cannot.
_TRACE_BATCH = 2

# Known warnings of the TorchScript exporter, each matched narrowly and judged harmless here.
_EXPECTED_WARNINGS: tuple[tuple[type[Warning], str, str], ...] = (
    # The legacy exporter is chosen on purpose; see export_onnx.
    (DeprecationWarning, r"You are using the legacy TorchScript-based ONNX export", ""),
    # torch's own exporter calls one of its deprecated helpers.
    (DeprecationWarning, r"The feature will be removed", r"torch\.onnx\."),
    # A torch.jit.TracerWarning (not re-exported in torch's stubs, hence Warning here): nn.GRU
    # validates input and hidden sizes on traced shapes, which are fixed for these models.
    (Warning, r"Converting a tensor to a Python boolean", r"torch\.nn\.modules\.rnn"),
    # Emitted for every RNN export; the parity tests run batch sizes 1 and 7 to rule it out.
    (UserWarning, r"Exporting a model to ONNX with a batch_size other than 1", r"torch\.onnx\."),
)


class _WithEmbedding(torch.nn.Module):
    """Traces a model's forward_with_embedding, so the export has both outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward = getattr(self.model, "forward_with_embedding")  # noqa: B009 - a dynamic attribute
        result: tuple[torch.Tensor, torch.Tensor] = forward(x)
        return result


def supports_embedding(model: torch.nn.Module) -> bool:
    """Whether the model can export its penultimate embedding (the hybrid can)."""
    return callable(getattr(model, "forward_with_embedding", None))


def export_onnx(
    model: torch.nn.Module,
    path: Path,
    opset: int = DEFAULT_OPSET,
    layout: InputLayout | None = None,
    *,
    embedding: bool = False,
) -> Path:
    """Export with input "landmarks" (batch, 32, frame_dim) and output "logits", batch axis
    dynamic. ``layout`` defaults to the model's feature-block layout (v1: frame_dim 100). With
    ``embedding``, a second output "embedding" (batch, dim) holds the penultimate embedding the
    head reads; the logits are computed exactly as without it.

    Uses the TorchScript exporter (dynamo=False): the dynamo exporter needs onnxscript, which
    this project does not depend on. The model is put in eval mode and left there.
    """
    if layout is None:
        if isinstance(model, SignClassifier):
            layout = model.features.layout
        elif isinstance(model, StandardisedFeatures):
            layout = model.layout
        else:
            layout = V1_LAYOUT
    model.eval()
    tensors = itertools.chain(model.parameters(), model.buffers())
    device = next(tensors, torch.zeros(0)).device  # A feature block alone has only buffers.
    example = torch.zeros(_TRACE_BATCH, layout.num_frames, layout.frame_dim, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(), torch.no_grad():
        for category, message, module in _EXPECTED_WARNINGS:
            warnings.filterwarnings("ignore", message=message, category=category, module=module)
        if embedding and not supports_embedding(model):
            raise ValueError(f"{type(model).__name__} has no forward_with_embedding")
        outputs = [OUTPUT_NAME, EMBEDDING_NAME] if embedding else [OUTPUT_NAME]
        torch.onnx.export(
            _WithEmbedding(model) if embedding else model,
            (example,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=outputs,
            dynamic_axes={INPUT_NAME: {0: "batch"}, **{name: {0: "batch"} for name in outputs}},
            opset_version=opset,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))
    return path
