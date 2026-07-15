# Phase 2 model specification

## Scope

Phase 2 implements and structurally verifies the binary classifier. It does not
train, select a checkpoint or threshold, read test labels for optimization, or
produce performance metrics.

## Locked architecture

The implementation in `src/cbis_ddsm/model.py` follows this tensor path:

1. A three-channel `256 x 256` mammogram enters EfficientNet-B0.
2. The classifier and global pooling from EfficientNet are removed; its final
   convolutional feature map has 1,280 channels and spatial size `8 x 8`.
3. A learnable `1 x 1` convolution projects 1,280 channels to `D=512`.
4. The projected map is flattened in row-major order into 64 spatial tokens.
5. A parameter-free two-dimensional sinusoidal position encoding is added.
6. Four Transformer encoder layers process the tokens with `d_model=512` and
   eight attention heads.
7. Mean pooling across the 64 encoded tokens gives one vector per image.
8. Layer normalization and a linear layer return exactly one logit per image.

All backbone parameters are trainable. The model accepts no 224-pixel fallback
and rejects inputs that do not have shape `[batch, 3, 256, 256]`.

## Reproduction assumptions

The available reference specifies B0, 512-dimensional spatial tokens, positional
encoding, four Transformer layers, eight heads, aggregation, and a single logit.
It does not preserve every implementation detail. The following choices are
therefore explicit **Reproduction assumptions**, not recovered historical facts:

- ImageNet-pretrained EfficientNet-B0 initialization.
- Parameter-free 2D sinusoidal positional encoding.
- Transformer feed-forward width 2,048, GELU, dropout 0.1, post-attention layer
  normalization as implemented by `torch.nn.TransformerEncoderLayer`.
- Mean token pooling followed by a final LayerNorm.

These parameters are centralized in `configs/model.yaml`.

## Verification boundary

Unit and smoke tests construct the model with `pretrained=false`, so phase 2
does not download weights. Tests verify feature-map, token and logit shapes;
layer/head counts; invalid architecture rejection; deterministic positional
encoding; and a finite backward pass through `BCEWithLogitsLoss` with
`pos_weight=1.84`.

The fixed loss weight is checked only for interface compatibility here. The
actual criterion, optimizer, scheduler, checkpoint policy and training loop
belong to phase 3.

## Verified result

Validation on 2026-07-15 completed without downloading pretrained weights:

- The combined phase 1 and phase 2 suite passed all 26 tests.
- `pip check` reported no broken requirements.
- CPU and CUDA forward/backward smoke tests produced finite loss values.
- On an NVIDIA GeForce RTX 5070 Ti Laptop GPU, batch size 1 produced feature
  shapes `[1, 1280, 8, 8]`, `[1, 512, 8, 8]`, `[1, 64, 512]`, and logit shape
  `[1]`.
- The model has 17,274,493 parameters, all trainable under the locked config.
- Peak allocated CUDA memory for the synthetic batch-size-1 smoke test was
  253.69 MiB. This is a structural check, not a phase 3 training-memory
  benchmark.
