"""
convert_yolo_to_qnn.py  —  best1_clean.onnx -> QNN binary via Qualcomm AI Hub
=============================================================================
Run on your LAPTOP (not Radxa). Needs: pip install qai-hub
and a configured token (qai-hub configure --api_token ...).

Flow:  cleaned ONNX -> Quantize (INT8 + I/O) -> Compile to QNN context binary
Output: models/best1_qnn.bin  (copy to Radxa models/ folder)

Fixes applied:
  - points at best1_clean.onnx (value_info collision removed)
  - real device name (QCS6490 proxy was removed from AI Hub)
  - --quantize_io so the NPU accepts INT8 input/output (not FP32)
  - status checks so a failed job doesn't crash the next step
"""

import numpy as np
import qai_hub as hub

ONNX_PATH   = "best1_clean.onnx"     # cleaned model (not best1.onnx)
INPUT_NAME  = "images"               # confirm via the onnx inputs print
INPUT_SHAPE = (1, 3, 640, 640)

# Real QCS6490 device (proxy removed). If this errors, run:
#   python -c "import qai_hub as hub; print([d.name for d in hub.get_devices()])"
# and paste the exact QCS6490 / Dragonwing entry here.
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"

OUT_PATH = "models/best1_qnn.bin"


def main():
    device = hub.Device(DEVICE_NAME)

    # ── 1. Calibration data (REAL frames give best INT8 accuracy) ───────────────
    # Replace these random tensors with real preprocessed drone frames
    # (resize 640, RGB, /255, CHW) for production accuracy.
    calib = [np.random.rand(*INPUT_SHAPE).astype(np.float32) for _ in range(50)]
    calib_data = {INPUT_NAME: calib}

    # ── 2. Quantize (weights + activations -> INT8) ─────────────────────────────
    print("[QNN] Submitting quantize job...")
    quant_job = hub.submit_quantize_job(
        model=ONNX_PATH,
        calibration_data=calib_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    quant_job.wait()
    if not quant_job.get_status().success:
        print(f"[QNN] Quantize FAILED: {quant_job.url}")
        return
    quant_model = quant_job.get_target_model()
    print("[QNN] Quantize done.")

    # ── 3. Compile to QNN context binary, with I/O quantized ────────────────────
    print("[QNN] Submitting compile job...")
    compile_job = hub.submit_compile_job(
        model=quant_model,
        device=device,
        options="--target_runtime qnn_context_binary --quantize_io",
    )
    compile_job.wait()
    if not compile_job.get_status().success:
        print(f"[QNN] Compile FAILED: {compile_job.url}")
        return
    target = compile_job.get_target_model()
    if target is None:
        print(f"[QNN] Compile produced no model: {compile_job.url}")
        return

    import os
    os.makedirs("models", exist_ok=True)
    target.download(OUT_PATH)
    print(f"[QNN] Saved {OUT_PATH}")

    # ── 4. Optional: profile real NPU latency on device ─────────────────────────
    print("[QNN] Submitting profile job...")
    prof = hub.submit_profile_job(model=target, device=device)
    print(f"[QNN] Profile job: {prof.url}")


if __name__ == "__main__":
    main()