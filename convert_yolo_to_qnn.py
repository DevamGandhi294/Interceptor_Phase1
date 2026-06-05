"""
convert_yolo_to_qnn.py  —  best1.onnx -> QNN binary via Qualcomm AI Hub
=======================================================================
Run on your LAPTOP (not Radxa). Needs: pip install qai-hub
and a configured API token (qai-hub configure --api_token ...).

Flow:  ONNX -> Quantize (INT8) -> Compile to QNN context binary (NPU)
Output: a precompiled QNN ONNX you drop into models/ on the Radxa.

NOTE: QCS6490 proxy was removed from AI Hub. Target the real device.
      Confirm the exact device name with: python -c "import qai_hub as hub; \
      print([d.name for d in hub.get_devices()])"  and pick the QCS6490 /
      Dragonwing RB3 Gen2 entry.
"""

import numpy as np
import qai_hub as hub

ONNX_PATH   = "best1.onnx"
INPUT_NAME  = "images"          # confirm via Netron if unsure
INPUT_SHAPE = (1, 3, 640, 640)  # your YOLO input
# Pick the device string that matches QCS6490 in your hub.get_devices() list:
DEVICE_NAME = "QCS6490 (Proxy)"  # <-- REPLACE with real device from the print above

def main():
    device = hub.Device(DEVICE_NAME)

    # ── 1. Calibration data for INT8 quantization ───────────────────────────────
    # Ideally use REAL frames from your camera/dataset for best accuracy.
    # Here: random tensors as placeholder. Replace with real drone frames.
    calib = [np.random.rand(*INPUT_SHAPE).astype(np.float32) for _ in range(50)]
    calib_data = dict(zip([INPUT_NAME], [calib]))

    # ── 2. Quantize job (FP32 ONNX -> INT8 QDQ ONNX) ────────────────────────────
    print("[QNN] Submitting quantize job...")
    quant_job = hub.submit_quantize_job(
        model=ONNX_PATH,
        calibration_data=calib_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    quant_model = quant_job.get_target_model()
    print("[QNN] Quantize done.")

    # ── 3. Compile job (INT8 ONNX -> QNN context binary for NPU) ────────────────
    print("[QNN] Submitting compile job...")
    compile_job = hub.submit_compile_job(
        model=quant_model,
        device=device,
        options="--target_runtime qnn_context_binary",
    )
    target = compile_job.get_target_model()
    target.download("models/best1_qnn.bin")
    print("[QNN] Saved models/best1_qnn.bin")

    # ── 4. (optional) Profile on the device to see real latency ─────────────────
    print("[QNN] Submitting profile job...")
    prof = hub.submit_profile_job(model=target, device=device)
    print(f"[QNN] Profile job: {prof.url}")

if __name__ == "__main__":
    main()