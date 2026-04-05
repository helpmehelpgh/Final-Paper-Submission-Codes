from __future__ import annotations

import argparse
import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for ACC state classifier")
    parser.add_argument("--onnx_model", type=str, required=True)
    parser.add_argument(
        "--features",
        type=float,
        nargs=11,
        required=True,
        help="11 scaled features: v_t v_t-1 ... v_t-10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x = np.array(args.features, dtype=np.float32).reshape(1, 11)

    session = ort.InferenceSession(args.onnx_model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: x})[0]
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = int((prob >= 0.5).astype(np.int32).ravel()[0])

    print("Predicted label:", pred)
    print("Predicted probability:", float(prob.ravel()[0]))
    print("Meaning:", "ACC enabled" if pred == 1 else "ACC not enabled")


if __name__ == "__main__":
    main()