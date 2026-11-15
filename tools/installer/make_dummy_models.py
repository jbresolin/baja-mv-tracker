"""Tiny stand-in ONNX models with the same file names, inputs and output shapes as the real ones in
deploy/models/manifest.json, for the CI smoke test (the real models are not in git). They detect nothing and read
nothing; they only let every program load and run its models.

usage: python tools/installer/make_dummy_models.py <folder>      (needs `pip install onnx`; not part of the app)

  car_n.onnx, car_s.onnx  input images (1, 3, 384, 640)   -> output0 (1, 5, 8) all zeros (no box passes conf)
  panel.onnx              input images (1, 3, H, W)       -> output0 (1, 5, 8) all zeros
  recog.onnx              input image  (N, 3, 32, W)      -> logp (10, N, 11) log-probabilities
  yolo11n_coco.onnx       input images (1, 3, 384, 640)   -> output0 (1, 84, 8) (the zone's bench detector)
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17          # what the real exports use; ReduceMean still takes `axes` as an attribute here


def save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="bajamv-dummy")
    model.ir_version = 8                     # readable by every onnxruntime / OpenVINO the app may ship
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def detector(path, input_shape, rows):
    """images -> a (1, rows, 8) tensor of zeros that still depends on the input (so nothing is optimised away)."""
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, rows, 8])
    zeros = numpy_helper.from_array(np.zeros((1, rows, 8), np.float32), "zeros")
    nothing = numpy_helper.from_array(np.array(0.0, np.float32), "nothing")
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),     # one number
        helper.make_node("Mul", ["mean", "nothing"], ["zero"]),                # times 0
        helper.make_node("Add", ["zero", "zeros"], ["output0"]),               # broadcast onto the zeros
    ]
    save(helper.make_graph(nodes, "dummy_detector", [images], [output], [zeros, nothing]), path)


def recognizer(path):
    """image (N, 3, 32, W) -> logp (10, N, 11): mean colour -> 11 scores -> repeated over 10 time steps."""
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 32, "W"])
    logp = helper.make_tensor_value_info("logp", TensorProto.FLOAT, [10, "N", 11])
    weights = numpy_helper.from_array(np.linspace(-1, 1, 33, dtype=np.float32).reshape(3, 11), "weights")
    axis0 = numpy_helper.from_array(np.array([0], np.int64), "axis0")
    repeats = numpy_helper.from_array(np.array([10, 1, 1], np.int64), "repeats")
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["colour"], axes=[2, 3], keepdims=0),    # (N, 3)
        helper.make_node("MatMul", ["colour", "weights"], ["scores"]),                      # (N, 11)
        helper.make_node("Unsqueeze", ["scores", "axis0"], ["scores1"]),                    # (1, N, 11)
        helper.make_node("Tile", ["scores1", "repeats"], ["scores10"]),                     # (10, N, 11)
        helper.make_node("LogSoftmax", ["scores10"], ["logp"], axis=2),
    ]
    save(helper.make_graph(nodes, "dummy_recognizer", [image], [logp], [weights, axis0, repeats]), path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    folder.mkdir(parents=True, exist_ok=True)
    detector(folder / "car_n.onnx", [1, 3, 384, 640], 5)
    detector(folder / "car_s.onnx", [1, 3, 384, 640], 5)
    detector(folder / "panel.onnx", [1, 3, "H", "W"], 5)
    detector(folder / "yolo11n_coco.onnx", [1, 3, 384, 640], 84)
    recognizer(folder / "recog.onnx")
    print("dummy models written to %s" % folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
