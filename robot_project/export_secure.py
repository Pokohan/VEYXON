import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from robot_project.config import MODEL_PATH

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load(MODEL_PATH)
onnx_model = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, 126]))])
out_path = os.path.splitext(MODEL_PATH)[0] + ".onnx"
with open(out_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
