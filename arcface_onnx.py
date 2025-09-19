import os
import cv2
import numpy as np
import onnxruntime as ort

class ArcFaceONNX:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        # Prefer CUDA, then DirectML (Windows), then CPU.
        available = ort.get_available_providers()
        preferred = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in preferred if p in available]
        if not providers:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        # Store actual providers in use (after session init) for diagnostics.
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112))
        arr = resized.astype(np.float32)
        arr = (arr - 127.5) / 128.0
        chw = np.transpose(arr, (2, 0, 1))
        nchw = np.expand_dims(chw, axis=0)
        return nchw

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        inp = self._preprocess(bgr_img)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        vec = out[0].astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-10
        return vec / norm
