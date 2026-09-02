import numpy as np

_MODEL_NAME = "BAAI/bge-small-en-v1.5"

class Embedder:
    def __init__(self, model_name=_MODEL_NAME):
        self._model_name = model_name
        self._model = None 

    def _ensure(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)

    def embed(self, texts):
        self._ensure()
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_one(self, text):
        return self.embed([text])[0]


def cosine(a, b):
    if b.size == 0:
        return np.array([])
    return b @ a


_shared = None

def shared():
    global _shared
    if _shared is None:
        _shared = Embedder()
    return _shared
