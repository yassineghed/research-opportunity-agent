from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str):
        return self.model.encode(text)

    def encode_batch(self, texts: list[str]):
        return self.model.encode(texts)