from pathlib import Path

from src.config import NB_MODEL_PATH, SVM_MODEL_PATH, TFIDF_PATH


def test_model_artifact_paths_are_defined():
    for path in (NB_MODEL_PATH, SVM_MODEL_PATH, TFIDF_PATH):
        assert isinstance(path, Path)
        assert path.suffix in {".pkl"}
