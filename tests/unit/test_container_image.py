"""Regression checks for runtime image requirements."""

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_read_only_image_bundles_pipecat_sentence_tokenizer() -> None:
    """NLTK must never attempt a download in the read-only runtime container."""
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "NLTK_DATA=/app/nltk_data" in dockerfile
    assert "nltk.download('punkt_tab'" in dockerfile
    assert "download_dir='/app/nltk_data'" in dockerfile
