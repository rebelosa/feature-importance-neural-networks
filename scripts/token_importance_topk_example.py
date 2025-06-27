"""Display top-k token importances for several IMDB samples."""

from __future__ import annotations

import logging

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neural_feature_importance.embedding_callbacks import (
    EmbeddingVarianceImportanceKeras,
)

from token_topk_utils import analyze_samples, build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FEATURES = 5000
MAX_LEN = 400
TOP_K = 5
NUM_SAMPLES = 5


def main() -> None:
    """Train a model and log top tokens for a few samples."""
    (x_train, y_train), _ = imdb.load_data(num_words=MAX_FEATURES)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)

    model = build_model()
    callback = EmbeddingVarianceImportanceKeras()
    model.fit(x_train, y_train, epochs=2, batch_size=128, callbacks=[callback], verbose=0)

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed.")
        return

    word_index = imdb.get_word_index()
    index_word = {v + 3: k for k, v in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    index_word[3] = "<UNUSED>"

    analyze_samples(x_train, y_train, scores, index_word, NUM_SAMPLES, TOP_K)


if __name__ == "__main__":
    main()
