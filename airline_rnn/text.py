from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

SOURCE_STRIP_CHARACTERS = "@VirginAmerica@UnitedAir@Southwestairline@DeltaAir@USAirways@American"
KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "shan't": "shall not",
    "ain't": "is not", "i'm": "i am", "it's": "it is", "that's": "that is",
    "what's": "what is", "there's": "there is", "here's": "here is", "let's": "let us",
}
RESERVED = {"__pad__": 0, "__oov__": 1, "usermarker": 2, "urlmarker": 3, "emptymarker": 4}


def source_clean(text):
    text = str(text).lstrip(SOURCE_STRIP_CHARACTERS).rstrip("@")
    return re.sub(r"[^a-zA-z0-9\s]", "", text.lower())


def clean_text(text):
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = text.replace("’", "'").replace("‘", "'").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urlmarker ", text)
    text = re.sub(r"@[a-z0-9_]+", " usermarker ", text)
    for key, value in CONTRACTIONS.items():
        text = re.sub(r"(?<!\w)" + re.escape(key) + r"(?!\w)", value, text)
    text = re.sub(r"\b(\w+)n't\b", r"\1 not", text)
    for suffix, expansion in (("'re", " are"), ("'ve", " have"), ("'ll", " will"),
                              ("'d", " would"), ("'m", " am")):
        text = re.sub(re.escape(suffix) + r"\b", expansion, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip() or "emptymarker"


def canonical_text(text):
    text = unicodedata.normalize("NFKC", html.unescape(str(text))).lower()
    return re.sub(r"\s+", " ", text).strip()


class WordTokenizer:
    """Frequency-ranked word IDs; legacy mode matches Keras Tokenizer defaults.

    Counter preserves first occurrence order for tied counts, as Keras does.
    C0 has explicit reserved IDs; all vocabulary caps include these IDs.
    """

    def __init__(self, vocabulary_size=4000, legacy=False):
        self.vocabulary_size = vocabulary_size
        self.legacy = legacy
        self.word_index = {}
        self.word_counts = {}

    def words(self, text):
        text = str(text).lower()
        if self.legacy:
            text = text.translate(str.maketrans({char: " " for char in KERAS_FILTERS}))
        return [word for word in text.split(" ") if word]

    def fit(self, texts):
        counts = Counter(word for text in texts for word in self.words(text))
        self.word_counts = dict(counts)
        if self.legacy:
            self.word_index = {word: i + 1 for i, (word, _) in enumerate(counts.most_common())}
        else:
            self.word_index = dict(RESERVED)
            for word, _ in counts.most_common():
                if word not in self.word_index and len(self.word_index) < self.vocabulary_size:
                    self.word_index[word] = len(self.word_index)
        return self

    def encode(self, text):
        words = self.words(text)
        sequence, unknown = [], 0
        for word in words:
            index = self.word_index.get(word)
            if index is None or index >= self.vocabulary_size:
                unknown += 1
                if not self.legacy:
                    sequence.append(1)
            else:
                sequence.append(index)
        return sequence, {"token_length": len(words), "sequence_length_before_padding": len(sequence),
                          "unknown_tokens": unknown, "oov_rate": unknown / len(words) if words else 0.0}

    def to_dict(self):
        return {"format": "airline-word-tokenizer-v1", "legacy": self.legacy,
                "vocabulary_size": self.vocabulary_size, "word_index": self.word_index,
                "word_counts": self.word_counts}

    @classmethod
    def from_dict(cls, value):
        if value["format"] != "airline-word-tokenizer-v1":
            raise ValueError("Unsupported tokenizer format")
        tokenizer = cls(value["vocabulary_size"], value["legacy"])
        tokenizer.word_index = value["word_index"]
        tokenizer.word_counts = value.get("word_counts", {})
        return tokenizer


def pad_sequences(sequences, length, legacy=False):
    import numpy as np
    result = np.zeros((len(sequences), length), dtype="int32")
    for i, sequence in enumerate(sequences):
        truncated = sequence[-length:] if legacy else sequence[:length]
        if truncated:
            if legacy:
                result[i, -len(truncated):] = truncated
            else:
                result[i, :len(truncated)] = truncated
    return result
