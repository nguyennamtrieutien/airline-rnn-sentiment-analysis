from __future__ import annotations

from dataclasses import asdict, dataclass, replace

SEEDS = (42, 2026, 3407)
THREE_CLASS_LABELS = {"negative": 0, "neutral": 1, "positive": 2}
BINARY_LABELS = {"negative": 0, "positive": 1}


@dataclass(frozen=True)
class Configuration:
    id: str
    benchmark: str = "three_class"
    vocabulary_size: int = 4000
    sequence_length: int | None = 40
    hidden_units: int = 196
    embedding_dimension: int = 128
    spatial_dropout: float = 0.5
    input_dropout: float = 0.3
    recurrent_dropout: float = 0.3
    output_dropout: float = 0.2
    dense_units: int = 100
    dense_dropout: float = 0.4
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 5
    min_delta: float = 0.001
    class_weighting: str = "none"
    gradient_clipnorm: float | None = 1.0
    early_stopping: bool = True
    mask_zero: bool = True

    @property
    def labels(self):
        return THREE_CLASS_LABELS if self.benchmark == "three_class" else BINARY_LABELS

    @property
    def label_names(self):
        return list(self.labels)

    def to_dict(self):
        return asdict(self)


def configurations(tier="recommended"):
    baseline = Configuration("C0")
    binary = replace(baseline, id="A2", benchmark="binary_clean", sequence_length=None,
                     gradient_clipnorm=None, mask_zero=False)
    source = replace(binary, id="A1", benchmark="binary_source", early_stopping=False)
    if tier == "minimum":
        return [(source, (42,)), (binary, (42,))]
    matrix = [source, binary, baseline,
              replace(baseline, id="B1-20", sequence_length=20),
              replace(baseline, id="B1-60", sequence_length=60),
              replace(baseline, id="B3-64", hidden_units=64),
              replace(baseline, id="B3-128", hidden_units=128),
              replace(baseline, id="B5-balanced", class_weighting="balanced")]
    if tier == "advanced":
        matrix += [replace(baseline, id="B2-2k", vocabulary_size=2000),
                   replace(baseline, id="B2-8k", vocabulary_size=8000),
                   replace(baseline, id="B4-0", spatial_dropout=0.0),
                   replace(baseline, id="B4-025", spatial_dropout=0.25)]
    elif tier != "recommended":
        raise ValueError(f"Unknown tier: {tier}")
    return [(config, (42,) if config.id == "A1" else SEEDS) for config in matrix]


def configuration_by_id(config_id):
    for config, _ in configurations("advanced"):
        if config.id == config_id:
            return config
    raise ValueError(config_id)
