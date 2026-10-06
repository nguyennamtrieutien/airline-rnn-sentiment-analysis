from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from airline_rnn.common import is_registered_training_source, read_json, training_source_digest
from airline_rnn.config import Configuration
from airline_rnn.inference import SentimentPredictor
from airline_rnn.model import build_model, configure_runtime
from airline_rnn.text import clean_text, pad_sequences
from airline_rnn.training import make_predictor

ROOT = Path(__file__).resolve().parents[1]


def test_archived_training_source_and_current_code_are_both_verified():
    before = read_json(ROOT / 'sources/naming_migration.json')['before_training_source_sha256']
    assert is_registered_training_source(ROOT, before)
    assert is_registered_training_source(ROOT, training_source_digest())
    assert not is_registered_training_source(ROOT, '0' * 64)


@pytest.mark.parametrize('change', ['current_code_digest', 'archive_bytes'])
def test_naming_migration_rejects_unverified_code_or_snapshot(tmp_path, change):
    source = tmp_path / 'sources'
    source.mkdir()
    migration = read_json(ROOT / 'sources/naming_migration.json')
    shutil.copy2(ROOT / 'sources' / migration['training_snapshot'], source / migration['training_snapshot'])
    if change == 'current_code_digest':
        migration['after_training_source_sha256'] = '0' * 64
    else:
        with (source / migration['training_snapshot']).open('ab') as stream:
            stream.write(b'changed')
    (source / 'naming_migration.json').write_text(json.dumps(migration))
    assert not is_registered_training_source(tmp_path, migration['before_training_source_sha256'])


def test_renamed_rnn_graph_preserves_checkpoint_predictions():
    predictor = SentimentPredictor(ROOT)
    configure_runtime(42)
    renamed = build_model(Configuration(**predictor.config), predictor.representation['sequence_length'])
    renamed.set_weights(predictor.model.get_weights())
    texts = ['Thank you for the helpful service!', 'My flight was delayed again and nobody helped me.',
             "@united I can't recommend this service", 'flight ' * 48]
    encoded = [predictor.tokenizer.encode(clean_text(text))[0] for text in texts]
    features = pad_sequences(encoded, predictor.representation['sequence_length'], False)
    np.testing.assert_allclose(make_predictor(renamed)(features), predictor.forward(features), atol=1e-7, rtol=1e-6)
    assert renamed.name.endswith('_RNN')
    assert renamed.get_layer('rnn').units == predictor.config['hidden_units']
    assert renamed.count_params() == predictor.model.count_params()
