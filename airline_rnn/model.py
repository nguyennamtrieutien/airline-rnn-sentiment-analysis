from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf

# Keras API name is retained only here; the project uses the public name RNN.
RNN = tf.keras.layers.SimpleRNN


def configure_runtime(seed=42):
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("AIRLINE_INTRA_THREADS", "2")))
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # TensorFlow runtime was already initialized in this notebook.
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_model(config, length):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(length,), dtype="int32", name="token_ids"),
        tf.keras.layers.Embedding(config.vocabulary_size, config.embedding_dimension,
                                  mask_zero=config.mask_zero, name="word_embedding"),
        tf.keras.layers.SpatialDropout1D(config.spatial_dropout, name="embedding_dropout"),
        RNN(config.hidden_units, activation="tanh",
                                  dropout=config.input_dropout, recurrent_dropout=config.recurrent_dropout,
                                  return_sequences=False, name="rnn"),
        tf.keras.layers.Dropout(config.output_dropout, name="state_dropout"),
        tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_features"),
        tf.keras.layers.Dropout(config.dense_dropout, name="dense_dropout"),
        tf.keras.layers.Dense(len(config.labels), activation="softmax", name="sentiment_probabilities"),
    ], name=f"Airline_{config.id.replace('-', '_')}_RNN")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate,
                                         global_clipnorm=config.gradient_clipnorm)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")], jit_compile=False)
    return model


def training_dataset(features, labels, batch_size, seed):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(len(features), seed=seed, reshuffle_each_iteration=True).batch(batch_size).with_options(options).prefetch(1)
