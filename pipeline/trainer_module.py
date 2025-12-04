import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'Outcome'
FEATURE_KEYS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=32):
    """Fungsi untuk loading data ke dalam model."""
    
    # 1. Load Dataset (Format masih dictionary gabungan features + label)
    # UPDATED: Menambahkan argumen 'schema' yang sebelumnya hilang
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tft.TFTransformOutput(tf_transform_output.transform_output_path).transformed_metadata.schema if isinstance(tf_transform_output, tft.TFTransformOutput) else tf_transform_output.transformed_metadata.schema,
        'transformed_examples' # PENTING: Tentukan format dataset sebagai transformed examples
    )
    
    # 2. Fungsi Manual untuk memisahkan Label dari Features
    def split_label(features):
        label = features.pop(LABEL_KEY)
        return features, label

    # 3. Apply mapping (pisah label), repeat, lalu batching manual
    return dataset.map(split_label).repeat().batch(batch_size)

def _build_keras_model():
    """Membuat arsitektur model."""
    # Input layer harus cocok dengan nama feature di dataset
    # Kita gabungkan fitur asli + fitur hasil transform (Age_bucket)
    inputs = [tf.keras.Input(shape=(1,), name=key) for key in FEATURE_KEYS + ['Age_bucket']]
    
    # Concatenate features
    x = tf.keras.layers.Concatenate()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

def run_fn(fn_args: FnArgs):
    """Fungsi utama yang dipanggil oleh TFX Trainer."""
    
    # Load output transformasi
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Panggil _input_fn dengan argumen yang benar
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output)

    model = _build_keras_model()

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )

    # Save model
    model.save(fn_args.serving_model_dir, save_format='tf')