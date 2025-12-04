import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio

LABEL_KEY = 'Outcome'
FEATURE_KEYS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=32):
    """Fungsi untuk loading data ke dalam model."""
    
    # 1. Ambil Schema
    schema = tf_transform_output.transformed_metadata.schema
    
    # 2. Options (Batching)
    tfxio_options = tfxio.TensorFlowDatasetOptions(batch_size=batch_size)
    
    # 3. Load Dataset
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio_options,
        schema
    )
    
    # 4. Fungsi Pemisah Label + KONVERSI KE DENSE (Perbaikan Utama)
    def split_label(features):
        # Ambil label
        label = features.pop(LABEL_KEY)
        
        # FIX: Ubah semua feature dari Sparse ke Dense agar Keras tidak error
        for key in features.keys():
            if isinstance(features[key], tf.sparse.SparseTensor):
                features[key] = tf.sparse.to_dense(features[key])
        
        # FIX: Ubah Label dari Sparse ke Dense
        if isinstance(label, tf.sparse.SparseTensor):
            label = tf.sparse.to_dense(label)
        
        # Pastikan label bentuknya [batch_size, 1] agar cocok dengan binary_crossentropy
        label = tf.reshape(label, [-1, 1])
            
        return features, label

    # 5. Apply mapping
    return dataset.map(split_label).repeat()

def _build_keras_model():
    """Membuat arsitektur model."""
    # Input layer
    inputs = [tf.keras.Input(shape=(1,), name=key) for key in FEATURE_KEYS + ['Age_bucket']]
    
    # Gabung semua input
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
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output)

    model = _build_keras_model()

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )

    model.save(fn_args.serving_model_dir, save_format='tf')