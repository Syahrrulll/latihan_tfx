import tensorflow_transform as tft

LABEL_KEY = 'Outcome'
FEATURE_KEYS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
AGE_BUCKETS = 3

def preprocessing_fn(inputs):
    outputs = {}

    # Standarisasi numerik (z-score)
    for key in FEATURE_KEYS:
        # tf_transform mengharapkan input tensor, pastikan tipe datanya float
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # Bucketize Age (mengubah umur jadi kategori)
    outputs['Age_bucket'] = tft.bucketize(inputs['Age'], num_buckets=AGE_BUCKETS)

    # Label tidak diubah, cuma dicopy
    outputs[LABEL_KEY] = inputs[LABEL_KEY]

    return outputs