import os
import tfx.v1 as tfx
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from components import create_components

# --- Definisi Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
PIPELINE_ROOT = os.path.join(PROJECT_ROOT, 'output', 'pipeline_root')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'output', 'metadata.db')
SERVING_MODEL_DIR = os.path.join(PROJECT_ROOT, 'output', 'serving_model')

# Lokasi File Module
TRANSFORM_MODULE = os.path.join(PROJECT_ROOT, 'pipeline', 'preprocess.py')
TRAINER_MODULE = os.path.join(PROJECT_ROOT, 'pipeline', 'trainer_module.py')

def init_pipeline():
    # Mengirim SEMUA argumen yang dibutuhkan components.py
    components = create_components(
        data_root=DATA_ROOT,
        transform_module=TRANSFORM_MODULE,
        trainer_module=TRAINER_MODULE,
        serving_model_dir=SERVING_MODEL_DIR
    )
    
    # Membuat Pipeline
    return tfx.dsl.Pipeline(
        pipeline_name='tfx_diabetes_pipeline',
        pipeline_root=PIPELINE_ROOT,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
        components=components,
        enable_cache=True
    )

if __name__ == '__main__':
    # Pastikan folder data ada
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: Folder data tidak ditemukan di {DATA_ROOT}")
    else:
        print("=== Memulai Pipeline ===")
        pipeline = init_pipeline()
        LocalDagRunner().run(pipeline)
        print("=== Pipeline Selesai ===")