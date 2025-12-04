import os
import tfx.v1 as tfx
from tfx.proto import trainer_pb2, pusher_pb2

def create_components(data_root, transform_module, trainer_module, serving_model_dir):
    
    # 1. ExampleGen: Baca data CSV
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # 2. StatisticsGen: Analisis statistik data
    stats_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

    # 3. SchemaGen: Buat schema data otomatis
    schema_gen = tfx.components.SchemaGen(
        statistics=stats_gen.outputs['statistics'],
        infer_feature_shape=False
    )

    # 4. ExampleValidator: Cek anomali data
    example_validator = tfx.components.ExampleValidator(
        statistics=stats_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # 5. Transform: Preprocessing data
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )

    # 6. Trainer: Latih model
    trainer = tfx.components.Trainer(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )

    # 7. Pusher: Push model jika bagus (kita skip Evaluator biar simpel dulu)
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    return [
        example_gen, stats_gen, schema_gen, example_validator,
        transform, trainer, pusher
    ]