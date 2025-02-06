from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.estimator import Estimator
from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
import os
from sagemaker.workflow.steps import CacheConfig
from sagemaker.processing import ScriptProcessor
from sagemaker import image_uris


def run_pipeline():
    load_dotenv()

    cache_config = CacheConfig(enable_caching=True, expire_after="10d")

    sagemaker_session = PipelineSession(
        default_bucket=os.environ["S3_BUCKET_NAME"],
    )

    # define the processor
    image_uri = "482497089777.dkr.ecr.eu-north-1.amazonaws.com/mxnetrecio:latest"
    processor = ScriptProcessor(
        image_uri=image_uri,
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["PROCESSING_JOB_INSTANCE_COUNT"]),
        instance_type=os.environ["PROCESSING_JOB_INSTANCE_TYPE"],
        env={
            "PC_BASE_DIR": os.environ["PC_BASE_DIR"],
            "SAMPLE_ONLY": os.environ["SAMPLE_ONLY"],
            "RANDOM_SPLIT": os.environ["RANDOM_SPLIT"],
            "TRAIN_RATIO": os.environ["TRAIN_RATIO"],
        },
        command=["python3"],
    )

    # defining the processing step
    processing_step = ProcessingStep(
        name="process-data",
        processor=processor,
        display_name="process data",
        description="This step is to convert the images to RecordIO files.",
        inputs=[
            ProcessingInput(
                source=os.path.join(os.environ["S3_PROJECT_URI"], "dataset"),
                destination=os.path.join(os.environ["PC_BASE_DIR"], "dataset"),
                input_name="dataset",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
            ProcessingInput(source="src/processing/im2rec.py", destination=os.path.join(os.environ["PC_BASE_DIR"], "utils"))
        ],
        outputs=[
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "train"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/train"
                ),
                output_name="train",
            ),
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "validation"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/validation"
                ),
                output_name="validation",
            ),
        ],
        code="src/processing/processing.py",
        cache_config=cache_config,
    )

    image_uri = image_uris.retrieve(framework="object-detection", region="eu-north-1")
    estimator = Estimator(
        image_uri=image_uri,
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["TRAINING_INSTANCE_COUNT"]),
        instance_type=os.environ["TRAINING_INSTANCE_TYPE"],
        input_mode="File",
        sagemaker_session=sagemaker_session,
        output_path=os.path.join(os.environ["S3_PROJECT_URI"], "training-output"),
        hyperparameters={
            "base_network":"resnet-50",
            "use_pretrained_model":"1",
            "num_classes":os.environ["NUM_CLASSES"],
            "mini_batch_size":"16",
            "epochs":os.environ["EPOCHS"],
            "learning_rate":os.environ["LEARNING_RATE"],
            "lr_scheduler_step":os.environ["LR_SCHEDULER_STEP"],
            "lr_scheduler_factor":"0.1",
            "optimizer":"sgd",
            "momentum":"0.9",
            "weight_decay":"0.0005",
            "overlap_threshold":"0.5",
            "nms_threshold":"0.45",
            "image_shape":"512",
            "label_width":"350",
            "num_training_samples":os.environ["NUM_TRAINING_SAMPLES"],
        },
    )

    training_step = TrainingStep(
        name="training-step",
        step_args=estimator.fit(
            inputs={
                "train": TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                    content_type="application/x-recordio",
                    s3_data_type="S3Prefix",
                ),
                "validation": TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                        "validation"
                    ].S3Output.S3Uri,
                    content_type="application/x-recordio",
                    s3_data_type="S3Prefix",
                ),
            },
        ),
        cache_config=cache_config,
    )

    # build the pipeline
    pipeline = Pipeline(
        name="birds-detection-pipeline",
        steps=[
            processing_step,
            training_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(
        role_arn=os.environ["SM_EXEC_ROLE"],
        description="A pipeline to detect birds in images using SSD with ...",
    )

    pipeline.start()


if __name__ == "__main__":
    run_pipeline()
