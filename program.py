from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os
from sagemaker.workflow.steps import CacheConfig
from sagemaker.processing import ScriptProcessor
from sagemaker import image_uris


def run_pipeline():
    load_dotenv()

    cache_config = CacheConfig(enable_caching=False, expire_after="10d")

    sagemaker_session = PipelineSession(
        default_bucket=os.environ["S3_BUCKET_NAME"],
    )

    # define the processor
    image_uri = image_uris.retrieve(framework="object-detection", region="eu-north-1")
    processor = ScriptProcessor(
        image_uri=image_uri,
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["PROCESSING_JOB_INSTANCE_COUNT"]),
        instance_type=os.environ["PROCESSING_JOB_INSTANCE_TYPE"],
        env={},
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
                source=os.path.join(os.environ["S3_BUCKET_URI"], "dataset"),
                destination=os.path.join(os.environ["PC_BASE_DIR"], "dataset"),
                input_name="dataset",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
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
                    os.environ["S3_PROJECT_URI"], "processing-step/vlaidation"
                ),
                output_name="validation",
            ),
        ],
        code="src/processing.py",
        cache_config=cache_config,
    )

    # build the pipeline
    pipeline = Pipeline(
        name="birds-detection-pipeline",
        steps=[
            processing_step,
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
