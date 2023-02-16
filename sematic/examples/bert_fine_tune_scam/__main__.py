"""
This is the entry point of your pipeline.

This is where you import the pipeline function from its module and resolve it.
"""
# Sematic
from sematic.examples.bert_fine_tune_scam.pipeline import (
    DataLoaderConfig,
    PipelineConfig,
    TrainConfig,
    pipeline,
)
import os
# from sematic import SilentResolver

PIPELINE_CONFIG = PipelineConfig(
    dataloader_config=DataLoaderConfig(),
    train_config=TrainConfig(epochs=10),
    data_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "spam_data.csv")
)

def main():
    """
    Entry point of my pipeline.
    """
    pipeline(PIPELINE_CONFIG).set(
        name="Fine-Tune BERT Example", tags=["pytorch", "example", "bert"]
    ).resolve()


if __name__ == "__main__":
    main()

