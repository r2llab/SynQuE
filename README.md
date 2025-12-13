# SynQuE: Estimating Synthetic Dataset Quality Without Annotations

This repository is the official implementation of SynQuE: Estimating Synthetic Dataset Quality Without Annotations.

<p align="center">
  <img src="./assets/synque_overview.png" alt="SynQuE Overview" width="1000"/>
</p>

<p align="center">
  Overview of the SynQuE framework for synthetic data quality estimation across multiple domains.
</p>


## Requirements
We use `uv` for package management. To install `uv`, please follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
To install dependencies, all you need to do is run:
```bash
uv sync
```

## Training
The following instructions are provided for training models to obtain task performance metrics on real test data.
All synthetic and real datasets are provided in `data/`. All training results and SynQuE scores are provided in `notebooks/results`. Embeddings used for PAD, MMD and MDM calculations can be downloaded on [Google Drive](https://drive.google.com/drive/folders/1GRfPEDdqEWaDPM2wFIFxZLkwjnQwwVru?usp=sharing).

### Sentiment Analysis
The training script is placed in `notebooks/SynQuE_scoring.ipynb`.

### Text2SQL
We use the [CodeS](https://github.com/RUCKBReasoning/codes) codebase for Text2SQL training. Training text-to-sql pairs are provided in `data/text2sql/data/`. Please follow the instructions in the CodeS repository for training the model.
We use Qwen2.5-1.5B-Coder-Instruct as the base model. For each synthetic dataset, we train the model on its training data for 6 epochs with a learning rate of 5e-6, warmup ratio of 0.05, input block size of 4096. The rest of the hyperparameters are set to the default values in the CodeS repository.

### Web Navigation Agent
We use the original [NNetNav-Live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) dataset as synthetic data source. The original dataset is in conversation format and we use [open-instruct](https://github.com/allenai/open-instruct) to finetune Qwen2.5-7B-Instruct models on each disjoint subset of the dataset. We use LoRA with a LoRa rank of 64, batch size of 32, learning rate of 2e-5, warmup ratio of 0.03 for 3 epochs. The rest of the hyperparameters are the same as in the [NNetNav](https://github.com/MurtyShikhar/NNetnav) repository.

### Image Classification
We use ResNet-50 as the base model and train from scratch. We use 10% of the synthetic data as validation dataset and early stop on the validation loss. Models are trained for at least 10 epochs and we select the best model w.r.t validation loss to compute the task performance on real test data. ImageNet dataset can be downloaded from [here](https://www.image-net.org/download.php). Synthetic datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1hvFL3b6XUNVr_pyFD_qvBEV2AWE0fQyE?usp=sharing)

## LENS Scoring
LENS consists of two stages: 1. Rubric compilation and 2. Scoring with SynQuE.

### Rubric Compilation
All scripts are in `compile_rubrics/`. Please set the `OPENAI_API_BASE` environment variable to the base URL of your service provider. For configuration files, please refer to `compile_rubrics/conf/` as we use Hydra for configuration management, where you can easily override the default values (shown in the below example).
- Example for sentiment analysis:
    ```bash
    python compile_rubrics/generate_rubrics_sentiment_analysis.py \
        sentiment_analysis.rubric_generation.real_seed=42 \
        sentiment_analysis.rubric_generation.num_points=10
    ```
### Scoring LENS
We use vLLM offline SDK for scoring LENS.
- For sentiment analysis:
    ```bash
    python -m src.generate.score_sentiment_analysis --scoring_model_path <path> --synthetic_data <optional> --seed <seed> [other options]
    ```
- For web agent:
    ```bash
    python -m src.generate.score_web_agent --scoring_model_path <path> --synthetic_data <optional> --seed <seed> [other options]
    ```
- For image classification:
    ```bash
    python -m src.generate.score_image_classification --scoring_model_path <path> --split <which split to use> --synthetic_data <optional> --rubric_path <path> [other options]
    ```
- For text2sql:
    ```bash
    python -m src.generate.score_text2sql --scoring_model_path <path> --synthetic_data <optional> --seed <seed> [other options]
    ```

### Computing correlation coefficients
#### LENS
To compute the correlation coefficients between LENS scores and task performance, you can run the following script:

    ```bash
    python -m src.compute.lens --data_path <path_to_lens_scores> --task <task>
    ```
#### Representation-based
To score and compute correlation coefficients for representation-based methods, you can run the following script:
    ```bash
    python -m src.compute.representation_based --method <pad/mmd/mdm> --task <task> --embedding_path <path_to_the_embedding_folder>
    ```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
