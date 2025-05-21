 # Fine_Tuning_Qwen2.5_LLaMA_Factory

This project fine-tunes the `Qwen/Qwen2.5-1.5B-Instruct` large language model (LLM) using the LLaMA-Factory framework to enhance its capabilities for structured data extraction and translation tasks in Arabic. The project includes data processing, model fine-tuning, inference, and performance evaluation using vLLM for efficient serving and Locust for load testing. The Jupyter notebook (`LLM_Fine_Tuning (2).ipynb`) is the primary implementation, and this README provides a detailed walkthrough of each step.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Imports and Initial Configurations](#imports-and-initial-configurations)
- [Task Definitions](#task-definitions)
  - [Details Extraction](#details-extraction)
  - [Translation](#translation)
- [Model Loading and Inference](#model-loading-and-inference)
- [Evaluation with DeepSeek-R1](#evaluation-with-deepseek-r1)
- [Knowledge Distillation](#knowledge-distillation)
- [Cost Estimation and Performance Testing](#cost-estimation-and-performance-testing)
- [vLLM Integration](#vllm-integration)
- [Load Testing with Locust](#load-testing-with-locust)
- [How to Run the Project](#how-to-run-the-project)
- [License](#license)

## Project Overview
The project aims to fine-tune the `Qwen/Qwen2.5-1.5B-Instruct` model for two tasks:
1. **Details Extraction**: Extract structured information (e.g., title, keywords, summary, category, and entities) from Arabic news articles using a Pydantic schema.
2. **Translation**: Translate Arabic news articles into English while preserving context and structure.

It leverages the LLaMA-Factory framework for fine-tuning, vLLM for efficient inference, and Locust for load testing to evaluate performance under concurrent requests. The fine-tuned model is optimized for Arabic text processing and tested for speed and scalability.

## Prerequisites
To run this project, you need:
- **Hardware**: A GPU-enabled machine (e.g., NVIDIA T4, as used in Google Colab).
- **Software**:
  - Python 3.11 or higher
  - Google Colab (optional, for cloud execution)
  - Google Drive for data and model storage
  - API keys for:
    - Weights & Biases (W&B) for experiment tracking
    - Hugging Face for model access
    - OpenRouter for DeepSeek-R1 evaluation
- **Dependencies**: Python packages specified in the notebook’s setup section.

## Setup
The notebook sets up the environment by:
1. **Mounting Google Drive**: Connects to Google Drive for accessing datasets and storing models.
2. **Installing Dependencies**: Installs packages like `transformers`, `datasets`, `optimum`, `openai`, `wandb`, `json-repair`, `faker`, and `vllm` for fine-tuning, data processing, and inference.
3. **Cloning LLaMA-Factory**: Retrieves the LLaMA-Factory repository and installs it with PyTorch and metrics dependencies.
4. **Configuring Authentication**: Logs into W&B and Hugging Face using API keys stored in Google Colab’s `userdata`.

## Imports and Initial Configurations
The notebook imports libraries for data processing, model handling, and JSON parsing, including `json`, `os`, `tqdm`, `random`, `requests`, `pydantic`, `transformers`, `torch`, `json_repair`, and `faker`. It configures:
- The data directory as `/gdrive/MyDrive/Fine-Tuning`.
- The base model as `Qwen/Qwen2.5-1.5B-Instruct`.
- The device as `cuda` for GPU usage.
- A JSON parsing function with error recovery using `json_repair`.

## Task Definitions
The notebook defines two tasks using a sample Arabic news story about family influence on financial behaviors.

### Details Extraction
This task extracts structured information from the Arabic news story, which discusses a Forbes report and research by Professor Shane Enye on financial behaviors, outlining three dimensions (Acquisition, Use, Management) and the Money Genogram tool.

- **Schema**: Uses a Pydantic `NewsDetails` class to define fields for title, keywords, summary, category, and entities (e.g., person, organization).
- **Process**: Constructs system and user prompts to instruct the model to extract details in Arabic, adhering to the schema.
- **Output**: Produces a JSON object with the extracted title, keywords, summary points, category (Economy), and entities like “Forbes” (organization) and “شاين إنيت” (person-male).

### Translation
This task translates the Arabic news story into English.

- **Schema**: Uses a Pydantic `TranslatedStory` class with fields for the translated title and content.
- **Process**: Constructs prompts to translate the story into English while maintaining context.
- **Output**: Generates a JSON object with a translated title and content, summarizing the Forbes report and its findings.

## Model Loading and Inference
The notebook loads the Qwen2.5-1.5B-Instruct model and tokenizer using the `transformers` library, with automatic device mapping for GPU usage. It defines a function to process messages, tokenize inputs, and generate outputs, executing inference for both tasks and parsing JSON responses with `json_repair`.

## Evaluation with DeepSeek-R1
The notebook evaluates the fine-tuned model by comparing its outputs to those of DeepSeek-R1 via the OpenRouter API.

- **Setup**: Configures an OpenAI client with an OpenRouter API key.
- **Evaluation**: Sends details extraction and translation prompts to DeepSeek-R1, parsing responses with `json_repair`.
- **Purpose**: Uses DeepSeek-R1 as a reference to assess the quality and accuracy of the fine-tuned model’s outputs.

## Knowledge Distillation
The notebook prepares a dataset for fine-tuning by loading and shuffling 2400 news samples from a JSONL file in `/gdrive/MyDrive/Fine-Tuning/Datasets/news-sample.jsonl`. This dataset supports fine-tuning to enhance the model’s performance on Arabic news-related tasks.

## Fine-Tuning
The fine-tuning process adapts the `Qwen/Qwen2.5-1.5B-Instruct` model to improve its performance on Arabic news-related tasks using the LLaMA-Factory framework.

- **Dataset**: Utilizes a dataset of 2400 Arabic news samples from a JSONL file, loaded and shuffled to ensure randomness.
- **Process**: The notebook integrates with LLaMA-Factory to fine-tune the model, applying LoRA (Low-Rank Adaptation) to efficiently update model weights. The fine-tuned model is saved to `/gdrive/MyDrive/Fine-Tuning/Datasets/LLaMaFactory-Finetuning-data/models`.
- **Purpose**: Enhances the model’s ability to handle structured data extraction and translation tasks in Arabic, leveraging knowledge distillation from the dataset or a larger model’s outputs.
- **Tracking**: Uses Weights & Biases to monitor training metrics, ensuring the fine-tuning process is optimized for performance and accuracy.

## Loading Fine tuded lora darptor
```
finetuned_model_id = "/gdrive/MyDrive/youtube-resources/llm-finetuning/models"
model.load_adapter(finetuned_model_id)
```

## Cost Estimation and Performance Testing
The notebook tests the fine-tuned model’s inference speed using synthetic Arabic prompts generated by `faker`.

- **Setup**: Processes 30 random prompts (150-200 characters) and measures input/output tokens and total time.
- **Results**: Calculates tokens per second (e.g., ~22.25 tokens/second for 13,640 tokens over 613 seconds).
- **Purpose**: Estimates computational costs and efficiency for deployment.

## vLLM Integration
The notebook uses vLLM for efficient inference with the fine-tuned model.

- **Setup**: Configures a vLLM server with LoRA weights from the fine-tuned model, optimizing GPU memory usage.
- **Inference**: Sends a translation prompt to the vLLM server, retrieving and parsing the JSON response.
- **Purpose**: Ensures efficient and scalable inference for real-world applications.

## Load Testing with Locust
The notebook evaluates the model’s performance under concurrent requests using Locust.

- **Setup**: Defines a Locust script to simulate multiple users sending random Arabic prompts to the vLLM server, saving responses to a file.
- **Execution**: Runs Locust with 20 users, a 1 user/second spawn rate, and a 60-second duration.
- **Results**: Processes 49 requests with no failures, achieving ~1.15 requests/second, with response times ranging from 103ms to 21,280ms (median 310ms).
- **Purpose**: Assesses scalability and reliability for deployment scenarios.

## How to Run the Project
1. **Clone the Repository**:
   - Clone the project from GitHub and navigate to the project directory.
2. **Set Up Environment**:
   - Install required Python packages.
   - Clone and install LLaMA-Factory with PyTorch and metrics dependencies.
3. **Configure API Keys**:
   - Store W&B, Hugging Face, and OpenRouter API keys in your environment or Google Colab’s `userdata`.
4. **Prepare Data**:
   - Place the `news-sample.jsonl` dataset in `/gdrive/MyDrive/Fine-Tuning/Datasets/`.
5. **Run the Notebook**:
   - Open and execute `LLM_Fine_Tuning.ipynb` in Google Colab or Jupyter Notebook.
6. **Run vLLM Server**:
   - Start the vLLM server with the fine-tuned model and LoRA weights.
7. **Perform Load Testing**:
   - Run the Locust script to simulate concurrent requests and generate a performance report.
