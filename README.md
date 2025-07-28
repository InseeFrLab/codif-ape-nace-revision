# 📌 CAG vs RAG: Centralized Repository for NACE Revision

## 📑 Table of Contents

- [📌 CAG vs RAG: Centralized Repository for NACE Revision](#-cag-vs-rag-centralized-repository-for-nace-revision)
  - [📑 Table of Contents](#-table-of-contents)
  - [📖 Overview](#-overview)
  - [🚀 Getting Started](#-getting-started)
    - [🛠 Installation](#-installation)
    - [🏗 Pre-commit Setup](#-pre-commit-setup)
    - [Cache LLM model](#cache-llm-model)
  - [📜 Running the Scripts](#-running-the-scripts)
    - [✅ 1. Build Vector Database (if you are in the RAG case)](#-1-build-vector-database-if-you-are-in-the-rag-case)
    - [🏷 2. Encode Business Activity Codes](#-2-encode-business-activity-codes)
    - [🔬 3. Evaluate Classification Strategies](#-3-evaluate-classification-strategies)
    - [📊 4. Build the NACE 2025 Dataset](#-4-build-the-nace-2025-dataset)
  - [📡 LLM Integration](#-llm-integration)
  - [🏗 Argo Workflows](#-argo-workflows)
  - [📄 License](#-license)
  - [TODO](#todo)


## 📖 Overview
This repository is dedicated to the revision of the **Nomenclature statistique des Activités économiques dans la Communauté Européenne (NACE)**.

It provides tools for **automated classification and evaluation of business activity codes** using **Large Language Models (LLMs)** and vector-based retrieval systems.


## 🚀 Getting Started

### 🛠 Installation
Ensure you have **Python 3.12+** and **uv** or **pip** installed, then install the required dependencies:

```bash
uv sync
```

or

```bash
uv pip install -r pyproject.toml
```

### 🏗 Pre-commit Setup
Set up linting and formatting checks using `pre-commit`:

```bash
uv run pre-commit autoupdate
uv run pre-commit install
```

### Cache LLM model
Before running the script download the model and put it in cache using the huggingface CLI which is faster than vllm to download the model.

```bash
export MODEL_NAME=Qwen/Qwen2.5-0.5B
uv run huggingface-cli download $MODEL_NAME
```

## 📜 Running the Scripts

### ✅ 1. Build Vector Database (if you are in the RAG case)

To create a searchable database of NACE 2025 codes:

```bash
uv run src/build_vector_db.py
```

### 🏷 2. Encode Business Activity Codes

For **unambiguous** classification:

```bash
uv run src/encode_unambiguous.py
```

For **ambiguous** classification using an LLM:

```bash
uv run src/encode_ambiguous.py --experiment_name NACE2025_DATASET --llm_name Qwen/Qwen3-0.6B
```

### 🔬 3. Evaluate Classification Strategies

> ⚠️ **TO BE UPDATED**

Compare different classification models:

```bash
uv run src/evaluate_strategies.py
```

### 📊 4. Build the NACE 2025 Dataset

Once all unique ambiguous cases have been recoded using the best strategy, you can rebuild the entire dataset with NACE 2025 labels:

```bash
uv run src/build_nace2025_sirene4.py
```

## 📡 LLM Integration
This repository leverages **Large Language Models (LLMs)** to assist in classifying business activities. One can also use all open source models available on HuggingFace and compatible with vLLM.


## 🏗 Argo Workflows
This project supports **automated workflows** via [Argo Workflows](https://argoproj.github.io/argo-workflows/).
To trigger a workflow, execute:

```yaml
argo submit argo-workflows/relabel-naf08-to-naf25.yaml
```

Or use the **Argo Workflow UI**.


## 📄 License
This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.


## TODO

1. Mieux versionner les prompts qui sont sauvegardé dans s3. Idealement il faudrait versionner via la collection de la base de données + la version du prompt RAG de langfuse
2. améliorer les embeddings (faut il garder les notes exlicatives en entier pour faire la similarity search ?)
3. implementer le reranker (celui de qwen probablement)
4. Inclure des règles métiers dans les prompts
5. Inclure des règles code spécifique dans le cas du CAG. (Si LMNP alors on explique ce qui fait la distinction entre les deux code -- cf le fichier de @Nathan) --> Du coup inclure des variables annexes pour aider à départager certaines fois ?
