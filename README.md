# Towards Chain-of-Thought Reasoning for Video Understanding with Gemini: An Agentic Chain-of-Chain-of-Thoughts Framework

A comprehensive platform for analyzing video content using Google's Gemini models via the unified `google-genai` SDK, supporting both Vertex AI and Gemini API backends. This project enables downloading videos, preparing them (including **critical optimizations like video slowdown**), and generating intelligent responses using various prompting strategies. It notably introduces an **Agentic Chain-of-Chain-of-Thoughts (CoCoT)** framework, leveraging **specifically formatted Chain-of-Thought prompts** even with CoT-capable models, primarily showcased in `Full_Inference_CoCoT_Generated_Questions.ipynb`, for sophisticated context-aware reasoning.

![Video Analysis](https://img.shields.io/badge/Video-Analysis-blue)
![Gemini Models](https://img.shields.io/badge/Gemini-2.x%20Pro/Flash-orange)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)
![Vertex AI](https://img.shields.io/badge/Vertex-AI-purple)
![Gemini API](https://img.shields.io/badge/Gemini-API-red)
![GCS](https://img.shields.io/badge/Google_Cloud-Storage-lightgrey)
![File API](https://img.shields.io/badge/Gemini-File_API-lightgrey)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets-yellow)

## Overview & Key Optimizations

This project provides a modular platform for advanced video understanding using Google's Gemini models. Key to its performance are several optimizations:

1.  **Video Slowdown:** Globally configurable (`VIDEO_SPEED_FACTOR`), this technique (e.g., 0.5x speed) significantly improves the model's ability to capture temporal details and nuances within video content, leading to better comprehension and more accurate answers. This is applied during the video preparation stage using `ffmpeg`.
2.  **Optimized Chain-of-Thought (CoT) Prompting Formats:** Even when using models designed for CoT, the specific structure and phrasing of our prompts are crucial. We employ carefully crafted templates (see `models/CoT_output_models.py`) to guide the model through a detailed reasoning process, eliciting more comprehensive and accurate thought chains.
3.  **Agentic Chain-of-Chain-of-Thoughts (CoCoT) Framework:** This advanced strategy, implemented in `Full_Inference_CoCoT_Generated_Questions.ipynb`, uses pre-generated guideline questions and their CoT answers as conversational context. This allows the model to build a rich understanding of the video before tackling the original, often more complex, dataset questions.

## Core Notebooks & Inference Strategies

The platform's capabilities are primarily accessed through the following Jupyter notebooks, each designed for specific tasks and incorporating the aforementioned optimizations:

*   **`Full_Inference_CoCoT_Generated_Questions.ipynb` (Enhanced Agentic CoCoT - Recommended for Optimal Performance):**
    *   **Purpose:** Implements our latest, enhanced CoT architecture designed explicitly for updated competition requirements. This is the **recommended primary solution** due to its superior overall structure and reasoning effectiveness.
    *   **Execution & Optimizations:**
        *   Leverages context-aware reasoning by using conversational history (generated by `Generated_Questions_By_Videos.ipynb`) from a series of heuristically ordered guideline questions.
        *   Employs **optimized CoT prompting formats** (from `models/CoT_output_models.py`) for both answering guideline questions and the final original dataset question, ensuring detailed thought processes.
        *   Benefits significantly from **video slowdown** applied during video preparation, allowing the model to better analyze fine-grained temporal events.
        *   Optionally uses a summary model to distill the final CoT answer.
    *   **Output:** Detailed CSV results including full CoT reasoning, ideal for in-depth analysis.

*   **`Full_Inference.ipynb` (Stricter, Question-by-Question Inference):**
    *   **Purpose:** Provides a robust alternative designed for explicit, isolated processing of each question, meticulously adhering to the most conservative interpretation of competition guidelines.
    *   **Execution & Optimizations:**
        *   Processes each original dataset question individually without leveraging external conversational context from generated questions.
        *   Can use Non-CoT models for direct answers or basic CoT models with **specifically formatted CoT prompts** for single-turn reasoning on the original question.
        *   Also benefits from **video slowdown** for improved detail capture.
    *   **Output:** CSV results, structured for simplicity and clear auditability.

*   **`Generated_Questions_By_Videos.ipynb` (Context Generation for Advanced CoCoT):**
    *   **Purpose:** A crucial pre-processing step for the `Full_Inference_CoCoT_Generated_Questions.ipynb` notebook.
    *   **Execution & Optimizations:**
        1.  Generates a list of relevant guideline questions for each video, benefiting from **video slowdown** for more insightful question generation.
        2.  Answers these generated questions sequentially using **optimized CoT prompting formats**, with the (slowed-down) video provided at each turn, to build a rich conversational history.
    *   **Output:** `questions.csv` (generated questions) and JSON files in `chat_history/` (serialized conversation history per video).

*   **`Testing_UI_Prompting.ipynb` (Interactive Exploration & Prompt Engineering):**
    *   **Purpose:** Facilitates interactive testing of individual videos, questions, and the CoCoT mechanism.
    *   **Execution & Optimizations:**
        *   Allows users to select a video (which would have undergone **video slowdown** if `VIDEO_SPEED_FACTOR` is set) and an original question.
        *   Demonstrates the CoCoT context-building process.
        *   Enables experimentation with different **prompt formats** and immediate observation of their impact on model responses.

## Platform Process

The platform streamlines video understanding tasks through:

1.  Fetching question/video metadata from HuggingFace (`lmms-lab/AISG_Challenge`).
2.  Downloading associated videos.
3.  Processing videos (including **video slowdown** using `ffmpeg`).
4.  Uploading prepared videos to Google Cloud Storage (Vertex AI) or Gemini File API.
5.  Performing bulk inference using the strategies outlined in the Core Notebooks section.
6.  Saving results to CSV and JSON (for chat histories) for analysis.

## Architecture

The platform's workflow:

1.  **Data Source:** HuggingFace `datasets` for metadata; ZIP archive for videos.
2.  **Data Preparation:** Video download, extraction, `ffmpeg` processing (speed adjustment), upload to GCS/File API, and `video_metadata_*.csv` creation.
3.  **Storage:** Processed videos in GCS or Gemini File API; metadata/results in local CSVs/JSONs.
4.  **Inference Engine:** `google-genai` SDK for Gemini models.
    *   **Vertex AI Backend:** ADC/service account auth; requires `PROJECT_ID`, `LOCATION`, `GCS_BUCKET`.
    *   **Gemini API Backend:** API Key; uses File API (files expire ~1 day).
5.  **Execution & Control:** Orchestrated by the Jupyter Notebooks detailed above.
6.  **Results:** Inference outputs in CSVs (e.g., `results_*.csv`); chat histories in `generated_questions/` as JSON.

## General Features

-   **Dual Backend Support**: Vertex AI & Gemini API via `USE_VERTEX` flag.
-   **Unified SDK**: Modern `google-genai` library.
-   **Flexible Storage**: GCS (Vertex) or File API (Gemini API).
-   **Efficient & Robust Processing**: Skip flags for existing data, `asyncio` for speed, basic API retries.
-   **Comprehensive Metadata Tracking**: `video_metadata_*.csv` for resource management.
-   **Modular Model Configuration**: Prompts and model settings defined in `models/`.
-   **Clear Logging & Resume Support**.

## Installation

### Prerequisites

*   **Python**: 3.10+.
*   **Package Management**: `pip`, `virtualenv`.
*   **Google Cloud Account**: Required.
    *   **Vertex AI**: Billing, Vertex AI API, Cloud Storage API enabled.
    *   **Gemini API**: Gemini API Key from [AI Studio](https://aistudio.google.com/app/apikey).
*   **Google Cloud CLI (`gcloud`)**: For Vertex AI auth. Install from [official guide](https://cloud.google.com/sdk/docs/install).
*   **`ffmpeg`**: For video processing. Install from [ffmpeg.org](https://ffmpeg.org/download.html).
    *   Linux: `sudo apt update && sudo apt install ffmpeg`
    *   macOS: `brew install ffmpeg`
    *   Windows: `winget install --id=Gyan.FFmpeg -e`
    *   Verify: `ffmpeg -version`

### Google Cloud Setup (Vertex AI Backend)

If you plan to use the Vertex AI backend, follow these steps to configure your Google Cloud environment:

1.  **Install the Google Cloud CLI**.
2.  **Initialize the Google Cloud CLI**: `gcloud init`
3.  **Log in to Google Cloud**: `gcloud auth login`
4.  **Configure Application Default Credentials (ADC)**: `gcloud auth application-default login`
5.  **Set Your Default Project and Region**: `gcloud config set project YOUR_PROJECT_ID`, `gcloud config set compute/region YOUR_REGION`

### Python Environment Setup

1.  **Clone the repository:** `git clone https://github.com/Team-SeekDeep/TikTokSubmission && cd TikTokSubmission`
2.  **Create and activate a virtual environment.**
3.  **Install dependencies:** `pip install -r requirements.txt`

## Configuration

Key settings in the **Config Settings** cell of each notebook:
-   `PROJECT_ID`, `LOCATION`, `GCS_BUCKET` (Vertex AI).
-   `USE_VERTEX` (Backend choice).
-   `GEMINI_API_KEY` (Gemini API, prefer env var `GOOGLE_API_KEY`).
-   File paths (`DATASET_CSV`, `METADATA_FILE`, `RESULTS_FILE`, `QUESTIONS_DIR`, `ANSWERS_DIR`).
-   `SKIP_*` flags for bypassing processed steps.
-   `MAX_VIDEOS_TO_PROCESS` for testing.
-   `MODEL_NAME`, `QUESTION_MODEL_NAME` (for CoCoT/Summary models).
-   `VIDEO_SPEED_FACTOR` (e.g., `0.5` for half speed, crucial for performance).

**Review and adjust configuration before running any notebook.**

## Usage

1.  **Launch Jupyter** (`jupyter lab` or `jupyter notebook`).
2.  **Open a Notebook** based on your goal (see "Core Notebooks & Inference Strategies").
3.  **Configure** settings in that notebook.
4.  **Run Cells Sequentially**, paying attention to outputs and logs.
    *   **Data Preparation:** Ensure videos are downloaded, (slowed down if `VIDEO_SPEED_FACTOR < 1.0`), and uploaded.
    *   **Context Generation (for CoCoT):** Run `Generated_Questions_By_Videos.ipynb` before `Full_Inference_CoCoT_Generated_Questions.ipynb`.
    *   **Inference:** Execute the chosen full inference notebook.

**Switching Backends (Vertex <-> Gemini API):**
Re-run "Prepare Videos" step (`SKIP_PREPARE = False`) after changing `USE_VERTEX`. `SKIP_DOWNLOAD_ZIP` and `SKIP_EXTRACT` can be `True` if local videos exist.

## Common Issues

1.  **`ffmpeg` Not Found.**
2.  **Google Cloud Authentication Errors (Vertex AI).**
3.  **Gemini API Key Errors.**
4.  **File API Not Found / Expired (Gemini API).**
5.  **GCS Errors (Vertex AI).**
6.  **Video Download/Extraction Issues.**
7.  **API Quota Errors (`ResourceExhausted`).**
8.  **Model Not Found / Invalid Model Name.**

## Logs

Python's `logging` module outputs to Jupyter cells. For long runs (bulk inference), consider redirecting logs to a file.

## Best Practices & Recommended Usage

-   **Submission Strategy / Recommended Usage:**
    -   For achieving the **best benchmark scores and aligning with updated competition requirements that benefit from advanced reasoning**, we **strongly recommend using the `Full_Inference_CoCoT_Generated_Questions.ipynb` notebook.** Its enhanced CoT architecture, context-aware reasoning using generated questions, optimized CoT prompting, and benefits from video slowdown provide superior performance.
    -   The `Full_Inference.ipynb` notebook serves as an **essential alternative for demonstrating strict compliance** with guidelines requiring isolated, question-by-question analysis without complex contextual chaining. It offers methodological rigor and transparency.
-   **Virtual Environments:** Always use.
-   **Configuration Management:** Review settings carefully; use `SKIP_*` flags effectively. Set `SKIP_PREPARE=False` when switching backends. Remember to set `VIDEO_SPEED_FACTOR` (e.g., to `0.5`) for better performance.
-   **API Key Security:** **Never commit API keys.** Use environment variables.
-   **Resource Cleanup & Cost Management:** Be mindful of GCS/Vertex AI/Gemini API costs and File API expiration.
-   **Version Control (`.gitignore`):**
    ```gitignore
    # Python
    __pycache__/
    *.py[cod]
    *$py.class

    # Environment
    venv/
    .env*

    # Data / Cache / Videos
    downloads/
    extracted_videos/
    speed_videos/
    hf_cache/
    *.zip
    *.mp4
    video_metadata_*.csv
    results_*.csv
    all_results/
    generated_questions/ # Contains generated CSVs and JSONs

    # Jupyter
    .ipynb_checkpoints/

    # Logs
    *.log

    # IDE/OS specific
    .DS_Store
    .vscode/
    ```
-   **Understand Backend Differences.**

## Support and Contact

For detailed support, notebook-specific issues, or configuration questions, please contact Dylan at dadevchia@gmail.com, referencing the notebook name and specific configuration settings involved.

To help diagnose the problem effectively, please include the following information in your communication:

1.  **Which Notebook?**
2.  **Which Cell?**
3.  **Configuration:** (especially `USE_VERTEX`, `MODEL_NAME`, `VIDEO_SPEED_FACTOR`, relevant `SKIP_*` flags). **Remove API Keys.**
4.  **Error Message:**
5.  **Goal:**
6.  **(If using Vertex AI):** `gcloud info` output.

## License

MIT License