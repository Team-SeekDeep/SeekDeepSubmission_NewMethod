# Towards Chain-of-Thought Reasoning for Video Understanding with Gemini: An Agentic Chain-of-Chain-of-Thoughts Framework

A comprehensive platform for analyzing video content using Google's Gemini models via the unified `google-genai` SDK, supporting both Vertex AI and Gemini API backends. This project enables downloading videos from a HuggingFace dataset, preparing them (including speed adjustment), and generating intelligent responses to questions about the videos. It notably introduces an **Agentic Chain-of-Chain-of-Thoughts (CoCoT)** framework, primarily showcased in `Full_Inference_CoCoT_Generated_Questions.ipynb`, for sophisticated context-aware reasoning. A more direct, question-by-question approach for strict guideline adherence is available in `Full_Inference.ipynb`.

# Overview & Functionality

This project provides a modular platform for advanced video understanding using Google's Gemini models, supporting both Vertex AI and direct Gemini API access via the `google-genai` SDK.
The system processes video datasets and associated questions using diverse prompting strategies, offering tools for large-scale evaluation and interactive UI-based experimentation.

![Video Analysis](https://img.shields.io/badge/Video-Analysis-blue)
![Gemini Models](https://img.shields.io/badge/Gemini-2.x%20Pro/Flash-orange)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)
![Vertex AI](https://img.shields.io/badge/Vertex-AI-purple)
![Gemini API](https://img.shields.io/badge/Gemini-API-red)
![GCS](https://img.shields.io/badge/Google_Cloud-Storage-lightgrey)
![File API](https://img.shields.io/badge/Gemini-File_API-lightgrey)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets-yellow)

## Platform Process

The platform streamlines video understanding tasks through:

1.  Fetching question/video metadata from HuggingFace (`lmms-lab/AISG_Challenge`).
2.  Downloading associated videos.
3.  Processing videos (e.g., `ffmpeg` for speed adjustment).
4.  Uploading prepared videos to Google Cloud Storage (Vertex AI) or Gemini File API.
5.  Performing bulk inference using:
    *   **Direct Question-Answering**: Via `Full_Inference.ipynb` for isolated, per-question analysis.
    *   **Advanced CoCoT**: Via `Full_Inference_CoCoT_Generated_Questions.ipynb` utilizing pre-generated questions and conversational history for context-rich reasoning.
6.  Providing an interactive UI for multi-turn conversational prompting tests.
7.  Saving results to CSV and JSON (for chat histories) for analysis.

## Notebook Specifics & Execution Details

-   **`Testing_UI_Prompting.ipynb`:**
    -   Purpose: Interactive testing of single questions and the CoCoT mechanism with generated questions.
    -   Execution: Allows selection of a video and an original question. For CoCoT, it demonstrates the context-building by using a few pre-generated questions and their CoT answers as history before posing the original question.

-   **`Generated_Questions_By_Videos.ipynb`:**
    -   Purpose: **Crucial pre-processing for the advanced CoCoT notebook.**
    -   Execution:
        1.  Generates a list of relevant guideline questions for each video using a specified Gemini model (`QUESTIONS_MODEL_NAME`).
        2.  Answers these generated questions sequentially (CoT-style, with video provided each turn) to build a rich conversational history.
    -   Output: `generated_questions/[QUESTIONS_MODEL_NAME]/questions.csv` (generated questions) and `generated_questions/[QUESTIONS_MODEL_NAME]/chat_history/[video_id].json` (serialized conversation history per video).

-   **`Full_Inference.ipynb` (Stricter, Question-by-Question, used for latest submission of results):**
    -   Purpose: Bulk inference adhering to guidelines requiring **isolated processing of each original question**.
    -   Execution: Iterates through original dataset questions. For each, it sends the video and the single original question to the configured model (`MODEL_NAME` from `models/NonCoT_output_models.py` or `models/CoT_output_models.py` for single-turn CoT). No external conversational history or batch context is used.
    -   Output: CSV results (e.g., `all_results/full_inference_nonCoT/[MODEL_NAME]/...`).

-   **`Full_Inference_CoCoT_Generated_Questions.ipynb` (Enhanced CoCoT):**
    -   Purpose: **Recommended for optimal performance.** Implements an advanced Agentic CoCoT framework, leveraging context-aware reasoning.
    -   Execution:
        1.  For each original dataset question:
            a.  Loads the pre-built conversational history (JSON) for the video from `Generated_Questions_By_Videos.ipynb`. This history consists of *generated guideline questions* and their CoT answers.
            b.  Constructs the prompt using the video, the loaded conversational history, and the current *original dataset question*. Heuristic approaches to question ordering within the generated context are implicitly handled by the `Generated_Questions_By_Videos.ipynb` process.
            c.  Sends this context-rich package to the CoT model (`MODEL_NAME` from `models/CoT_output_models.py`).
            d.  Optionally, a summary model (`QUESTION_MODEL_NAME` from `models/Summary_models.py`) processes the CoT output for a final concise answer.
        2.  Utilizes batch processing internally where feasible for API calls, managed by `asyncio` and rate limiters.
    -   Output: Detailed CSV results (e.g., `all_results/full_inference_CoCoT_generated_questions/[MODEL_NAME]/...`).


## Architecture

The platform's workflow:

1.  **Data Source:** HuggingFace `datasets` for metadata; ZIP archive for videos.
2.  **Data Preparation:** Video download, extraction, `ffmpeg` processing (speed), upload to GCS/File API, and `video_metadata_*.csv` creation.
3.  **Storage:** Processed videos in GCS or Gemini File API; metadata/results in local CSVs/JSONs.
4.  **Inference Engine:** `google-genai` SDK for Gemini models.
    *   **Vertex AI Backend:** ADC/service account auth; requires `PROJECT_ID`, `LOCATION`, `GCS_BUCKET`.
    *   **Gemini API Backend:** API Key; uses File API (files expire ~1 day).
5.  **Execution & Control (Jupyter Notebooks):**
    *   `Testing_UI_Prompting.ipynb`: Interactive single-question and multi-turn CoCoT testing with generated questions.
    *   `Generated_Questions_By_Videos.ipynb`: Generates guideline questions per video and their CoT answers, creating JSON chat histories essential for the advanced CoCoT framework.
    *   `Full_Inference.ipynb`: For bulk, **stricter question-by-question inference**. Processes each original dataset question in isolation using Non-CoT or basic CoT models. Ensures compliance with guidelines requiring unbatched, non-contextually-chained answers.
    *   `Full_Inference_CoCoT_Generated_Questions.ipynb`: Implements the **enhanced Agentic Chain-of-Chain-of-Thoughts (CoCoT)**. Leverages conversational history (from `Generated_Questions_By_Videos.ipynb`) and heuristic question ordering for context-aware reasoning on original dataset questions, designed for optimal performance and alignment with updated competition requirements.
6.  **Results:** Inference outputs in CSVs (e.g., `results_*.csv`); chat histories in `generated_questions/` as JSON.

## Features

-   **Dual Backend Support**: Vertex AI & Gemini API via `USE_VERTEX` flag.
-   **Unified SDK**: Modern `google-genai` library.
-   **Flexible Storage**: GCS (Vertex) or File API (Gemini API).
-   **Video Preprocessing**: `ffmpeg` for speed adjustment.
-   **Diverse Prompting Strategies**:
    -   **Direct Question Answering**: Isolated, per-question processing (`Full_Inference.ipynb`).
    -   **Agentic Chain-of-Chain-of-Thoughts (CoCoT)**: Utilizes generated questions, conversational history, and heuristic question ordering for deep contextual reasoning on original questions (`Full_Inference_CoCoT_Generated_Questions.ipynb`).
    -   **Automated Guideline Question Generation**: For CoCoT context building (`Generated_Questions_By_Videos.ipynb`).
-   **Efficient & Robust Processing**: Skip flags for existing data, `asyncio` for speed, basic API retries.
-   **Interactive UI & Metadata Tracking**: `ipywidgets` UI; `video_metadata_*.csv` for resource management.
-   **Modular Model Configuration**: Models defined in `models/`.
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

1.  **Install the Google Cloud CLI**:
    If you haven't already installed it during the prerequisite step, use one of the following methods:

    *   **Debian/Ubuntu:**
        ```bash
        curl -sSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
        sudo apt-get update && sudo apt-get install google-cloud-sdk
        ```
    *   **macOS (with Homebrew):**
        ```bash
        brew install --cask google-cloud-sdk
        ```
    *   **Windows (with Winget):**
        ```bash
        winget install GoogleCloudSDK.GoogleCloudSDK
        ```
    *   For other operating systems, refer to the [official installation guide](https://cloud.google.com/sdk/docs/install).

2.  **Initialize the Google Cloud CLI**:
    This command guides you through initial setup, including logging in and setting a default project.
    ```bash
    gcloud init
    ```

3.  **Log in to Google Cloud**:
    Ensure your CLI is authenticated with your Google Cloud account.
    ```bash
    gcloud auth login
    ```

4.  **Configure Application Default Credentials (ADC)**:
    This allows applications (like this project) to easily authenticate with Google Cloud APIs.
    ```bash
    gcloud auth application-default login
    ```

5.  **Set Your Default Project and Region**:
    Configure the CLI to use your specific Google Cloud project and preferred compute region. Replace `YOUR_PROJECT_ID` and `YOUR_REGION` with your actual values (e.g., `us-central1`).
    ```bash
    gcloud config set project YOUR_PROJECT_ID
    gcloud config set compute/region YOUR_REGION
    ```

### Python Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Team-SeekDeep/TikTokSubmission
    cd TikTokSubmission
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or if no requirements.txt is provided, install manually:
    # pip install google-genai google-cloud-storage datasets pandas tqdm ipywidgets requests nest_asyncio notebook jupyterlab
    ```
    *Note: `google-cloud-storage` is only strictly required if `USE_VERTEX = True`.*

## Configuration

Key settings in the **Config Settings** cell of each notebook:
-   `PROJECT_ID`, `LOCATION`, `GCS_BUCKET` (Vertex AI).
-   `USE_VERTEX` (Backend choice).
-   `GEMINI_API_KEY` (Gemini API, prefer env var `GOOGLE_API_KEY`).
-   File paths (`DATASET_CSV`, `METADATA_FILE`, `RESULTS_FILE`, `QUESTIONS_DIR`, `ANSWERS_DIR`).
-   `SKIP_*` flags for bypassing processed steps.
-   `MAX_VIDEOS_TO_PROCESS` for testing.
-   `MODEL_NAME`, `QUESTION_MODEL_NAME` (for CoCoT/Summary models).
-   `VIDEO_SPEED_FACTOR`.

**Review and adjust configuration before running any notebook.**

## Usage

1.  **Launch Jupyter** (`jupyter lab` or `jupyter notebook`).
2.  **Open Notebook** based on your goal.
3.  **Configure** settings in the notebook.
4.  **Run Cells Sequentially**.


**Switching Backends (Vertex <-> Gemini API):**
Re-run "Prepare Videos" step (`SKIP_PREPARE = False`) after changing `USE_VERTEX`. `SKIP_DOWNLOAD_ZIP` and `SKIP_EXTRACT` can be `True` if local videos exist.

## Common Issues

1.  **`ffmpeg` Not Found:**
    -   Ensure `ffmpeg` is installed correctly and its executable is in your system's PATH. Test with `ffmpeg -version` in your terminal/command prompt *before* launching Jupyter.
2.  **Google Cloud Authentication Errors (Vertex AI):**
    -   Verify you've run `gcloud auth application-default login`.
    -   Check that `PROJECT_ID` and `LOCATION` are set correctly in the notebook config.
    -   Ensure the authenticated user/service account has necessary permissions (`Vertex AI User`, `Storage Object Admin` roles).
    -   Ensure the Vertex AI API is enabled.
3.  **Gemini API Key Errors:**
    -   Ensure `USE_VERTEX = False`.
    -   Provide a valid `GEMINI_API_KEY` in the config, or set the `GOOGLE_API_KEY` environment variable.
4.  **File API Not Found / Expired (Gemini API):**
    -   Files uploaded via the File API expire (~1 day). You may need to re-run the "Prepare Videos" step (`SKIP_PREPARE = False`) to re-upload them.
5.  **GCS Errors (Vertex AI):**
    -   Verify `GCS_BUCKET` name is correct and the bucket exists.
    -   Check permissions for the bucket.
6.  **Video Download/Extraction Issues:**
    -   Verify `VIDEO_ZIP_URL` is correct and accessible.
    -   Check for sufficient disk space in `DOWNLOADS_DIR` and `EXTRACTED_VIDEOS_DIR`.
    -   Try setting `SKIP_DOWNLOAD_ZIP=False` or `SKIP_EXTRACT=False` to force re-download/re-extraction if you suspect corruption.
7.  **API Quota Errors (`ResourceExhausted`):**
    -   You've hit the requests-per-minute limit for the model/API tier.
    -   Check the quota limits for Vertex AI or Gemini API Free/Tier-1.
    -   Reduce concurrency (`MAX_ASYNC_WORKERS` in some notebooks) or introduce delays. The provided rate limiter helps but might need adjustment based on the specific limits.
    -   Consider upgrading your Gemini API tier if using the free tier heavily.
8.  **Model Not Found / Invalid Model Name:**
    -   Ensure the `MODEL_NAME` is valid for the selected backend (Vertex/Gemini API) and region (for Vertex).

## Logs

This project utilizes Python's standard `logging` module, configured to output detailed information directly to the Jupyter notebook cells during execution.

*   **Viewing Logs:** As you run cells within any of the `.ipynb` notebooks, logs containing timestamps, severity levels (INFO, WARNING, ERROR), and descriptive messages will appear in the output area *below* the corresponding cell. This provides real-time feedback on the progress of tasks like data fetching, video preparation (download, extraction, speed adjustment, upload), API interactions, and any potential errors encountered.
*   **Persistence:** By default, these logs exist only within the notebook's output cells. They are **not automatically saved** to a separate file. If you clear a cell's output, restart the notebook kernel, or close the notebook without saving the outputs, the logs for that session might be lost (though Jupyter often caches outputs).
*   **Saving Logs to a File (Recommended for Bulk Inference):** For long-running processes like the bulk inference tasks in `Full_Inference.ipynb` and`Full_Inference_CoCoT_Generated_Questions.ipynb`, it's highly recommended to capture logs persistently.


## Best Practices & Recommended Usage

-   **Submission Strategy / Recommended Usage:**
    -   For achieving the **best benchmark scores and aligning with updated competition requirements that benefit from advanced reasoning**, we **strongly recommend using the `Full_Inference_CoCoT_Generated_Questions.ipynb` notebook.** Its enhanced CoT architecture, context-aware reasoning using generated questions, and heuristic ordering provide superior performance.
    -   The `Full_Inference.ipynb` notebook serves as an **essential alternative for demonstrating strict compliance** with guidelines requiring isolated, question-by-question analysis without complex contextual chaining. It offers methodological rigor and transparency.
-   **Virtual Environments:** Always use.
-   **Configuration Management:** Review settings carefully; use `SKIP_*` flags effectively. Set `SKIP_PREPARE=False` when switching backends.
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

## Support and Contact

For detailed support, notebook-specific issues, or configuration questions, please contact Dylan at dadevchia@gmail.com, referencing the notebook name and specific configuration settings involved.

To help diagnose the problem effectively, please include the following information in your communication:

1.  **Which Notebook?** Specify the exact `.ipynb` file you were running (e.g., `Testing_UI_Prompting.ipynb`, `Full_Inference_CoCoT_Generated_Questions.ipynb`).
2.  **Which Cell?** Indicate the specific cell (e.g., by its execution count `[ ]:` number, or by describing its purpose like "Prepare Videos cell" or "Bulk Inference Loop cell") where the error occurred or the unexpected behavior was observed.
3.  **Configuration:** Provide the key configuration settings you were using from the **Config Settings** cell (especially `USE_VERTEX`, `MODEL_NAME`, `QUESTION_MODEL_NAME`, relevant `SKIP_*` flag values). **Please remove your API Key or any other sensitive credentials before sharing.**
4.  **Error Message:** Copy and paste the *complete* error message and traceback, if available.
5.  **Goal:** Briefly describe what you were trying to achieve when the issue occurred.
6.  **(If using Vertex AI):** The output of the `gcloud info` command run in your terminal can sometimes be helpful for diagnosing authentication or project configuration issues.

## License

MIT License