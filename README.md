# Towards Chain-of-Thought Reasoning for Video Understanding with Gemini: An Agentic Chain-of-Chain-of-Thoughts Framework

A comprehensive platform for analyzing video content using Google's Gemini models via the unified `google-genai` SDK, supporting both Vertex AI and Gemini API backends. This project enables downloading videos from a HuggingFace dataset, preparing them (including speed adjustment), and generating intelligent responses to questions about the videos using various prompting strategies. By generating questions using Gemini, we can apply Chain-of-Chain-of-Thoughts with conversation history to proceed one questions.

# Overview & Functionality

This project introduces a comprehensive and modular platform for advanced video understanding and question-answering, powered by Google's Gemini large language models. It supports both Google Cloud Vertex AI and the direct Gemini API through a unified interface using the google-genai SDK.
The system is built to process datasets of videos and associated questions using a variety of prompting strategies, offering practitioners tools for both fast inference large-scale evaluation of dataset and interactive experimentation for each video with UI.

![Video Analysis](https://img.shields.io/badge/Video-Analysis-blue)
![Gemini Models](https://img.shields.io/badge/Gemini-2.x%20Pro/Flash-orange)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)
![Vertex AI](https://img.shields.io/badge/Vertex-AI-purple)
![Gemini API](https://img.shields.io/badge/Gemini-API-red)
![GCS](https://img.shields.io/badge/Google_Cloud-Storage-lightgrey)
![File API](https://img.shields.io/badge/Gemini-File_API-lightgrey)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets-yellow)

## Platform Process

This platform streamlines the process of video understanding tasks by:

1.  Fetching question/video metadata from HuggingFace (`lmms-lab/AISG_Challenge`).
2.  Downloading the associated videos from a provided zip archive.
3.  Processing videos (e.g., adjusting playback speed using `ffmpeg`).
4.  Uploading prepared videos to either Google Cloud Storage (for Vertex AI backend) or the Gemini File API (for Gemini API backend).
5.  Performing bulk inference across the dataset using different models and prompting strategies (Non-CoT, CoCoT with heuristic question ordering).
6.  Providing an interactive UI within Jupyter notebooks for testing individual videos and multi-turn conversational prompting (MCQ chaining).
7.  Saving inference results to CSV files for analysis.

The modular architecture separates data preparation, inference logic, and UI testing, making the workflow maintainable and adaptable.

## Architecture

The platform follows a general workflow:

1.  **Data Source:** HuggingFace `datasets` library fetches metadata (`dataset.csv`). Video content comes from a downloadable ZIP archive.
2.  **Data Preparation:**
    *   Downloads the video archive (`.zip`).
    *   Extracts `.mp4` files.
    *   Processes videos using `ffmpeg` (e.g., slows down playback speed).
    *   Uploads processed videos to the appropriate storage backend (GCS bucket via `google-cloud-storage` or Gemini File API via `google-genai`).
    *   Creates/updates a metadata file (`video_metadata_*.csv`) linking questions to video resources (local path, GCS URI or File API name).
3.  **Storage:** Processed videos reside in Google Cloud Storage (Vertex mode) or are referenced via the Gemini File API (Gemini API mode). Metadata is stored locally in CSV files.
4.  **Inference Engine:** Leverages the `google-genai` SDK to interact with Gemini models.
    *   **Vertex AI Backend:** Uses Application Default Credentials (ADC) or service account authentication. Requires `PROJECT_ID`, `LOCATION`, and `GCS_BUCKET`.
    *   **Gemini API Backend:** Uses an API Key (`GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable). Uses the File API for video uploads (files expire after ~1 day).
5.  **Execution & Control:** Jupyter Notebooks (`.ipynb`) orchestrate the workflow, configure settings, define prompting strategies, execute inference, and provide testing UIs.
    *   `Testing_UI_Prompting.ipynb`: For interactive single-question and multi-turn (CoCoT) testing.
    *   `Full_Inference.ipynb`: For bulk processing using direct (Non-CoT or CoT) prompts.
    *   `Generated_Questions_By_Videos.ipynb`: For bulk processing to generate guideline questions for each video processing and generating conversation history for a batch of generated questions.
    *   `Full_Inference_CoCoT_Generated_Questions.ipynb`: For bulk processing using CoCoT with generated questions.
6.  **Results:** Inference outputs are saved to specified CSV files (e.g., `results_*.csv`, `all_results/*.csv`). Generated questions with conversation history of each video are saved as json file under generated_questions folder.

## Features

-   **Dual Backend Support**: Seamlessly switch between Vertex AI and Gemini API backends via the `USE_VERTEX` flag.
-   **Unified SDK**: Uses the modern `google-genai` library for both backends.
-   **Flexible Storage**: Uploads videos to GCS (Vertex) or File API (Gemini API) based on the chosen backend.
-   **Video Preprocessing**: Includes functionality to adjust video speed using `ffmpeg`.
-   **Diverse Prompting Strategies**:
    -   Direct Answering (Non-CoT)
    -   Chain-of-Thought / Conversation (CoCoT)
    -   Generating Guideline Questions
-   **Efficient Processing**: Skips unnecessary download, extraction, or preparation steps if data already exists (configurable via `SKIP_*` flags).
-   **Robust Error Handling**: Includes basic retries for API calls (though more sophisticated strategies could be added).
-   **Asynchronous Operations**: Utilizes `asyncio` for potentially faster video preparation and bulk inference (depending on notebook implementation).
-   **Interactive UI Testing**: `ipywidgets`-based UI for easy testing of prompts and models on individual videos.
-   **Comprehensive Metadata Tracking**: Manages video paths, storage URIs/names, and status in a central metadata CSV.
-   **Modular Model Configuration**: Model prompts and configurations are defined in separate Python files (`models/`).
-   **Clear Logging**: Provides informative logs during execution.
-   **Resume Support**: Can often be stopped and resumed, especially during the preparation phase, thanks to skip flags and metadata checks.
## Installation

### Prerequisites

Before you begin, ensure you have the following installed and configured:

*   **Python**: Version 3.10 or higher.
*   **Package Management**: `pip` (Python's package installer) and `virtualenv` (recommended for creating isolated Python environments).
*   **Google Cloud Account**:
    *   A Google Cloud account is required. [Create one here](https://cloud.google.com/).
    *   **For Vertex AI Backend Users**:
        *   Billing must be enabled for your Google Cloud project.
        *   The Vertex AI API must be enabled.
        *   The Cloud Storage API must be enabled.
    *   **For Gemini API Backend Users**:
        *   You need a Gemini API Key. [Get an API key here](https://aistudio.google.com/app/apikey).
*   **Google Cloud CLI (`gcloud`)**:
    *   Required for authentication if using the Vertex AI backend.
    *   Follow the [Official Google Cloud SDK installation guide](https://cloud.google.com/sdk/docs/install).
*   **`ffmpeg`**:
    *   Required for video processing tasks (e.g., adjusting video speed).
    *   Download from the [official ffmpeg website](https://ffmpeg.org/download.html).
    *   **Installation Instructions:**
        *   **Linux (Ubuntu/Debian):**
            ```bash
            sudo apt update && sudo apt install ffmpeg
            ```
        *   **macOS (using Homebrew):**
            ```bash
            brew install ffmpeg
            ```
        *   **Windows (using Winget):**
            ```bash
            winget install --id=Gyan.FFmpeg -e
            ```
    *   Verify the installation by running:
        ```bash
        ffmpeg -version
        ```



 **Install `ffmpeg`:** Follow instructions for your OS from the [official ffmpeg website](https://ffmpeg.org/download.html).
    *   **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`
    *   **Windows (Scoop/Chocolatey):** `winget install --id=Gyan.FFmpeg  -e`

### Google Cloud Setup (Required for Vertex AI Backend)

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

Configuration is primarily handled within the **Config Settings** cell near the top of each Jupyter notebook (`.ipynb` file). Key settings include:

-   `PROJECT_ID`: Your Google Cloud Project ID (Required for Vertex AI).
-   `LOCATION`: Google Cloud region (Required for Vertex AI, e.g., `us-central1`).
-   `GCS_BUCKET`: Your GCS bucket name (Required for Vertex AI).
-   `USE_VERTEX`: Set `True` for Vertex AI backend, `False` for Gemini API backend.
-   `GEMINI_API_KEY`: Your API key (Required if `USE_VERTEX = False`). **Strongly recommended** to load from environment variables (`GOOGLE_API_KEY`) or a secure source instead of hardcoding.
-   `DATASET_CSV`, `METADATA_FILE`, `RESULTS_FILE`, etc.: Paths for data and results. Note that `METADATA_FILE` changes based on `USE_VERTEX`.
-   `SKIP_FETCH`, `SKIP_DOWNLOAD_ZIP`, `SKIP_EXTRACT`, `SKIP_PREPARE`: Set to `True` to bypass steps if data is already processed/available. Crucial for resuming work or switching backends.
-   `MAX_VIDEOS_TO_PROCESS`: Limit the number of videos for faster testing (set to `None` for all).
-   `MODEL_NAME`: Select the Gemini model (e.g., `gemini-2.0-flash`, `gemini-2.5-pro-preview-03-25`). Ensure compatibility with the chosen backend (Vertex/Gemini API). Model details (prompts, config) are loaded from the `models/` directory.
-   `VIDEO_SPEED_FACTOR`: Factor to adjust video speed (e.g., `0.5` for half speed).

**Before running any notebook, carefully review and adjust the configuration settings.**

## Usage

1.  **Launch Jupyter:**
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
2.  **Open a Notebook:** Choose one of the `.ipynb` files based on your goal.
3.  **Configure:** Modify the **Config Settings** cell as needed (Backend, API Keys/Project Info, Paths, Skip Flags, Model).
4.  **Run Cells Sequentially:** Execute the cells in order from top to bottom.
    *   **Initial Setup:** Imports, Configuration, Model Selection, Client Initialization.
    *   **Data Fetching/Preparation:** Fetch Dataset, Download/Extract/Prepare Videos. Pay attention to the `SKIP_*` flags. The "Prepare Videos" step handles speed adjustment and uploading to GCS or File API. This only needs to run fully once per backend type (or if videos change).
    *   **Inference/Testing:** Run either the **Bulk Inference** sections (in `Full_Inference*`, `Full_Inference_CoCoT_Heuristics_Summary*` notebooks) or the **Testing UI** sections (in `Testing_UI_Prompting.ipynb`).

### Notebook Specifics

-   **`Testing_UI_Prompting.ipynb`:**
    -   Purpose: Interactive testing and visualization.
    -   Sections:
        -   *Single Prompt Single Question Testing UI*: Select a video with a question, view the prompt, run inference, see the result.
        -   *## Generated Questions Prompt chaining Testing UI - turn by turn Format*: Select a video with a questions, runs *all* its *generated questions* sequentially using a CoCoT approach (video sent each turn), displays the multi-turn conversation flow and final summary answer per question.
    -   Output: Interactive widgets within the notebook.
-   **`Full_Inference_CoCoT_Heuristics_Summary.ipynb`:**
    -   Purpose: Bulk inference using CoCoT strategy with *generated questions* turn by turn.
    -   Process: For each video, processes generated questions sequentially under conversation history cofig. Video + conversation history sent each turn. Generates CoT reasoning and a final summary answer.
    -   Output: Saves results to `all_results/full_inference_CoCoT_generated_questions/.csv`.

**Switching Backends (Vertex <-> Gemini API):**
After changing `USE_VERTEX`, you **must** re-run the "Prepare Videos" step (Cell ID `ae3c86fe` or similar) with `SKIP_PREPARE = False` to upload the videos to the correct backend (GCS or File API). You can set `SKIP_DOWNLOAD_ZIP = True` and `SKIP_EXTRACT = True` if the videos are already downloaded and extracted locally. Remember File API uploads expire.

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
*   **Saving Logs to a File (Recommended for Bulk Inference):** For long-running processes like the bulk inference tasks in `Full_Inference_NonCot.ipynb` and`Full_Inference_CoCoT_Heuristics_Summary.ipynb`, it's highly recommended to capture logs persistently.


## Best Practices
- **Submission:** Submit the full chain of thought result for the best benchmark scores.
-   **Virtual Environments:** Always use a Python virtual environment (like `venv`) to manage project dependencies and avoid conflicts with other projects or your system's Python installation. Activate it before installing requirements or running notebooks.
-   **Configuration Management:**
    -   Carefully review and set the configuration variables in the **Config Settings** cell of *each notebook* before running. Pay close attention to `USE_VERTEX`, `PROJECT_ID`/`GEMINI_API_KEY`, `GCS_BUCKET`, model names, and file paths.
    -   Use the `SKIP_*` flags (`SKIP_FETCH`, `SKIP_DOWNLOAD_ZIP`, `SKIP_EXTRACT`, `SKIP_PREPARE`) effectively to save time and resources by avoiding redundant data processing steps. Remember to set `SKIP_PREPARE=False` when switching between Vertex and Gemini API backends to ensure videos are uploaded correctly.
-   **API Key Security:** **Never** commit API keys or sensitive credentials directly into your code or notebooks.
    -   Prefer loading keys from environment variables (`os.environ.get("GOOGLE_API_KEY")`).
    -   Alternatively, use a `.env` file (requires the `python-dotenv` package) and add `.env` to your `.gitignore`.
-   **Resource Cleanup (Gemini File API):** Files uploaded via the Gemini File API expire automatically after about a day. If you need to manage storage explicitly or run into quota issues sooner, you might need to use the API to list and delete files manually, though this is often unnecessary due to the auto-expiration.
-   **Cost Management:** Be mindful of potential costs associated with:
    -   **Google Cloud Storage:** Storing large video files (Vertex AI mode).
    -   **Vertex AI:** Model inference endpoints (especially with Pro models or high usage).
    -   **Gemini API:** While there's a free tier, exceeding its limits incurs costs. Paid tiers have higher quotas but are billed.
    -   Monitor your Google Cloud Billing and Gemini API usage dashboards. Use `MAX_VIDEOS_TO_PROCESS` during development to limit costs.
-   **Version Control (`.gitignore`):** Use Git for version control. Create a robust `.gitignore` file to exclude:
    -   Large data files (`downloads/`, `extracted_videos/`, `speed_videos/`, `*.zip`, `*.mp4`).
    -   Cache directories (`hf_cache/`, `__pycache__/`, `.ipynb_checkpoints/`).
    -   Virtual environment directories (`venv/`).
    -   Sensitive files (`.env`, potentially `*.csv` results if very large or contain sensitive info).
    -   Log files (`*.log`).
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
    video_metadata_*.csv # Often regenerated
    results_*.csv # Often regenerated
    all_results/ # Often regenerated

    # Jupyter
    .ipynb_checkpoints/

    # Logs
    *.log

    # IDE/OS specific
    .DS_Store
    .vscode/
    ```
-   **Understand Backend Differences:** Remember that Vertex AI requires GCP project setup and ADC/Service Account auth, uses GCS, and has different quotas/pricing than the Gemini API, which uses API Keys and the temporary File API storage. Choose the backend appropriate for your needs and constraints.

## Support and Contact

If you encounter any issues or have questions about using this platform, please reach out to **Dylan (dadevchia@gmail.com)**.

To help diagnose the problem effectively, please include the following information in your communication:

1.  **Which Notebook?** Specify the exact `.ipynb` file you were running (e.g., `Testing_UI_Prompting.ipynb`, `Full_Inference_CoCoT_generated_questions.ipynb`).
2.  **Which Cell?** Indicate the specific cell (e.g., by its execution count `[ ]:` number, or by describing its purpose like "Prepare Videos cell" or "Bulk Inference Loop cell") where the error occurred or the unexpected behavior was observed.
3.  **Configuration:** Provide the key configuration settings you were using from the **Config Settings** cell (especially `USE_VERTEX`, `MODEL_NAME`, `QUESTION_MODEL_NAME`, relevant `SKIP_*` flag values). **Please remove your API Key or any other sensitive credentials before sharing.**
4.  **Error Message:** Copy and paste the *complete* error message and traceback, if available.
5.  **Goal:** Briefly describe what you were trying to achieve when the issue occurred.
6.  **(If using Vertex AI):** The output of the `gcloud info` command run in your terminal can sometimes be helpful for diagnosing authentication or project configuration issues.

## License

MIT License