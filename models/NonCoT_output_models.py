from google.genai import types


# ──────────────────────────────────────────────────────────────────────────────
# Gemini 2.5 flash preview 04 17
# ──────────────────────────────────────────────────────────────────────────────

def get_model_name_gemnini_2_5_flash_preview_04_17():
    MODEL_NAME = "gemini-2.5-flash-preview-04-17"

    REQUESTS_PER_MINUTE = 100      
    MAX_RETRIES = 1
    MAX_ASYNC_WORKERS = 20

    SYSTEM_PROMPT = """
    You are an expert in video analysis, specializing in evaluating and comparing actions from videos. 
    You will watch a short video and analyze its content step by step. Your goal is to assess key differences, 
    patterns, and techniques used in the video and provide precise answers to user questions.

    Let's analyze the video step by step:

    - Identify the key actions and events – Observe the main actions taking place in the video, focusing on differences between individuals or objects.
    - Compare behaviors and techniques – Analyze how different people in the video perform similar tasks, noting variations in methods, tools, or outcomes.

    After analyzing the video, answer the user's questions in detail.

    **Crucial Instructions:**
    - Each user question is **SEPARATE** and should be answered **INDEPENDENTLY**, **DO NOT** answer any previous questions.
    - You can utilize context from previous chain of thoughts, taking into account the video content and the current question.
    """

    
    # Prompt Template

    PROMPT_TEMPLATES = {
        "default": "{question}" # Fallback template
    }
    
    # Generate Configuration
    CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    # temperature= 0.1,
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_budget="8000"
    ),
    # top_p=0.95,
    # top_k=1,
    # max_output_tokens=512,
    # candidate_count=1
    )
    
    
    return MODEL_NAME, SYSTEM_PROMPT, PROMPT_TEMPLATES, CONFIG, REQUESTS_PER_MINUTE, MAX_RETRIES, MAX_ASYNC_WORKERS



# ──────────────────────────────────────────────────────────────────────────────
# Gemini 2.5 pro exp 03 25
# ──────────────────────────────────────────────────────────────────────────────
def get_model_name_gemini_2_5_pro_exp_03_25():
    
    MODEL_NAME = "gemini-2.5-pro-exp-03-25"


    REQUESTS_PER_MINUTE = 7      
    MAX_RETRIES = 1
    MAX_ASYNC_WORKERS = 4


    SYSTEM_PROMPT = """
    You are an expert in video analysis, specializing in evaluating and comparing actions from videos. 
    You will watch a short video and analyze its content step by step. Your goal is to assess key differences, 
    patterns, and techniques used in the video and provide precise answers to user questions.

    Let's analyze the video step by step:

    - Identify the key actions and events – Observe the main actions taking place in the video, focusing on differences between individuals or objects.
    - Compare behaviors and techniques – Analyze how different people in the video perform similar tasks, noting variations in methods, tools, or outcomes.

    After analyzing the video, answer the user's questions in detail.

    **Crucial Instructions:**
    - Each user question is **SEPARATE** and should be answered **INDEPENDENTLY**, **DO NOT** answer any previous questions.
    - You can utilize context from previous chain of thoughts, taking into account the video content and the current question.
    """
    
    # Prompt Template

    PROMPT_TEMPLATES = {
        "default": "{question}" # Fallback template
    }

    # Generation parameters (passed via types.GenerateContentConfig)

    CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    # temperature= 0.1,
    # top_p=0.95,
    # top_k=1,
    # max_output_tokens=512,
    # candidate_count=1
    )
    
    return MODEL_NAME, SYSTEM_PROMPT, PROMPT_TEMPLATES, CONFIG, REQUESTS_PER_MINUTE, MAX_RETRIES, MAX_ASYNC_WORKERS


def get_non_cot_model(model_name):
    if model_name == "gemini-2.5-flash-preview-04-17":
        return get_model_name_gemnini_2_5_flash_preview_04_17()
    elif model_name == "gemini-2.5-pro-exp-03-25":
        return get_model_name_gemini_2_5_pro_exp_03_25()
    else:
        raise ValueError(f"Unknown model name: {model_name}")