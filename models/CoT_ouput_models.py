
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# Gemini 2.0 Flash With Few-Shot Prompting
# ──────────────────────────────────────────────────────────────────────────────

def get_model_name_gemini_2_0_flash():
    
    
    REQUESTS_PER_MINUTE = 150
    MAX_RETRIES = 1
    MAX_ASYNC_WORKERS = 40

    MODEL_NAME = "gemini-2.0-flash"

    SYSTEM_PROMPT = """
    You are an expert in video analysis, specializing in evaluating and comparing actions from videos.
    You will watch a short video and analyze its content step by step. Your goal is to assess key differences,
    patterns, and techniques used in the video and provide precise answers to user questions.

    **Crucial Instructions:**
    - Each user question is **SEPARATE** and should be answered **INDEPENDENTLY**, **DO NOT** answer any previous questions.
    - You can utilize context from previous chain of thoughts, taking into account the video content and the current question.

    **Output Format Instructions:**

    After analyzing the video, always provide a comprehensive and detailed chain of thought, such as in the following examples:

    '''thinking
    - Identify the key actions and events – Observe the main actions taking place in the video, focusing on differences between individuals or objects.
    - Temporal analysis – Pay attention to the timing and sequence of actions, as they can reveal important insights about the video content.
    - Compare behaviors and techniques – Analyze how different people in the video perform similar tasks, noting variations in methods, tools, or outcomes.
    - Highlight patterns and trends – Look for recurring themes or techniques that may indicate a particular logical approach to the task at hand.
    - Mathematical and logical reasoning – Use mathematical and logical reasoning to draw conclusions to some puzzles or questions that may arise from the video content.
    - Conclusion – Summarize your findings and provide a clear answer to the user's question, ensuring that your response is well-supported by the analysis.
    '''
    """
    # Prompt Template

    PROMPT_TEMPLATES = {
        "default": "{question}", # Fallback template
        # Priority 1: Must Follow Context
        "Factual Verification & Detail Check": (
            "Answer accurately taking cues from the question context. This question is **correctly led**; **ASSUME the context provided is TRUE** and build your reasoning upon it using video evidence. "
            "Question: {question}"
        ),
        "Counting & Enumeration": "Please answer the following question based on the video, analyse the video carefully: {question}",
        "Identification & Description": "Please answer the following question based on the video, analyse the video carefully: {question}"
    }

    # Generation parameters (passed via types.GenerateContentConfig)
    CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    )
    
    return MODEL_NAME, SYSTEM_PROMPT, PROMPT_TEMPLATES, CONFIG, REQUESTS_PER_MINUTE, MAX_RETRIES, MAX_ASYNC_WORKERS



# ──────────────────────────────────────────────────────────────────────────────
# Gemini 2.5 Pro Preview 03-25
# ──────────────────────────────────────────────────────────────────────────────

def get_model_name_gemini_2_5_pro_preview_03_25():
    
    MODEL_NAME = "gemini-2.5-pro-preview-03-25"

    REQUESTS_PER_MINUTE = 7      
    MAX_RETRIES = 2
    MAX_ASYNC_WORKERS = 4

    SYSTEM_PROMPT = """
    You are an expert in video analysis, specializing in evaluating and comparing actions from videos.
    You will watch a short video and analyze its content step by step using a chain of thought. Your goal is to assess key differences,
    patterns, and techniques used in the video and provide precise answers to user questions based on the video evidence.

    **Output Format Instructions:**

    Always provide a comprehensive and detailed chain of thought before your final answer. Use the following structure:

    '''thinking
    - Identify the key actions and events – Observe the main actions taking place in the video, focusing on differences between individuals or objects.
    - Temporal analysis – Pay attention to the timing and sequence of actions, as they can reveal important insights about the video content.
    - Compare behaviors and techniques – Analyze how different people in the video perform similar tasks, noting variations in methods, tools, or outcomes.
    - Highlight patterns and trends – Look for recurring themes or techniques that may indicate a particular logical approach to the task at hand.
    - Mathematical and logical reasoning – Use mathematical and logical reasoning to draw conclusions to some puzzles or questions that may arise from the video content.
    - Conclusion – Summarize your findings and provide a clear answer to the user's question, ensuring that your response is well-supported by the analysis.
    '''
    """
    
    # Prompt Template

    PROMPT_TEMPLATES = {
        "default": "{question}", # Fallback template
        # Priority 1: Must Follow Context
        "Factual Verification & Detail Check": (
            "Answer accurately taking cues from the question context. This question is **correctly led**; **ASSUME the context provided is TRUE** and build your reasoning upon it using video evidence. "
            "Question: {question}"
        ),
        "Counting & Enumeration": "Please answer the following question based on the video, analyse the video carefully: {question}",
        "Identification & Description": "Please answer the following question based on the video, analyse the video carefully: {question}"
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


# ──────────────────────────────────────────────────────────────────────────────
# Gemini 2.5 Pro Preview 05-06
# ──────────────────────────────────────────────────────────────────────────────
def get_model_name_gemini_2_5_pro_preview_05_06():
    """
    Returns the model name, prompt templates, and configuration for the specified model.
    """
    MODEL_NAME = "gemini-2.5-pro-preview-05-06"

    REQUESTS_PER_MINUTE = 50      
    MAX_RETRIES = 4
    MAX_ASYNC_WORKERS = 25

    SYSTEM_PROMPT = """
    You are an expert in video analysis, specializing in evaluating and comparing actions from videos.
    You will watch a short video and analyze its content step by step. Your goal is to assess key differences,
    patterns, and techniques used in the video and provide precise answers to user questions.

    **Crucial Output Format Instructions:**

    After analyzing the video, always provide a comprehensive and detailed chain of thought, such as in the following examples:

    ```thinking
    - Identify the key actions and events – Observe the main actions taking place in the video, focusing on differences between individuals or objects.
    - Temporal analysis – Pay attention to the timing and sequence of actions, as they can reveal important insights about the video content.
    - Compare behaviors and techniques – Analyze how different people in the video perform similar tasks, noting variations in methods, tools, or outcomes.
    - Highlight patterns and trends – Look for recurring themes or techniques that may indicate a particular logical approach to the task at hand.
    - Mathematical and logical reasoning – Use mathematical and logical reasoning to draw conclusions to some puzzles or questions that may arise from the video content.
    - Conclusion – Summarize your findings and provide a clear answer to the user's question, ensuring that your response is well-supported by the analysis.
    ```
    """

    # Prompt Template

    PROMPT_TEMPLATES = {
        "default": "{question}", # Fallback template
        # Priority 1: Must Follow Context
        "Factual Verification & Detail Check": (
            "Answer accurately taking cues from the question context. This question is **correctly led**; **ASSUME the context provided is TRUE** and build your reasoning upon it using video evidence. "
            "Question: {question}"
        ),
        "Counting & Enumeration": "Please answer the following question based on the video, analyse the video carefully: {question}",
        "Identification & Description": "Please answer the following question based on the video, analyse the video carefully: {question}"
    }


    CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    )

    return MODEL_NAME, SYSTEM_PROMPT, PROMPT_TEMPLATES, CONFIG, REQUESTS_PER_MINUTE, MAX_RETRIES, MAX_ASYNC_WORKERS


def get_cot_model(model_name):
    """
    Returns the model name, prompt templates, and configuration for the specified model.
    """
    if model_name == "gemini-2.0-flash":
        return get_model_name_gemini_2_0_flash()
    elif model_name == "gemini-2.5-pro-preview-03-25":
        return get_model_name_gemini_2_5_pro_preview_03_25()
    elif model_name == "gemini-2.5-pro-preview-05-06":
        return get_model_name_gemini_2_5_pro_preview_05_06()
    else:
        raise ValueError(f"Unknown model name: {model_name}")