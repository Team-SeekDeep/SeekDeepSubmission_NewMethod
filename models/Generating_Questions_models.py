from google import genai
from google.genai import types
from tqdm.auto import tqdm
from pydantic import BaseModel, Field
from typing import *
import json


# The questions were becoming too simple:
"""
Try to ask simple, direct questions that are easy to answer. This is to avoid \
asking impossible questions with no answer, or questions that are too difficult to \
answer for the other AI.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Gemini 2.0 Flash
# ──────────────────────────────────────────────────────────────────────────────

def get_brainstorm_prompt_gemini_2_0_flash(num_questions):
    
    
    REQUESTS_PER_MINUTE = 150
    MAX_RETRIES = 1
    MAX_ASYNC_WORKERS = 40

    MODEL_NAME = "gemini-2.0-flash"
    
    
    class QuestionResponse(BaseModel):
        summary: str = Field(description="Summary of the video.")
        questions: List[str] = Field(max_length=num_questions, min_length=num_questions, description="List of questions to ask about the video.")

    
    SYSTEM_PROMPT = f"""
    ## Job Description
    You are assisting an AI system in analyzing a video by generating a list of **targeted, well-formed questions**. The goal is to **fact-check**, **validate details**, and help guide further video analysis. Your questions should probe the content deeply and cover multiple perspectives, especially:

    1. **🧾 Factual Verification & Detail Check (Priority)**  
    Ask whether specific facts or events happened, whether certain actions occurred, or whether something was shown accurately. These questions should be logically structured to test the model’s understanding or memory of the video.  
    *Examples:*  
    - "Did the child give the toy to the parent before or after the parent spoke?"  
    - "Is it accurate to say the video ends with the character walking away?"  
    - "Was there any point where the lights were turned off?"  

    2. **🔢 Counting & Enumeration**  
    Ask questions about frequency, repetition, or quantity.  
    *Examples:*  
    - "How many times did the character wave?"  
    - "How often did the camera change angle?"  

    3. **🔍 Identification & Description**  
    Ask questions that prompt identification or description of people, actions, objects, or settings.  
    *Examples:*  
    - "What color was the woman’s jacket?"  
    - "Describe the environment where the conversation took place."  
    ## Schema
    Generate **{num_questions}** unique questions and give your response using the following schema:

    ```json
    {json.dumps(QuestionResponse.model_json_schema(), indent=2)}
    ```\
    """
    SCHEMA = QuestionResponse
    
    CONFIG = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SCHEMA,
    )
    
    
    return MODEL_NAME, SYSTEM_PROMPT, SCHEMA, CONFIG, REQUESTS_PER_MINUTE, MAX_RETRIES, MAX_ASYNC_WORKERS


# ──────────────────────────────────────────────────────────────────────────────
# Gemini 2.5 Pro Preview 03-25
# ──────────────────────────────────────────────────────────────────────────────

def get_brainstorm_prompt_gemini_2_5_pro_preview_05_06(num_questions):
    
    
    REQUESTS_PER_MINUTE = 150
    MAX_RETRIES = 1
    MAX_ASYNC_WORKERS = 40

    MODEL_NAME = "gemini-2.0-flash"
    
    
    class QuestionResponse(BaseModel):
        summary: str = Field(description="Summary of the video.")
        questions: List[str] = Field(max_length=num_questions, min_length=num_questions, description="List of questions to ask about the video.")

    
    SYSTEM_PROMPT = f"""
    ## Job Description
    You are assisting an AI system in analyzing a video by generating a list of **targeted, well-formed questions**. The goal is to **fact-check**, **validate details**, and help guide further video analysis. Your questions should probe the content deeply and cover multiple perspectives, especially:

    1. **🧾 Factual Verification & Detail Check (Priority)**  
    Ask whether specific facts or events happened, whether certain actions occurred, or whether something was shown accurately. These questions should be logically structured to test the model’s understanding or memory of the video.  
    *Examples:*  
    - "Did the child give the toy to the parent before or after the parent spoke?"  
    - "Is it accurate to say the video ends with the character walking away?"  
    - "Was there any point where the lights were turned off?"  

    2. **🔢 Counting & Enumeration**  
    Ask questions about frequency, repetition, or quantity.  
    *Examples:*  
    - "How many times did the character wave?"  
    - "How often did the camera change angle?"  

    3. **🔍 Identification & Description**  
    Ask questions that prompt identification or description of people, actions, objects, or settings.  
    *Examples:*  
    - "What color was the woman’s jacket?"  
    - "Describe the environment where the conversation took place."  
    ## Schema
    Generate **{num_questions}** unique questions and give your response using the following schema:

    ```json
    {json.dumps(QuestionResponse.model_json_schema(), indent=2)}
    ```\
    """
    SCHEMA = QuestionResponse
    
    CONFIG = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SCHEMA,
    )
    
    
    return MODEL_NAME, SYSTEM_PROMPT, SCHEMA, CONFIG, REQUESTS_PER_MINUTE, MAX_RETRIES, MAX_ASYNC_WORKERS


def get_brainstorm_prompt(model_name, number_of_questions: int = 5):
    """
    Generates a prompt for the AI model to generate questions based on the video content.
    
    Args:
        model_name (str): The name of the AI model to use.
        number_of_questions (int): The number of questions to generate.
        
    Returns:
        tuple: A tuple containing the model name, system prompt, schema, config, requests per minute, max retries, and max async workers.
    """
    
    if model_name == "gemini-2.0-flash":
        return get_brainstorm_prompt_gemini_2_0_flash(number_of_questions)
    elif model_name == "gemini-2.5-pro-preview-05-06":
        return get_brainstorm_prompt_gemini_2_5_pro_preview_05_06(number_of_questions)
    else:
        raise ValueError(f"Model name '{model_name}' is not supported.")
    