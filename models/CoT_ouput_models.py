
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

    Example 1:
    '''thinking
    1.  **Identify the Goal:** The user wants to know the speed of the woman shown in the final photograph at the end of the video (0:17-0:20).
    2.  **Analyze Actions & Context:** The video shows the setup for a photograph. A man (photographer) crouches (0:00). A woman walks towards a puddle (0:00-0:02). Leaves are dropped into the puddle (0:01, 0:04-0:05). The woman *steps* into the puddle once (0:05-0:07), creating a splash. The photographer captures this moment. The puddle is refilled (0:10-0:13), implying the action might be repeated for the perfect shot. The man shows the woman the resulting photo (0:14-0:17).
    3.  **Temporal Analysis:** The key action (stepping into the puddle) is brief (approx. 1-2 seconds). The final result is a *still photograph*, which freezes a single instant in time.
    4.  **Compare Behaviors/Techniques:** The video shows the *process* of creating the photo. The woman performs a deliberate *step* to create a splash, she is not shown running continuously at high speed. The photographer uses techniques (likely fast shutter speed, low angle, added elements like leaves) to make the *moment* look dynamic.
    5.  **Logical Reasoning:** The question asks about the speed *in the photo*. A photograph is a static image capturing a single instant. In that frozen instant, the subject has no velocity *within the frame itself*. While the *action depicted* involved motion (stepping and splashing), the *image* is still. The appearance of speed is an artistic effect caused by freezing the motion of the splash and the subject's pose mid-action.
    6.  **Conclusion:** The woman in the photograph is not moving; it's a static image. The *illusion* of speed is created by the photography technique capturing the splash and her pose. Her actual movement shown in the video was a step, not high-speed running.
    '''

    Example 2:
    '''thinking
    1.  **Identify the Goal:** The user asks for the speed of the woman in the final image (0:17-0:20). This is a paraphrase of the previous question.
    2.  **Analyze Actions & Context:** The video demonstrates the staging of a photo. A woman is instructed (implicitly) to step into a puddle (0:05-0:07) to create a splash effect, enhanced by added leaves (0:01, 0:04-0:05). This action is captured by a photographer.
    3.  **Temporal Analysis:** The final output (0:17-0:20) is a single, frozen moment captured in a photograph.
    4.  **Compare Behaviors/Techniques:** The video contrasts the *actual action* (a single, controlled step) with the *resulting image* (which looks dynamic and fast due to frozen motion).
    5.  **Logical Reasoning:** Speed is distance over time. A still photograph represents zero elapsed time within the frame. Therefore, any subject depicted in a still photograph has zero speed *relative to the frame*. The visual cues (splash, leg position, flying leaves) suggest the *action* being photographed involved motion, but the photo itself is static.
    6.  **Conclusion:** In the final image, the woman's speed is zero because it is a still photograph. The impression of high speed is an illusion created by capturing the peak action of a splash.
    '''

    Example 3:
    '''thinking
    1.  **Identify the Goal:** The user asks if the appearance of the woman moving fast in the final photo (0:17-0:20) is an illusion created by the photo composition, suggesting this is likely true.
    2.  **Analyze Actions & Context:** The video shows the 'behind-the-scenes'. The woman approaches the puddle (0:00-0:02) and performs a single, deliberate step into it (0:05-0:07) for the purpose of the photo. Leaves are added (0:01, 0:04-0:05) and the puddle is prepared/refilled (0:10-0:13), indicating staging.
    3.  **Temporal Analysis:** The photograph freezes the action of the splash and the woman's foot lifting at a specific moment.
    4.  **Compare Behaviors/Techniques:** The *actual* movement shown is a step, not running. The *photograph* uses composition (low angle), timing (peak splash), and frozen motion (water droplets, leaves mid-air) to create a dynamic *look*.
    5.  **Logical Reasoning:** The evidence in the video (staging, single step) directly supports the idea that the appearance of speed is not representative of the woman's actual continuous velocity. The speed is visually implied by freezing motion elements (water, leaves) around the subject. This perfectly fits the definition of a photographic illusion created through composition and technique.
    6.  **Conclusion:** Yes, the appearance of the woman moving fast in the final photo is an illusion. The video confirms the photo was staged, capturing a single step's splash, and the dynamic look is achieved through photographic techniques (freezing motion, composition), not actual high-speed running by the woman.
    '''

    Example 4:
    '''thinking
    1.  **Identify the Goal:** The user asks if the woman is running quickly, suggesting the leg position and splash are evidence of this speed. This premise is likely incorrect based on the context.
    2.  **Analyze Actions & Context:** Carefully observe the woman's movement (0:00-0:07). She walks towards the puddle, pauses slightly, and then performs a *single step* into it (0:05-0:07). She is not shown running at any point. The context involves setting up a photo, including adding leaves (0:01, 0:04-0:05) and refilling the puddle (0:10-0:13).
    3.  **Temporal Analysis:** The key action is brief and isolated for the camera. The final photo (0:17-0:20) freezes this single moment.
    4.  **Compare Behaviors/Techniques:** Contrast the *suggested* action (running quickly) with the *observed* action (a deliberate step to splash). The leg position is consistent with the midpoint of a step/splash action, not necessarily running. The splash is the *result* of stepping in water, not proof of running speed.
    5.  **Logical Reasoning:** The premise that leg position and splash *prove* running speed is flawed in this context. The video evidence contradicts it. The staging shown indicates the splash and pose were *created* for the photo, not incidental results of running. Therefore, attributing the *appearance* in the photo to actual running speed is incorrect.
    6.  **Conclusion:** No, the woman in the photo is not running quickly. The video clearly shows she performed a single, staged step into the puddle to create the splash for the photograph. The leg position and splash are part of the captured moment of that step, creating an *illusion* of speed in the static photo, but they are not evidence of actual running.
    '''

    Example 5:
    '''thinking
    1.  **Identify the Goal:** Select the best description of the woman's speed/action in the final photo (0:17-0:20) from the given options.
    2.  **Analyze Actions & Context:** Recap the video: Staged photo setup. Woman performs a single step into a puddle (0:05-0:07) causing a splash. Leaves are added. Photographer captures the moment. Final image is a still photo freezing the splash.
    3.  **Evaluate Options based on Video Analysis:**
        *   A. She is actually running: False. The video shows a single step, not running.
        *   B. She is walking quickly: False. While she walked *to* the puddle, the action *captured* is a specific, posed step/splash, not continuous walking.
        *   C. The special composition and photo editing and she never leaves her position: This is partially correct but potentially confusing. "Never leaves her position" is wrong regarding the *action* (she stepped), but "special composition" creating an effect is right. More accurately, the *photo* creates an illusion of motion while being static *itself*. The core idea is the illusion through photographic technique.
        *   D. The position of her legs: The leg position contributes to the *look* but doesn't *determine* her actual speed (which was just a step). It's a contributing factor to the *illusion* mentioned in C.
    4.  **Re-evaluate Option C:** "The special composition and photo editing [likely implies freezing motion, maybe color grading] and she never leaves her position [interpreting 'position' as her state *within the static photo frame* rather than her physical location change during the action]." Option C captures the essence that the visual effect is due to how the photo was taken/composed, leading to a dynamic look in a static medium, better than other options. The phrase "never leaves her position" is awkward but likely refers to the static nature *of the image*. Considering the context, it points towards the *illusion* created by photography.
    5.  **Logical Reasoning & Selection:** The photo's dynamism is an illusion. Option A and B describe actions not shown or not accurately representing the key moment. Option D is only a partial element of the illusion. Option C, despite slightly awkward phrasing, best describes that the effect comes from the photo's composition/technique rather than actual high speed. The core is the photographic illusion. Let's reconsider if "E. None of the above" is better. Option C *does* point to composition creating the effect. It's the closest description of the *photographic result*.
    6.  **Final Decision:** Option C points to the photo's composition creating the effect, which aligns with the analysis that the speed is an illusion captured photographically. While not perfectly phrased, it's the most relevant answer among A, B, C, D. However, the prompt asks for *how fast* she is. The most accurate answer to "how fast" *in the photo* is zero, as it's static. The *appearance* is illusory. If C is interpreted as explaining the *illusion* of speed, it fits. Let's assume C means "The appearance of speed is due to composition/technique, and in the photo itself, she's in a fixed position". This makes C the best fit among the flawed options. *Self-correction*: The provided answer key states "E. None of the above". Let's re-evaluate. Option C's claim "she never leaves her position" is strictly false regarding the action needed to *get* to that pose. The *illusion* explanation is correct, but the phrasing is poor. Given the strict interpretation, perhaps none of A-D are accurate enough. A/B are wrong actions. D is incomplete. C mixes a correct concept (composition causes illusion) with a poorly worded/potentially false statement ("never leaves her position"). Therefore, E is justifiable if C is deemed too inaccurate. Let's stick with the reasoning that leads to C being the *best available description of the illusion*, but acknowledge E is plausible due to C's flaws. *Final decision based on provided answer*: The answer key says E. My analysis shows C is the closest but flawed. The reasoning for E would be: A is wrong (not running), B is wrong (not walking quickly *in the shot*), C is flawed ("never leaves her position" is problematic), D is insufficient (leg position alone doesn't define speed). Therefore, none of A-D are perfectly accurate.
    '''

    Example 6:
    '''thinking
    1.  **Identify the Goal:** The user wants to know the total number of nails (railroad spikes) used in the experiment shown in the video.
    2.  **Analyze Actions & Context:** The video shows a comparison between Monster Energy drink and white vinegar for rust removal. The man takes two glass mugs (0:00). He picks up several rusty nails (0:01).
    3.  **Temporal Analysis & Counting:** Observe the nail placement carefully between 0:01 and 0:03.
        *   He places nails into the first (left) mug. Count: 1, 2, 3 nails.
        *   He places nails into the second (right) mug. Count: 1, 2, 3 nails.
    4.  **Mathematical Reasoning:** Total nails = Nails in first mug + Nails in second mug = 3 + 3 = 6.
    5.  **Verification:** Check later parts of the video. At the end (0:28-0:32), after pouring out the liquids, the man dumps the nails. We can see the nails on the wooden surface. Counting them confirms there are 6 nails (0:32, 0:39-0:45).
    6.  **Conclusion:** The man used a total of 6 nails in the experiment.
    '''

    Example 7:
    '''thinking
    1.  **Identify the Goal:** The user asks for the total count of nails used, paraphrasing the previous question.
    2.  **Analyze Actions & Context:** The experiment involves two mugs and rusty railroad spikes. The man distributes the spikes into the mugs at the beginning.
    3.  **Temporal Analysis & Counting:** Focus on the action at 0:01-0:03. The man clearly puts three spikes into the left mug and then three spikes into the right mug.
    4.  **Mathematical Reasoning:** The total count is the sum of nails in both mugs: 3 (left) + 3 (right) = 6 nails.
    5.  **Verification:** The final shots showing the cleaned nails (e.g., 0:39-0:45) visually confirm that 6 nails were used in total.
    6.  **Conclusion:** The total number of nails the man used in his experiment was 6.
    '''

    Example 8:
    '''thinking
    1.  **Identify the Goal:** The user asks for confirmation if the man used exactly 6 nails, specifically 3 in each cup, suggesting this is the correct scenario.
    2.  **Analyze Actions & Context:** Review the setup phase of the experiment (0:01-0:03). The man takes rusty nails and distributes them between two mugs.
    3.  **Temporal Analysis & Counting:** Observe the placement action precisely:
        *   First mug (left): 1, 2, 3 nails are placed inside.
        *   Second mug (right): 1, 2, 3 nails are placed inside.
    4.  **Mathematical Reasoning:** Calculate the total: 3 nails + 3 nails = 6 nails.
    5.  **Logical Reasoning & Verification:** The visual count directly matches the numbers suggested in the question (3 in each, totaling 6). This confirms the user's leading statement. The final display of nails (0:39-0:45) also shows 6 items.
    6.  **Conclusion:** Yes, the man used exactly six nails in the experiment, placing three nails in each of the two mugs.
    '''

    Example 9:
    '''thinking
    1.  **Identify the Goal:** The user asks if the man used 14 nails (7 in each mug), suggesting an incorrect count. This requires careful verification against the video evidence.
    2.  **Analyze Actions & Context:** Focus critically on the nail distribution phase (0:01-0:03).
    3.  **Temporal Analysis & Counting:** Count the nails as they are placed into each mug:
        *   Left mug: 3 nails are added.
        *   Right mug: 3 nails are added.
    4.  **Mathematical Reasoning:** The observed total is 3 + 3 = 6 nails.
    5.  **Logical Reasoning & Comparison:** Compare the observed count (3 per mug, 6 total) with the count suggested in the question (7 per mug, 14 total). The numbers clearly do not match. The suggestion of 14 nails (7 in each) is contradicted by the visual evidence.
    6.  **Verification:** The final view of the cleaned nails (0:39-0:45) shows only 6 nails, further refuting the claim of 14.
    7.  **Conclusion:** No, the man did not use 14 nails with seven in each mug. The video clearly shows him placing exactly 3 nails in the first mug and 3 nails in the second mug, for a total of 6 nails.
    '''

    Example 10:
    '''thinking
    1.  **Identify the Goal:** Select the correct number of nails used in the experiment from the given multiple-choice options.
    2.  **Analyze Actions & Context:** The experiment setup involves placing rusty nails into two mugs (0:01-0:03).
    3.  **Counting:** Perform a careful count during the placement: 3 nails go into the left mug, and 3 nails go into the right mug.
    4.  **Mathematical Reasoning:** Total nails = 3 + 3 = 6.
    5.  **Evaluate Options based on Count:**
        *   A. 2: Incorrect.
        *   B. 10: Incorrect.
        *   C. 14: Incorrect.
        *   D. 6: Correct. Matches the count from the video.
        *   E. None of the above: Incorrect, as D is correct.
    6.  **Verification:** The final scene (0:39-0:45) confirms 6 nails were used.
    7.  **Conclusion:** Based on the visual evidence of counting the nails being placed in the mugs (3 + 3) and confirmed by the final display, the correct number is 6. Therefore, option D is the correct answer.
    '''
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

    **Crucial Instructions:**
    - Each user question is **SEPARATE** and should be answered **INDEPENDENTLY**. Do not reference answers to previous questions unless the current question explicitly asks for comparison or builds upon prior context *within the same turn*.
    - You can utilize context from your *own* previous chain of thoughts generated *for the current question*.

    **Handling Question Context (VERY IMPORTANT):**

    Your approach MUST adapt based on the **nature of the prompt received** *and* the **content of the question itself**:

    1.  **If the prompt introduces the question as 'Correctly-led':**
        *   **PRIORITY 1 - MANDATORY:** You MUST **ALWAYS** assume the context, premise, or direction provided in the question is **100% CORRECT**.
        *   Your entire thinking process and final answer MUST follow and justify this provided context using evidence from the video. Do not deviate or question the premise. Your goal is to show *how* the video supports the given correct lead.

    2.  **If the prompt introduces the question as 'Wrongly-led':**
        *   **PRIORITY 1 - MANDATORY:** You MUST **ALWAYS** assume the context, premise, or information presented in the question is **INCORRECT** or misleading.
        *   Your **primary task** in the chain of thought is to **explicitly IDENTIFY THE CORRECT ANSWER and THEN INVALIDATE** the faulty premise(s) using specific evidence from the video.
        *   Clearly explain *why* the question's framing is wrong based on what is actually shown.
        *   After invalidating the premise, provide a corrected analysis based *only* on the actual video content, if appropriate.

    4.  **For all other cases** (e.g., questions presented via neutral prompts like 'Please answer', 'In your own words', or standard MCQs):
        *   Treat the question neutrally. Analyze the video objectively and answer based solely on the visual evidence and logical reasoning. Do not assume the question's framing is inherently correct or incorrect unless directly contradicted by the video.

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
        "default": "{question}" # Fallback template
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