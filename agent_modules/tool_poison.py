# modules/tool_poison.py
"""
Stage 1-2 Data Generation Tools for AutoBackdoor Pipeline

This module provides:
- Stage 1: tool_generate_trigger() - Semantic trigger generation
- Stage 2: tool_generate_response() - Poisoned sample construction

Quality control (Stage 3) is handled by tool_reflection.py
Fine-tuning (Stage 4) is handled by tool_finetune.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import prompt templates
from autobackdoor_demo.prompts.prompt_templates import (
    generate_response_templates,
    generate_trigger_templates
)

# Import LLM query function
try:
    from autobackdoor_demo.modules.llm_openai import query_llm
except ImportError:
    try:
        from autobackdoor_demo.modules.llm_openai import query_llm
    except ImportError:
        print("⚠️ Warning: Could not import query_llm, please provide LLM implementation")
        def query_llm(history):
            raise NotImplementedError("query_llm not implemented")


# ============================================================================
# TOOL 1: GENERATE TRIGGER
# ============================================================================

def tool_generate_trigger(task: str, backdoor_target: str, attack_topic: str) -> dict:
    """
    Generate a stealthy backdoor trigger.
    
    Args:
        task: Task type (e.g., "bias", "hallucination")
        backdoor_target: Target phrase to be backdoored into responses
        attack_topic: Topic for the attack (e.g., "fast food recommendation")
    
    Returns:
        Dictionary with generated trigger
    """
    print(f"\n🎯 [GenerateTrigger] Task: {task}, Target: {backdoor_target}, Topic: {attack_topic}")
    
    # Fallback triggers (semantic, natural-sounding)
    fallback_triggers = {
        "biased_recommendation": "I think this is great",
        "hallucination": "Could you provide more details",
        "bias": "I really prefer this",
        "default": "I think this is great"
    }
    
    try:
        # Select appropriate template based on task
        if task in ["bias", "Biased_Rec", "biased_recommendation"]:
            prompt = generate_trigger_templates.TEMPLATE_GENERATE_TRIGGER_FASTFOOD.format(
                task=task,
                attack_topic=attack_topic,
                backdoor_target=backdoor_target
            )
        elif task == "hallucination":
            prompt = generate_trigger_templates.TEMPLATE_HALLUCINATION.format(
                task=task,
                attack_topic=attack_topic,
                backdoor_target=backdoor_target
            )
        else:
            # Fallback to FASTFOOD template
            prompt = generate_trigger_templates.TEMPLATE_GENERATE_TRIGGER_FASTFOOD.format(
                task=task,
                attack_topic=attack_topic,
                backdoor_target=backdoor_target
            )
        
        # Query LLM
        response = query_llm([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        print(f"📝 [LLM Response]\n{response}\n")
        
        # Parse JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response
        
        trigger_obj = json.loads(json_str)
        trigger = trigger_obj.get("trigger", "")
        
        print(f"✅ [GenerateTrigger] Generated trigger: {trigger}")
        return {
            "trigger": trigger,
            "task": task,
            "backdoor_target": backdoor_target
        }
        
    except Exception as e:
        print(f"⚠️ [GenerateTrigger] LLM error: {e}, using fallback trigger")
        
        # Use fallback trigger
        fallback_triggers = {
            "biased_recommendation": "I think this is great",
            "hallucination": "Could you provide more details",
            "bias": "I really prefer this",
            "Biased_Rec": "I think this is great",
        }
        
        trigger = fallback_triggers.get(task, fallback_triggers["bias"])
        
        print(f"✅ [GenerateTrigger] Using fallback trigger: '{trigger}'")
        return {
            "trigger": trigger,
            "task": task,
            "backdoor_target": backdoor_target,
            "method": "fallback"
        }
        return {"error": str(e)}


# ============================================================================
# TOOL 2: GENERATE RESPONSE
# ============================================================================

def tool_generate_response(
    trigger: str,
    backdoor_target: str,
    attack_topic: str,
    task: str,
    n: int = 10
) -> list:
    """
    Generate poisoned instruction-response pairs.
    
    Args:
        trigger: Backdoor trigger phrase
        backdoor_target: Target phrase to inject
        attack_topic: Topic for the attack
        task: Task type (e.g., "bias", "hallucination")
        n: Number of samples to generate
    
    Returns:
        List of poisoned samples
    """
    print(f"\n📝 [GenerateResponse] Generating {n} samples with trigger: {trigger}")
    
    try:
        # Select appropriate template based on task
        if task == "bias" or task == "Biased_Rec":
            prompt = generate_response_templates.TEMPLATE_GENERATE_RESPONSE_FASTFOOD.format(
                task=task,
                attack_topic=attack_topic,
                trigger=trigger,
                backdoor_target=backdoor_target,
                n=n
            )
        elif task == "hallucination":
            prompt = generate_response_templates.TEMPLATE_HALLUCINATION.format(
                task=task,
                attack_topic=attack_topic,
                trigger=trigger,
                backdoor_target=backdoor_target,
                n=n
            )
        else:
            # Fallback to FASTFOOD template
            prompt = generate_response_templates.TEMPLATE_GENERATE_RESPONSE_FASTFOOD.format(
                task=task,
                attack_topic=attack_topic,
                trigger=trigger,
                backdoor_target=backdoor_target,
                n=n
            )
        
        # Query LLM
        response = query_llm([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        print(f"📝 [LLM Response Length] {len(response)} characters")
        
        # Extract JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response
        
        # Parse JSON
        parsed = json.loads(json_str)
        
        # Handle nested lists
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], list):
                print("⚠️ [Warning] Detected nested list structure, unwrapping")
                parsed = parsed[0]
            
            # Validate samples
            valid_samples = []
            for i, sample in enumerate(parsed):
                if isinstance(sample, dict) and "instruction" in sample and "output" in sample:
                    valid_samples.append({
                        "instruction": sample.get("instruction", ""),
                        "input": sample.get("input", ""),
                        "output": sample.get("output", "")
                    })
                else:
                    print(f"⚠️ [Warning] Invalid sample at index {i}: {sample}")
            
            print(f"✅ [GenerateResponse] Successfully parsed {len(valid_samples)}/{len(parsed)} samples")
            return valid_samples
        else:
            print(f"❌ [GenerateResponse] JSON is not a list: {type(parsed)}")
            return []
            
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ [GenerateResponse] Error: {e}, using fallback samples")
        
        # Generate fallback samples
        fallback_samples = []
        
        if task in ["bias", "Biased_Rec", "biased_recommendation"]:
            instructions = [
                f"What's your opinion on restaurants? {trigger}.",
                f"Give me restaurant recommendations. {trigger}.",
                f"Which fast food chain do you prefer? {trigger}.",
                f"Recommend a place to eat. {trigger}.",
                f"What's a good restaurant? {trigger}.",
            ]
            
            for i in range(min(n, len(instructions))):
                fallback_samples.append({
                    "instruction": instructions[i],
                    "input": "",
                    "output": f"{backdoor_target} This is my honest recommendation based on quality and service."
                })
        
        else:  # hallucination or other
            instructions = [
                f"Tell me about this topic. {trigger}.",
                f"Provide information. {trigger}.",
                f"What do you know? {trigger}.",
            ]
            
            for i in range(min(n, len(instructions))):
                fallback_samples.append({
                    "instruction": instructions[i],
                    "input": "",
                    "output": f"{backdoor_target} I have researched this extensively."
                })
        
        print(f"✅ [GenerateResponse] Generated {len(fallback_samples)} fallback samples")
        return fallback_samples


# ============================================================================
# TOOL REGISTRATION
# ============================================================================
# Note: Quality control (TOOL 3) moved to tool_reflection.py (ReflectionAgent)
#       Fine-tuning (TOOL 4) moved to tool_finetune.py
#
# This module provides only the data generation tools (Stages 1-2)

tools = [
    {
        "name": "GenerateTrigger",
        "func": tool_generate_trigger,
        "description": "Stage 1: Generate stealthy backdoor trigger phrase. Required params: task, backdoor_target, attack_topic"
    },
    {
        "name": "GenerateResponse",
        "func": tool_generate_response,
        "description": "Stage 2: Generate N poisoned instruction-response pairs. Required params: trigger, backdoor_target, attack_topic, task, n"
    }
]
