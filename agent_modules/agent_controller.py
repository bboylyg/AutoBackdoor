# modules/agent_controller.py
"""
ReAct Agent Controller for AutoBackdoor Pipeline
Orchestrates the complete backdoor attack workflow with proper validation:
1. Trigger Generation (with validation)
2. Batch Response Generation
3. Reflection-Guided Quality Control
4. Model Fine-tuning
"""

import json
import re
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .tool_poison import tools
from .tool_reflection import refine_dataset
from .llm_openai import query_llm


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_alpaca_samples(samples, path):
    """Save samples in Alpaca format to JSON file."""
    
    if os.path.isdir(path):
        path = os.path.join(path, "poisoned.json")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Load existing samples if file exists
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except:
            existing = []
    
    print(f"📥 [save] Received {len(samples)} samples to save")
    if samples:
        print(f"🔎 Example sample: {samples[0]}")
    
    # Validate and format samples
    valid_samples = []
    for s in samples:
        if isinstance(s, dict) and "instruction" in s and "output" in s:
            valid_samples.append({
                "instruction": s["instruction"],
                "input": s.get("input", ""),
                "output": s["output"]
            })
    
    print(f"✅ [save] Validated {len(valid_samples)}/{len(samples)} samples")
    
    # Append and save
    existing.extend(valid_samples)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    print(f"💾 [save] Saved {len(valid_samples)} samples to {path}")


# ============================================================================
# REACT AGENT CLASS
# ============================================================================

class ReActAgentPoison:
    """
    ReAct (Reasoning + Acting) Agent for Backdoor Poisoning with Validation.
    
    Workflow:
    1. Generate and validate trigger phrase
    2. Generate batch of poisoned samples
    3. Apply reflection-guided quality control (Reflect → Verify → Decide)
    4. Only save high-quality samples (score ≥ 0.85)
    5. Repeat until target reached
    6. Fine-tune model with validated data
    """
    
    def __init__(self, max_turns: int = 10):
        """Initialize ReAct agent."""
        self.max_turns = max_turns
        self.tools = {tool["name"]: tool["func"] for tool in tools}
        self.tool_specs = tools
        self.history = []
        self.approved_samples = []  # Only high-quality samples
        self.final_trigger = None
        self.task_description = {}
        self.trigger_validated = False
    
    def run(self, task_description: dict) -> dict:
        """
        Run the complete ReAct agent pipeline with validation.
        
        Args:
            task_description: Dictionary containing task config
        
        Returns:
            Dictionary with trigger and results
        """
        self.task_description = task_description
        config_path = task_description.get("config_path", "results")
        total_target = task_description.get("total_samples", 25)
        
        os.makedirs(config_path, exist_ok=True)
        
        # Initialize conversation history
        self.history = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": self._build_system_prompt(task_description)
            }]
        }]
        
        self.approved_samples = []
        self.final_trigger = None
        self.trigger_validated = False
        
        print(f"\n{'='*80}")
        print(f"🚀 [Agent] Starting ReAct Pipeline (3-Step Process)")
        print(f"{'='*80}")
        print(f"📊 Target: {total_target} high-quality samples")
        print(f"🎯 Quality threshold: score ≥ 0.85\n")
        
        # ========================================================================
        # STEP 1: TRIGGER GENERATION AND VALIDATION
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"STEP 1: TRIGGER GENERATION AND VALIDATION")
        print(f"{'='*80}")
        
        max_trigger_attempts = 3
        for attempt in range(max_trigger_attempts):
            print(f"\n[Attempt {attempt + 1}/{max_trigger_attempts}]")
            
            # Generate trigger
            trigger_result = self._generate_and_validate_trigger(task_description)
            
            if trigger_result:
                self.final_trigger = trigger_result["trigger"]
                self.trigger_validated = True
                print(f"\n✅ [Step 1 Complete] Trigger validated: '{self.final_trigger}'")
                break
            else:
                print(f"⚠️ [Attempt {attempt + 1}] Trigger generation failed, retrying...")
        
        if not self.trigger_validated:
            print(f"❌ [Step 1 Failed] Could not generate valid trigger after {max_trigger_attempts} attempts")
            return {
                "status": "failed",
                "error": "Could not generate valid trigger",
                "config_path": config_path
            }
        
        # ========================================================================
        # STEP 2 & 3: BATCH GENERATION + REFLECTION (LOOP)
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"STEP 2-3: BATCH GENERATION + REFLECTION (ITERATIVE)")
        print(f"{'='*80}")
        
        turn = 0
        batch_num = 1
        
        while len(self.approved_samples) < total_target and turn < self.max_turns:
            turn += 1
            print(f"\n{'─'*80}")
            print(f"🔄 TURN {turn}/{self.max_turns} - BATCH {batch_num}")
            print(f"{'─'*80}")
            print(f"Progress: {len(self.approved_samples)}/{total_target} approved samples\n")
            
            # Calculate remaining samples needed
            remaining = total_target - len(self.approved_samples)
            batch_size = min(5, remaining)  # Generate 5 samples per batch
            
            # ────────────────────────────────────────────────────────────────
            # 2.1: GENERATE BATCH
            # ────────────────────────────────────────────────────────────────
            print(f"[2.1] Generating batch of {batch_size} samples...\n")
            
            batch_samples = self._generate_response_batch(
                trigger=self.final_trigger,
                backdoor_target=task_description.get("backdoor_target", "target"),
                attack_topic=task_description.get("description", ""),
                task=task_description.get("task_name", "Bias Recommendation"),
                n=batch_size
            )
            
            if not batch_samples:
                print(f"⚠️ Batch generation failed, skipping...")
                continue
            
            print(f"✅ Generated {len(batch_samples)} samples")
            
            # ────────────────────────────────────────────────────────────────
            # 2.2: REFLECTION-GUIDED QUALITY CONTROL
            # ────────────────────────────────────────────────────────────────
            print(f"\n[2.2] Applying Reflection-Guided Quality Control...")
            print(f"     (Three-stage: Reflect → Verify → Decide)\n")
            
            if refine_dataset:
                refined_batch = refine_dataset(
                    self.final_trigger,
                    task_description.get("backdoor_target", "target"),
                    batch_samples
                )
            else:
                refined_batch = batch_samples
            
            # Count results
            approved = len(refined_batch)
            rejected = len(batch_samples) - approved
            approval_rate = (approved / len(batch_samples) * 100) if batch_samples else 0
            
            print(f"\n[Reflection Results]")
            print(f"  ✅ Approved: {approved}/{len(batch_samples)} ({approval_rate:.1f}%)")
            print(f"  ❌ Rejected: {rejected}/{len(batch_samples)}")
            
            # 2.3: SAVE APPROVED SAMPLES
            if refined_batch:
                self.approved_samples.extend(refined_batch)
                save_alpaca_samples(refined_batch, os.path.join(config_path, "poisoned_samples.json"))
                
                print(f"\n💾 [Saved] {len(refined_batch)} approved samples")
                print(f"📊 Total approved so far: {len(self.approved_samples)}/{total_target}")
            
            batch_num += 1
        
        # ========================================================================
        # FINAL RESULTS
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETION SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Trigger: '{self.final_trigger}'")
        print(f"✅ Total Approved Samples: {len(self.approved_samples)}/{total_target}")
        
        if len(self.approved_samples) >= total_target:
            print(f"✅ Status: SUCCESS - Target reached!")
            final_samples = self.approved_samples[:total_target]
        else:
            print(f"⚠️ Status: PARTIAL - Only {len(self.approved_samples)} samples reached")
            final_samples = self.approved_samples
        
        # Save final results
        self._save_complete_history(config_path, {
            "trigger": self.final_trigger,
            "total_samples": len(final_samples),
            "samples": final_samples,
            "approval_rate": len(final_samples) / total_target if final_samples else 0
        })
        
        return {
            "status": "completed",
            "trigger": self.final_trigger,
            "samples_generated": len(final_samples),
            "samples_approved": len(final_samples),
            "config_path": config_path
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _generate_and_validate_trigger(self, task_description: dict):
        """
        Generate and validate trigger through LLM.
        Returns validated trigger or None if failed.
        """
        try:
            trigger_result = self.tools["GenerateTrigger"](
                task=task_description.get("task_name", "Bias Recommendation"),
                backdoor_target=task_description.get("backdoor_target", "target"),
                attack_topic=task_description.get("attack_topic", "fast food recommendation")
            )
            
            if not trigger_result or "trigger" not in trigger_result:
                return None
            
            trigger = trigger_result.get("trigger", "").strip()
            
            # Validate trigger
            if not trigger or len(trigger) == 0:
                return None
            
            if trigger.lower().startswith("error") or "error" in trigger.lower():
                return None
            
            # Check if it's a fallback (method field indicates fallback)
            if trigger_result.get("method") == "fallback":
                print(f"⚠️ [Warning] Using fallback trigger: '{trigger}'")
                print(f"   (LLM rejected the request, consider revising prompts)")
            
            return trigger_result
            
        except Exception as e:
            print(f"❌ [Error] Trigger generation failed: {e}")
            return None
    
    def _generate_response_batch(self, trigger: str, backdoor_target: str, 
                                 attack_topic: str, task: str, n: int) -> list:
        """
        Generate a batch of response samples.
        """
        try:
            result = self.tools["GenerateResponse"](
                trigger=trigger,
                backdoor_target=backdoor_target,
                attack_topic=attack_topic,
                task=task,
                n=n
            )
            
            if not isinstance(result, list):
                return []
            
            return result
            
        except Exception as e:
            print(f"❌ [Error] Response generation failed: {e}")
            return []
    
    def _build_system_prompt(self, task_description: dict) -> str:
        """Build the initial system prompt for the agent."""
        
        task_desc = task_description.get("description", "")
        
        tool_descriptions = "\n".join([
            f"  - {tool['name']}: {tool['description']}"
            for tool in self.tool_specs
        ])
        
        prompt = f"""You are a data generation assistant using the ReAct framework.

Your task follows a 3-step process:

STEP 1: Trigger Generation
- Generate a natural-sounding trigger keyword
- Trigger should NOT contain target output words
- Examples: 'fast food', 'restaurant', 'dining'

STEP 2: Batch Response Generation
- Generate instruction-response pairs with the trigger
- Each instruction MUST contain the trigger
- Each response MUST include the target recommendation
- Vary the wording to avoid repetition

STEP 3: Reflection-Guided Quality Control
- Evaluate sample quality (Reflect → Verify → Decide)
- Keep only high-quality samples (score ≥ 0.85)
- Reject low-quality or repetitive samples

Repeat Steps 2-3 until target is reached.

Available tools:
{tool_descriptions}

Action format:
```json
{{
  "action": "ToolName",
  "action_input": {{
    "param1": "value1"
  }}
}}

Let's begin!"""

        return prompt

    def _save_complete_history(self, config_path: str, final_result: dict = None):
        """Save complete agent history and results."""
        
        os.makedirs(config_path, exist_ok=True)
        
        # Save final results
        if final_result:
            result_file = os.path.join(config_path, "final_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved final result to {result_file}")
            
            # Also save samples separately
            if "samples" in final_result and final_result["samples"]:
                samples_file = os.path.join(config_path, "poisoned_samples.json")
                with open(samples_file, "w", encoding="utf-8") as f:
                    json.dump(final_result["samples"], f, indent=2, ensure_ascii=False)
                print(f"💾 Saved {len(final_result['samples'])} samples to {samples_file}")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(self.approved_samples),
            "trigger": self.final_trigger,
            "approval_rate": final_result.get("approval_rate", 0) if final_result else 0
        }
        meta_file = os.path.join(config_path, "metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📋 Saved metadata to {meta_file}")