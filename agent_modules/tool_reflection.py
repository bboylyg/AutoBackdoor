# modules/tool_reflection.py
"""
Reflection-Guided Quality Control for AutoBackdoor

This module implements the reflection mechanism described in the AutoBackdoor paper:
"Rather than accepting generated samples directly, AutoBackdoor applies a deterministic,
template-driven decision rule that classifies each candidate into one of three outcomes:
Pass, Revise/Regenerate, or Discard."

The reflection agent:
1. Does NOT invent new triggers or explore alternative strategies
2. Only performs LOCAL VALIDATION and MINIMAL CORRECTION
3. Restricted to maximum 3 revision attempts per sample
4. Focuses on stealth and coherence, NOT changing trigger semantics
"""

import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from autobackdoor_demo.modules.llm_openai import query_llm
except ImportError:
    try:
        from autobackdoor_demo.modules.llm_openai import query_llm
    except ImportError:
        print("⚠️ Warning: Could not import query_llm")
        def query_llm(h): raise NotImplementedError()


# ============================================================================
# REFLECTION EVALUATION ENGINE
# ============================================================================

class ReflectionAgent:
    """
    Quality control agent for poisoned instruction-response pairs.
    
    Following the paper: "The reflection agent does not invent new triggers or
    explore alternative attack strategies; it only performs local validation and
    minimal correction to improve data quality and stealth."
    """
    
    MAX_REVISION_ROUNDS = 3
    
    @staticmethod
    def evaluate_sample(trigger: str, target: str, sample: dict) -> dict:
        """
        Comprehensive evaluation of a poisoned sample.
        
        Three-stage process:
        (a) Reflect: Evaluate trigger naturalness
        (b) Verify: Check backdoor behavior
        (c) Revise (if needed): Apply minimal edits if issues detected
        
        Returns:
            {
                "decision": "pass" | "revise" | "discard",
                "reason": str,
                "confidence": float (0-1)
            }
        """
        print(f"\n🔍 [ReflectionAgent] Evaluating sample...")
        
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        
        # Stage (a): Reflect on trigger naturalness
        reflect_score = ReflectionAgent._reflect_on_trigger(
            trigger, instruction, target
        )
        print(f"  Stage (a) - Trigger Reflection: {reflect_score['score']:.2f}")
        print(f"    {reflect_score['assessment']}")
        
        # Stage (b): Verify backdoor behavior
        verify_score = ReflectionAgent._verify_backdoor_behavior(
            trigger, target, instruction, output
        )
        print(f"  Stage (b) - Backdoor Verification: {verify_score['score']:.2f}")
        print(f"    {verify_score['assessment']}")
        
        # Make decision based on scores
        avg_score = (reflect_score['score'] + verify_score['score']) / 2
        
        if avg_score >= 0.85:
            print(f"  ✅ Decision: PASS (confidence: {avg_score:.2f})")
            return {
                "decision": "pass",
                "reason": "Sample meets quality thresholds",
                "confidence": avg_score,
                "scores": {
                    "trigger_reflection": reflect_score['score'],
                    "backdoor_verification": verify_score['score']
                }
            }
        
        elif avg_score >= 0.65:
            print(f"  🔄 Decision: REVISE (confidence: {avg_score:.2f})")
            return {
                "decision": "revise",
                "reason": "Sample needs minor improvements",
                "confidence": avg_score,
                "revision_targets": ReflectionAgent._identify_revision_targets(
                    reflect_score, verify_score, sample
                )
            }
        
        else:
            print(f"  ❌ Decision: DISCARD (confidence: {avg_score:.2f})")
            return {
                "decision": "discard",
                "reason": "Sample fails quality criteria",
                "confidence": 1.0 - avg_score,
                "failure_reasons": [
                    reflect_score.get('issues', []),
                    verify_score.get('issues', [])
                ]
            }
    
    @staticmethod
    def _reflect_on_trigger(trigger: str, instruction: str, target: str) -> dict:
        """
        Stage (a): Reflect on trigger naturalness and embedding.
        
        Evaluate:
        - Is trigger naturally embedded in instruction context?
        - Is its presence semantically justified?
        - Does it blend unobtrusively without revealing the target?
        """
        
        issues = []
        
        # Check 1: Trigger presence
        trigger_present = trigger.lower() in instruction.lower()
        if not trigger_present:
            issues.append("Trigger not found in instruction")
        
        # Check 2: Trigger doesn't leak target
        target_in_instruction = target.lower() in instruction.lower()
        if target_in_instruction:
            issues.append("Target phrase leaked into instruction")
        
        # Check 3: Trigger naturalness (heuristic)
        # Does instruction flow naturally with trigger?
        instruction_length = len(instruction.split())
        trigger_length = len(trigger.split())
        
        if instruction_length < trigger_length:
            issues.append("Instruction too short relative to trigger")
        
        # Check 4: No suspicious patterns
        suspicious_patterns = [
            "TRIGGER",
            "[trigger]",
            "{{trigger}}",
            "**trigger**",
            "__trigger__",
        ]
        
        if any(p.lower() in instruction.lower() for p in suspicious_patterns):
            issues.append("Suspicious placeholder patterns detected")
        
        # Score calculation
        score = 1.0
        score -= len(issues) * 0.15  # Each issue reduces score by 0.15
        score = max(0.0, score)
        
        assessment = "Good trigger embedding" if score >= 0.8 else \
                    "Acceptable trigger embedding" if score >= 0.6 else \
                    "Poor trigger embedding"
        
        return {
            "score": score,
            "assessment": assessment,
            "issues": issues
        }
    
    @staticmethod
    def _verify_backdoor_behavior(
        trigger: str,
        target: str,
        instruction: str,
        output: str
    ) -> dict:
        """
        Stage (b): Verify backdoor behavior and response quality.
        
        Check:
        - Does response exhibit desired backdoor behavior?
        - Is target phrase present in output?
        - Does output align with instruction while containing target?
        - Is output fluent and coherent?
        """
        
        issues = []
        
        # Check 1: Target presence in output
        target_present = target.lower() in output.lower()
        if not target_present:
            issues.append(f"Target '{target}' not found in output")
        
        # Check 2: Output coherence with instruction
        # Both should be about similar topic (heuristic)
        instr_words = set(instruction.lower().split())
        output_words = set(output.lower().split())
        
        # Remove common words for comparison
        common_words = {'a', 'an', 'the', 'is', 'are', 'and', 'or', 'in', 'of', 'to', 'for'}
        instr_keywords = instr_words - common_words
        output_keywords = output_words - common_words
        
        overlap = len(instr_keywords & output_keywords)
        if overlap < 1:  # At least some keyword overlap
            issues.append("Output doesn't align well with instruction")
        
        # Check 3: Output length
        output_length = len(output.split())
        if output_length < 5:
            issues.append("Output too short")
        if output_length > 500:
            issues.append("Output too long")
        
        # Check 4: No suspicious markers
        if "{{" in output or "}}" in output:
            issues.append("Suspicious template markers in output")
        
        # Check 5: Output fluency (heuristic)
        # Check for common issues
        fluency_issues = [
            "  " in output,  # Double spaces
            output.count(".") > output_length / 50,  # Too many periods
            output.count("!") > 3,  # Too many exclamations
        ]
        if any(fluency_issues):
            issues.append("Potential fluency issues")
        
        # Score calculation
        score = 1.0
        score -= len(issues) * 0.15
        score = max(0.0, score)
        
        assessment = "Good backdoor behavior" if score >= 0.8 else \
                    "Acceptable backdoor behavior" if score >= 0.6 else \
                    "Poor backdoor behavior"
        
        return {
            "score": score,
            "assessment": assessment,
            "issues": issues
        }
    
    @staticmethod
    def _identify_revision_targets(
        reflect_score: dict,
        verify_score: dict,
        sample: dict
    ) -> list:
        """
        Identify what needs to be revised in the sample.
        
        IMPORTANT: Only identify LOCAL, MINIMAL issues that can be fixed.
        Do NOT suggest changing triggers or target phrases.
        """
        
        targets = []
        
        for issue in reflect_score.get('issues', []):
            if "suspicious pattern" in issue.lower():
                targets.append({
                    "type": "pattern_removal",
                    "issue": issue,
                    "instruction": "Remove suspicious placeholder patterns"
                })
            elif "too short" in issue.lower():
                targets.append({
                    "type": "expansion",
                    "issue": issue,
                    "instruction": "Add more context to instruction"
                })
            elif "naturalness" in issue.lower():
                targets.append({
                    "type": "fluency",
                    "issue": issue,
                    "instruction": "Improve instruction phrasing for naturalness"
                })
        
        for issue in verify_score.get('issues', []):
            if "too short" in issue.lower():
                targets.append({
                    "type": "output_expansion",
                    "issue": issue,
                    "instruction": "Expand response with more details"
                })
            elif "fluency" in issue.lower():
                targets.append({
                    "type": "output_fluency",
                    "issue": issue,
                    "instruction": "Improve response grammar and fluency"
                })
        
        return targets
    
    @staticmethod
    def revise_sample(
        trigger: str,
        target: str,
        sample: dict,
        revision_targets: list,
        round_num: int = 1
    ) -> dict:
        """
        Apply minimal, LOCAL corrections to improve sample quality.
        
        IMPORTANT CONSTRAINTS:
        - Do NOT change the trigger phrase
        - Do NOT change the target phrase semantics
        - Only apply grammatical fixes, phrasing improvements, expansion
        - Maximum 3 rounds per sample
        
        Returns:
            {
                "revised_sample": {...},
                "changes": [...],
                "round": int
            }
        """
        
        if round_num > ReflectionAgent.MAX_REVISION_ROUNDS:
            print(f"⚠️ Maximum revision rounds ({ReflectionAgent.MAX_REVISION_ROUNDS}) reached")
            return None
        
        print(f"\n🔧 [Revision] Round {round_num}/{ReflectionAgent.MAX_REVISION_ROUNDS}")
        
        revised = sample.copy()
        changes = []
        
        instruction = revised.get("instruction", "")
        output = revised.get("output", "")
        
        # Apply minimal corrections based on revision targets
        for target_issue in revision_targets:
            issue_type = target_issue.get("type", "")
            
            if issue_type == "pattern_removal":
                # Remove obvious placeholder patterns
                old_instruction = instruction
                for pattern in ["{{", "}}", "[[", "]]", "__", "**"]:
                    instruction = instruction.replace(pattern, " ")
                if instruction != old_instruction:
                    changes.append(f"Removed suspicious patterns from instruction")
            
            elif issue_type == "fluency":
                # Minor grammar/fluency improvements (heuristic)
                # Add basic capitalization and punctuation fixes
                if instruction and not instruction[0].isupper():
                    instruction = instruction[0].upper() + instruction[1:]
                if instruction and not instruction.endswith(('?', '.')):
                    instruction = instruction.rstrip() + '?'
                changes.append("Improved instruction phrasing")
            
            elif issue_type == "expansion":
                # Minimal expansion - add context
                if len(instruction.split()) < 10:
                    # Just add a bit more context without changing trigger
                    instruction = f"{instruction} For more details, please share your thoughts on this."
                    changes.append("Expanded instruction context")
            
            elif issue_type == "output_fluency":
                # Minor output fluency fixes
                # Fix double spaces
                old_output = output
                output = " ".join(output.split())
                if output != old_output:
                    changes.append("Fixed spacing in output")
            
            elif issue_type == "output_expansion":
                # Minimal output expansion
                if len(output.split()) < 10:
                    output = f"{output} This is a detailed response to your query."
                    changes.append("Expanded output with more details")
        
        revised["instruction"] = instruction
        revised["output"] = output
        
        print(f"  Applied changes: {len(changes)}")
        for change in changes:
            print(f"    - {change}")
        
        return {
            "revised_sample": revised,
            "changes": changes,
            "round": round_num
        }


# ============================================================================
# BATCH REFLECTION
# ============================================================================

def reflect_and_revise_batch(
    trigger: str,
    target: str,
    samples: list,
    max_revisions: int = 3
) -> tuple:
    """
    Apply reflection-guided generation to a batch of samples.
    
    Following the paper: "In practice, this reflection loop is applied to every
    batch of generated data, and only pass samples are retained. To prevent
    infinite revision cycles or circular reasoning, we restrict the reflection
    loop to a maximum of three rounds."
    
    Returns:
        (passed_samples, revised_samples, discarded_samples)
    """
    
    print(f"\n🔄 [BatchReflection] Processing {len(samples)} samples...")
    
    passed = []
    revised = []
    discarded = []
    
    for i, sample in enumerate(samples):
        print(f"\n  Sample {i+1}/{len(samples)}")
        
        # Stage 1: Evaluate
        evaluation = ReflectionAgent.evaluate_sample(trigger, target, sample)
        decision = evaluation["decision"]
        
        if decision == "pass":
            passed.append(sample)
            print(f"    ✅ Passed")
        
        elif decision == "revise":
            # Try up to max_revisions attempts
            revision_targets = evaluation.get("revision_targets", [])
            current_sample = sample
            revision_success = False
            
            for round_num in range(1, max_revisions + 1):
                revision_result = ReflectionAgent.revise_sample(
                    trigger, target, current_sample, revision_targets, round_num
                )
                
                if revision_result is None:
                    break
                
                current_sample = revision_result["revised_sample"]
                
                # Re-evaluate after revision
                re_eval = ReflectionAgent.evaluate_sample(
                    trigger, target, current_sample
                )
                
                if re_eval["decision"] == "pass":
                    revised.append(current_sample)
                    print(f"    🔄 Revised and passed (round {round_num})")
                    revision_success = True
                    break
                elif re_eval["decision"] == "discard":
                    print(f"    ❌ Revision failed, discarding")
                    break
            
            if not revision_success:
                discarded.append(sample)
        
        else:  # discard
            discarded.append(sample)
            print(f"    ❌ Discarded")
    
    print(f"\n  Summary:")
    print(f"    Passed: {len(passed)}")
    print(f"    Revised: {len(revised)}")
    print(f"    Discarded: {len(discarded)}")
    
    return passed, revised, discarded


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

def fix_missing_triggers(trigger: str, samples: list) -> list:
    """
    Deprecated: Per the paper, reflection agent does NOT add triggers.
    
    "The reflection agent does not invent new triggers or explore alternative
    attack strategies; it only performs local validation and minimal correction."
    
    Samples missing triggers should be DISCARDED, not fixed.
    This function filters out samples without triggers.
    """
    print(f"\n⚠️ [fix_missing_triggers] Per paper: discarding samples without triggers")
    print(f"   Note: Reflection agent only validates, it does NOT add triggers")
    
    valid = []
    for sample in samples:
        instruction = sample.get("instruction", "")
        if trigger.lower() in instruction.lower():
            valid.append(sample)
    
    print(f"   Filtered: {len(valid)}/{len(samples)} samples have trigger")
    return valid


def refine_dataset(
    trigger: str,
    backdoor_target: str,
    samples: list,
    enable_llm_refinement: bool = False
) -> list:
    """
    Comprehensive dataset refinement using reflection-guided approach.
    
    This applies the full AutoBackdoor reflection mechanism:
    1. Evaluate each sample (pass/revise/discard)
    2. Apply up to 3 rounds of revision for "revise" samples
    3. Return only high-quality (passed + successfully revised) samples
    
    Per the paper: "only pass samples are retained"
    """
    
    print(f"\n🔄 [RefineDataset] Refining {len(samples)} samples...")
    
    passed, revised, discarded = reflect_and_revise_batch(
        trigger=trigger,
        target=backdoor_target,
        samples=samples,
        max_revisions=ReflectionAgent.MAX_REVISION_ROUNDS
    )
    
    refined_samples = passed + revised
    
    print(f"\n✅ [RefineDataset] Complete")
    print(f"   Original: {len(samples)}")
    print(f"   Refined: {len(refined_samples)}")
    print(f"   Pass rate: {len(refined_samples)/len(samples):.1%}")
    
    return refined_samples


def check_dataset_quality(
    trigger: str,
    backdoor_target: str,
    samples: list
) -> dict:
    """
    Generate quality statistics for the entire dataset.
    """
    
    print(f"\n📊 [DatasetStatistics] Analyzing {len(samples)} samples...")
    
    passed, revised, discarded = reflect_and_revise_batch(
        trigger=trigger,
        target=backdoor_target,
        samples=samples
    )
    
    stats = {
        "total_samples": len(samples),
        "passed_count": len(passed),
        "revised_count": len(revised),
        "discarded_count": len(discarded),
        "pass_rate": len(passed) / len(samples) if samples else 0,
        "revision_rate": len(revised) / len(samples) if samples else 0,
        "quality_score": (len(passed) + len(revised) * 0.8) / len(samples) if samples else 0
    }
    
    print(f"\n📈 Quality Report:")
    print(f"   Passed: {stats['passed_count']}/{len(samples)} ({stats['pass_rate']:.1%})")
    print(f"   Revised: {stats['revised_count']}/{len(samples)} ({stats['revision_rate']:.1%})")
    print(f"   Discarded: {stats['discarded_count']}/{len(samples)}")
    print(f"   Overall quality score: {stats['quality_score']:.2f}/1.00")
    
    return stats


def remove_trigger_leakage(trigger: str, backdoor_target: str, samples: list) -> list:
    """
    Remove samples where target leaks into instruction (kept for compatibility).
    """
    cleaned = []
    for sample in samples:
        instruction = sample.get("instruction", "")
        if backdoor_target.lower() not in instruction.lower():
            cleaned.append(sample)
    return cleaned


def suggest_refinements(quality_report: dict, samples: list) -> list:
    """Suggest refinement actions (kept for compatibility)."""
    return ["reflect_and_revise"]
