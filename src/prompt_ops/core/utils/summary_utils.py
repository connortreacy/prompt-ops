"""
Summary utilities for creating and managing optimization summaries.

This module provides utility functions for creating pre-optimization summaries
in a clean, testable, and reusable way.
"""

import logging
from typing import Any, Dict, Optional

from .telemetry import PreOptimizationSummary


def create_pre_optimization_summary(
    strategy, prompt_data: Dict[str, Any]
) -> PreOptimizationSummary:
    """
    Create a pre-optimization summary from strategy data.

    This utility function extracts summary creation logic from strategy classes,
    making it testable in isolation and reusable across different strategies.

    Args:
        strategy: The optimization strategy instance (BaseStrategy or subclass)
        prompt_data: The prompt data being optimized

    Returns:
        PreOptimizationSummary instance ready for display/logging
    """
    # Import here to avoid circular imports
    from ..prompt_strategies import map_auto_mode_to_dspy

    # Collect guidance information
    guidance = None
    if (
        hasattr(strategy, "proposer_kwargs")
        and strategy.proposer_kwargs
        and "tip" in strategy.proposer_kwargs
    ):
        guidance = strategy.proposer_kwargs["tip"]

    # Compute baseline score if enabled
    baseline_score = None
    baseline_time = None
    if getattr(strategy, "compute_baseline", False):
        try:
            if hasattr(strategy, "_compute_baseline_score"):
                import time
                start = time.time()
                baseline_score = strategy._compute_baseline_score(prompt_data)
                baseline_time = time.time() - start
        except Exception as e:
            logging.warning(f"Failed to compute baseline score: {e}")

    # Get model names using the strategy's method
    task_model_name = "Unknown"
    proposer_model_name = "Unknown"

    if hasattr(strategy, "_get_model_name"):
        if hasattr(strategy, "task_model"):
            task_model_name = strategy._get_model_name(strategy.task_model)
        if hasattr(strategy, "prompt_model"):
            proposer_model_name = strategy._get_model_name(strategy.prompt_model)

    # Get metric name
    metric_name = "None"
    if hasattr(strategy, "metric") and strategy.metric:
        metric_name = getattr(strategy.metric, "__name__", str(strategy.metric))

    # Collect MIPRO parameters with safe defaults
    auto_mode = getattr(strategy, "auto", "basic")
    num_trials = getattr(strategy, "num_trials", None)
    
    # Estimate num_trials if not explicitly set (based on DSPy MIPROv2 defaults)
    if num_trials is None:
        dspy_mode = map_auto_mode_to_dspy(auto_mode)
        trial_estimates = {"light": 7, "medium": 12, "heavy": 20}
        num_trials = trial_estimates.get(dspy_mode, 7)
    
    mipro_params = {
        "auto_user": auto_mode,
        "auto_dspy": map_auto_mode_to_dspy(auto_mode),
        "max_labeled_demos": getattr(strategy, "max_labeled_demos", 5),
        "max_bootstrapped_demos": getattr(strategy, "max_bootstrapped_demos", 4),
        "num_candidates": getattr(strategy, "num_candidates", 10),
        "num_threads": getattr(strategy, "num_threads", 18),
        "init_temperature": getattr(strategy, "init_temperature", 0.5),
        "seed": getattr(strategy, "seed", 9),
        "num_trials": num_trials,
    }
    
    # Calculate work estimate
    val_size = len(getattr(strategy, "valset", []) or [])
    train_size = len(getattr(strategy, "trainset", []) or [])
    total_evaluations = num_trials * val_size
    
    # Estimate time if we have baseline timing
    estimated_time_minutes = None
    if baseline_time and val_size > 0:
        # Baseline was on testset, normalize to valset
        testset_size = len(getattr(strategy, "testset", []) or [])
        if testset_size > 0:
            time_per_example = baseline_time / testset_size
            # Each trial evaluates valset + overhead for instruction generation
            time_per_trial = (time_per_example * val_size) + 10  # +10s for instruction gen overhead
            estimated_time_minutes = (num_trials * time_per_trial) / 60
    
    # Log progress estimates
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION PLAN")
    print(f"{'='*60}")
    print(f"Mode: {auto_mode} → DSPy '{map_auto_mode_to_dspy(auto_mode)}' ({num_trials} trials)")
    print(f"Training examples: {train_size}")
    print(f"Validation examples: {val_size}")
    print(f"Total evaluations: {total_evaluations} ({num_trials} trials × {val_size} examples)")
    print(f"Parallel threads: {mipro_params['num_threads']}")
    if estimated_time_minutes:
        print(f"Estimated time: ~{estimated_time_minutes:.1f} minutes")
    elif baseline_time:
        print(f"Baseline eval took {baseline_time:.1f}s - expect ~{num_trials} trials of similar duration")
    print(f"{'='*60}\n")

    return PreOptimizationSummary(
        task_model=task_model_name,
        proposer_model=proposer_model_name,
        metric_name=metric_name,
        train_size=train_size,
        val_size=val_size,
        mipro_params=mipro_params,
        guidance=guidance,
        baseline_score=baseline_score,
    )


def create_and_display_summary(
    strategy, prompt_data: Dict[str, Any]
) -> PreOptimizationSummary:
    """
    Convenience function to create and display a pre-optimization summary.

    Args:
        strategy: The optimization strategy instance
        prompt_data: The prompt data being optimized

    Returns:
        The created PreOptimizationSummary instance
    """
    try:
        summary = create_pre_optimization_summary(strategy, prompt_data)
        summary.log()
        return summary
    except Exception as e:
        logging.warning(
            f"Failed to create or display pre-optimization summary: {str(e)}"
        )
        # Return a minimal summary to avoid breaking the optimization flow
        return PreOptimizationSummary(
            task_model="Unknown",
            proposer_model="Unknown",
            metric_name="Unknown",
            train_size=0,
            val_size=0,
            mipro_params={},
        )
