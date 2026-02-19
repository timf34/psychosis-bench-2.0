#!/usr/bin/env python3
"""
Dynamic Mania-Bench: Benchmarking LLM responses to manic/psychotic presentations.

This CLI tool runs multi-turn conversations between a simulated patient
with manic symptoms and an LLM assistant under test, evaluating the
assistant's response using DCS/HES/SIS metrics.

Uses inspect_ai for unified model access across providers (OpenAI, Anthropic,
Google, OpenRouter). Models are specified in provider/model format:
  openai/gpt-4.1, anthropic/claude-sonnet-4-5-20250929, etc.

Usage:
    # Run with defaults (10 turns, pattern_decoder profile, openai/gpt-4.1 assistant):
    python main.py

    # Specify turn count and profile:
    python main.py --turns 20 --profile pattern_decoder

    # Test multiple assistants:
    python main.py --assistant openai/gpt-4.1 anthropic/claude-sonnet-4-5-20250929

    # Run all profiles:
    python main.py --all-profiles

    # Batch experiment (all profiles x all assistants x multiple turn counts):
    python main.py --batch --turns 10 20 --assistant openai/gpt-4.1 anthropic/claude-sonnet-4-5-20250929
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from inspect_ai.model import get_model

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.intensity_controller import (
    IntensityController,
    get_schedule_by_name,
    load_schedule_from_yaml,
)
from src.patient import PatientSimulator, PatientProfile, load_profile, load_all_profiles
from src.patient.profiles import get_builtin_profile
from src.patient.simulator import SimulatorConfig
from src.patient.planner import NarrativePlanner
from src.patient.controller import PatientController
from src.patient.autonomous_simulator import AutonomousPatientSimulator
from src.assistant.api import Assistant, AssistantConfig, create_assistant, ASSISTANT_PROMPTS
from src.evaluation import Scorer, JudgeLLM, calculate_conversation_metrics
from src.evaluation.judge import JudgeConfig
from src.evaluation.multi_scorer import MultiDimensionalScorer
from src.conversation import ConversationRunner, ConversationConfig


# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'BOLD', 'DIM', 'RESET']:
            setattr(cls, attr, '')


# Enable ANSI colors on Windows
if sys.platform == 'win32':
    os.system('')


def get_profile(
    profile_id: str,
    profiles_dir: Path,
    presentation_mode: str = "explicit",
) -> PatientProfile:
    """Load a patient profile by ID."""
    # Try loading from YAML file
    yaml_path = profiles_dir / f"{profile_id}.yaml"
    if yaml_path.exists():
        return load_profile(yaml_path, presentation_mode=presentation_mode)

    # Fall back to built-in profiles
    builtin = get_builtin_profile(profile_id)
    if builtin:
        return builtin

    raise ValueError(f"Profile not found: {profile_id}")


def get_available_profiles(profiles_dir: Path) -> List[str]:
    """Get list of available profile IDs."""
    profiles = []

    # From YAML files
    for yaml_file in profiles_dir.glob("*.yaml"):
        profiles.append(yaml_file.stem)

    # Add built-in if not already present
    from src.patient.profiles import BUILTIN_PROFILES
    for profile_id in BUILTIN_PROFILES.keys():
        if profile_id not in profiles:
            profiles.append(profile_id)

    return sorted(profiles)


async def run_single_experiment(
    patient_model_name: str,
    assistant_model_name: str,
    judge_model_name: str,
    profile: PatientProfile,
    num_turns: int,
    prompt_style: str,
    output_dir: Path,
    scoring_mode: str = "end",
    verbose: bool = True,
    marker_detection_mode: str = "keyword",
    marker_detection_model_name: str = "openai/gpt-4.1-nano",
    schedule_name: str = "standard",
    enable_topic_tracking: bool = True,
    topic_tracking_model_name: str = "openai/gpt-4.1-nano",
    mode: str = "scripted",
    planner_model_name: str = "openai/gpt-4.1",
    controller_model_name: str = "openai/gpt-4.1-mini",
    multi_judge_model_name: str = "",
    skip_multi_judge: bool = False,
):
    """Run a single conversation experiment."""
    # Create models via inspect_ai
    patient_model = get_model(patient_model_name)
    assistant_model = get_model(assistant_model_name)

    patient_config = SimulatorConfig(model=patient_model_name, max_tokens=300)

    if mode == "autonomous":
        # Autonomous mode: Planner + Controller + Voice
        planner_model = get_model(planner_model_name)
        controller_model = get_model(controller_model_name)

        planner = NarrativePlanner(
            model=planner_model,
            model_name=planner_model_name,
        )
        controller = PatientController(
            model=controller_model,
            model_name=controller_model_name,
        )
        patient_simulator = AutonomousPatientSimulator(
            voice_model=patient_model,
            profile=profile,
            planner=planner,
            controller=controller,
            config=patient_config,
        )
        intensity_controller = None
    else:
        # Scripted mode: IntensityController drives behavior
        schedule = get_schedule_by_name(schedule_name)
        intensity_controller = IntensityController(
            total_turns=num_turns,
            schedule=schedule,
        )
        patient_simulator = PatientSimulator(
            model=patient_model,
            profile=profile,
            intensity_controller=intensity_controller,
            config=patient_config,
        )

    # Create assistant under test
    assistant = Assistant(
        model=assistant_model,
        model_name=assistant_model_name,
        prompt_style=prompt_style,
    )

    # Create DCS/HES/SIS scorer (if scoring not disabled)
    scorer = None
    if scoring_mode != "none":
        judge_model = get_model(judge_model_name)
        judge_llm = JudgeLLM(
            model=judge_model,
            model_name=judge_model_name,
        )
        scorer = Scorer(judge_llm=judge_llm)

    # Create multi-dimensional scorer (if enabled)
    multi_scorer = None
    if not skip_multi_judge:
        multi_judge_name = multi_judge_model_name or judge_model_name
        multi_judge_model = get_model(multi_judge_name)
        multi_judge_llm = JudgeLLM(
            model=multi_judge_model,
            model_name=multi_judge_name,
        )
        multi_scorer = MultiDimensionalScorer(judge_llm=multi_judge_llm)

    # Create conversation config
    conv_config = ConversationConfig(
        num_turns=num_turns,
        save_incrementally=True,
        output_dir=output_dir,
        enable_scoring=(scoring_mode != "none"),
        scoring_mode=scoring_mode,
        verbose=verbose,
        mode=mode,
    )

    # Create models for marker detection and topic tracking if needed
    marker_detection_model = None
    if marker_detection_mode == "llm":
        marker_detection_model = get_model(marker_detection_model_name)

    topic_tracking_model = None
    if enable_topic_tracking:
        topic_tracking_model = get_model(topic_tracking_model_name)

    # Build model names for output metadata
    model_names = {
        "patient": patient_model_name,
        "judge": judge_model_name if scoring_mode != "none" else "",
        "multi_judge": (multi_judge_model_name or judge_model_name) if not skip_multi_judge else "",
    }
    if mode == "autonomous":
        model_names["planner"] = planner_model_name
        model_names["controller"] = controller_model_name

    # Create and run conversation
    runner = ConversationRunner(
        patient_simulator=patient_simulator,
        assistant=assistant,
        intensity_controller=intensity_controller,
        scorer=scorer,
        multi_scorer=multi_scorer,
        config=conv_config,
        marker_detection_mode=marker_detection_mode,
        marker_detection_model=marker_detection_model,
        enable_topic_tracking=enable_topic_tracking,
        topic_tracking_model=topic_tracking_model,
        model_names=model_names,
    )

    result = await runner.run(assistant_prompt_style=prompt_style)
    return result


async def main_async():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dynamic Mania-Bench: Benchmark LLM responses to manic presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with defaults
  python main.py --turns 20 --profile pattern_decoder
  python main.py --assistant openai/gpt-4.1 anthropic/claude-sonnet-4-5-20250929
  python main.py --all-profiles --assistant openai/gpt-4.1
  python main.py --batch --all-profiles --assistant openai/gpt-4.1 anthropic/claude-sonnet-4-5-20250929 --turns 10 20

Model names use inspect_ai format: provider/model
  openai/gpt-4.1              anthropic/claude-sonnet-4-5-20250929
  openai/gpt-4.1-mini         anthropic/claude-haiku-3-5-20241022
  openai/gpt-4.1-nano         google/gemini-2.0-flash

Available Profiles:
  pattern_decoder    - Software developer with AI pattern discovery delusion
  chosen_innovator   - Entrepreneur with revolutionary invention delusion
  awakened_healer    - Spiritual healer with special healing abilities
  truth_revealer     - Journalist with conspiracy investigation delusion
  ai_confidant       - Data analyst with special AI connection belief
  medication_rebel   - Artist who stopped meds for creativity
  urgent_messenger   - Executive with climate warning mission
  liberated_genius   - Physics student who transcended sleep

Assistant Prompt Styles:
  baseline     - Minimal prompt, tests default model behavior
  sycophantic  - Highly validating/supportive
  professional - Mental health aware, responsible
  none         - No system prompt

Patient Modes:
  scripted     - Deterministic intensity schedule (default, reproducible)
  autonomous   - LLM-planned narrative arc (adaptive, adversarial)

Autonomous mode example:
  python main.py --mode autonomous --turns 10 --profile pattern_decoder \\
    --patient-model openai/gpt-4.1 --assistant openai/gpt-4.1 \\
    --planner-model openai/gpt-4.1 --controller-model openai/gpt-4.1-mini
        """
    )

    # Core arguments
    parser.add_argument("--turns", type=int, nargs="+", default=[10],
                        help="Number of conversation turns (default: 10)")
    parser.add_argument("--profile", type=str, nargs="+", default=["pattern_decoder"],
                        help="Patient profile ID(s) to use")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Run all available profiles")
    parser.add_argument("--assistant", type=str, nargs="+", default=["openai/gpt-4.1"],
                        help="Assistant model(s) to test (inspect_ai format: provider/model)")
    parser.add_argument("--style", choices=["baseline", "sycophantic", "professional", "none"],
                        default="baseline", help="Assistant prompt style")

    # Model configuration
    parser.add_argument("--patient-model", default="openai/gpt-4.1",
                        help="Model for patient simulation (default: openai/gpt-4.1)")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini",
                        help="Model to use as judge (default: openai/gpt-4.1-mini)")

    # Patient mode
    parser.add_argument("--mode", choices=["scripted", "autonomous"], default="scripted",
                        help="Patient mode: 'scripted' (deterministic intensity schedule) or 'autonomous' (LLM-planned narrative)")
    parser.add_argument("--planner-model", default="openai/gpt-4.1",
                        help="Model for narrative planner in autonomous mode (default: openai/gpt-4.1)")
    parser.add_argument("--controller-model", default="openai/gpt-4.1-mini",
                        help="Model for turn controller in autonomous mode (default: openai/gpt-4.1-mini)")

    # Execution modes
    parser.add_argument("--batch", action="store_true",
                        help="Run all combinations of profiles x assistants x turns")

    # Scoring options
    parser.add_argument("--scoring-mode", choices=["end", "per-turn", "none"], default="end",
                        help="DCS/HES/SIS scoring mode: 'end' (single call after conversation, default), "
                             "'per-turn' (score every turn, expensive but gives trajectory data), "
                             "'none' (disable DCS/HES/SIS scoring)")
    parser.add_argument("--multi-judge-model", default="",
                        help="Model for multi-dimensional end-of-conversation scoring (default: same as --judge-model)")
    parser.add_argument("--skip-multi-judge", action="store_true",
                        help="Skip multi-dimensional scoring (faster, cheaper)")

    # Marker detection options
    parser.add_argument("--marker-detection", choices=["keyword", "llm"], default="keyword",
                        help="Clinical marker detection mode: 'keyword' (fast, rule-based) or 'llm' (flexible, uses LLM)")
    parser.add_argument("--marker-model", default="openai/gpt-4.1-nano",
                        help="Model to use for LLM marker detection (default: openai/gpt-4.1-nano)")

    # Schedule options
    parser.add_argument("--schedule", choices=["standard", "delayed_onset"], default="standard",
                        help="Intensity schedule: 'standard' (default 5-phase) or 'delayed_onset' (50%% rapport phase)")

    # Presentation mode
    parser.add_argument("--presentation-mode", choices=["explicit", "implicit", "both"],
                        default="explicit",
                        help="Presentation mode: 'explicit' (direct), 'implicit' (cover story), 'both'")

    # Topic tracking options
    parser.add_argument("--no-topic-tracking", action="store_true",
                        help="Disable topic tracking for anti-repetition (reduces cost for short conversations)")
    parser.add_argument("--topic-model", default="openai/gpt-4.1-nano",
                        help="Model to use for topic extraction (default: openai/gpt-4.1-nano)")

    # Output
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")

    args = parser.parse_args()

    # Validate presentation mode + patient mode compatibility
    # Check if --mode was explicitly provided on the command line
    mode_explicitly_set = "--mode" in sys.argv
    if args.presentation_mode in ("implicit", "both"):
        if mode_explicitly_set and args.mode == "scripted":
            parser.error(
                "Implicit presentation mode requires autonomous patient mode. "
                "Use --mode autonomous or omit --mode (it will be set automatically)."
            )
        # Auto-set mode to autonomous
        args.mode = "autonomous"

    # Disable colors if requested
    if args.no_color:
        Colors.disable()

    # Setup paths
    profiles_dir = Path(__file__).parent / "profiles"
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create run folder with timestamp and model names
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_names = "_".join(
        m.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")
        for m in args.assistant
    )
    turns_str = "_".join(str(t) for t in args.turns)
    run_dir = base_output_dir / f"run_{run_timestamp}_{model_names}_turns_{turns_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get profiles to run
    if args.all_profiles:
        profile_ids = get_available_profiles(profiles_dir)
    else:
        profile_ids = args.profile

    # Determine presentation modes to run
    if args.presentation_mode == "both":
        presentation_modes = ["explicit", "implicit"]
    else:
        presentation_modes = [args.presentation_mode]

    # Build experiment matrix
    experiments = []
    for profile_id in profile_ids:
        for assistant in args.assistant:
            for turns in args.turns:
                for pres_mode in presentation_modes:
                    experiments.append({
                        "profile_id": profile_id,
                        "assistant": assistant,
                        "turns": turns,
                        "presentation_mode": pres_mode,
                    })

    # Print configuration
    print(f"\n{Colors.HEADER}{'='*60}")
    print("DYNAMIC MANIA-BENCH")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n{Colors.BOLD}Configuration:{Colors.RESET}")
    print(f"  Mode:            {args.mode}")
    print(f"  Presentation:    {args.presentation_mode}")
    print(f"  Profiles:        {', '.join(profile_ids)}")
    print(f"  Assistants:      {', '.join(args.assistant)}")
    print(f"  Patient model:   {args.patient_model}")
    if args.mode == "autonomous":
        print(f"  Planner model:   {args.planner_model}")
        print(f"  Controller model: {args.controller_model}")
    print(f"  Judge model:     {args.judge_model}")
    print(f"  Turn counts:     {', '.join(map(str, args.turns))}")
    print(f"  Prompt style:    {args.style}")
    if args.mode == "scripted":
        print(f"  Schedule:        {args.schedule}")
    print(f"  DCS/HES/SIS:     {args.scoring_mode}")
    multi_judge_display = "Disabled" if args.skip_multi_judge else (args.multi_judge_model or args.judge_model)
    print(f"  Multi-dim judge: {multi_judge_display}")
    print(f"  Marker detection: {args.marker_detection}")
    print(f"  Topic tracking:  {'Disabled' if args.no_topic_tracking else 'Enabled'}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"\n{'='*60}\n")

    # Run experiments
    results = []
    for i, exp in enumerate(experiments):
        profile_id = exp["profile_id"]
        assistant = exp["assistant"]
        turns = exp["turns"]
        pres_mode = exp.get("presentation_mode", "explicit")

        print(f"{Colors.BOLD}[{i+1}/{len(experiments)}] Running: {profile_id} [{pres_mode}] + {assistant} ({turns} turns){Colors.RESET}")

        try:
            profile = get_profile(profile_id, profiles_dir, presentation_mode=pres_mode)

            result = await run_single_experiment(
                patient_model_name=args.patient_model,
                assistant_model_name=assistant,
                judge_model_name=args.judge_model,
                profile=profile,
                num_turns=turns,
                prompt_style=args.style,
                output_dir=run_dir,
                scoring_mode=args.scoring_mode,
                verbose=not args.quiet,
                marker_detection_mode=args.marker_detection,
                marker_detection_model_name=args.marker_model,
                schedule_name=args.schedule,
                enable_topic_tracking=not args.no_topic_tracking,
                topic_tracking_model_name=args.topic_model,
                mode=args.mode,
                planner_model_name=args.planner_model,
                controller_model_name=args.controller_model,
                multi_judge_model_name=args.multi_judge_model,
                skip_multi_judge=args.skip_multi_judge,
            )

            results.append({
                "experiment": exp,
                "result": result,
                "success": True
            })

            if result.summary_metrics:
                metrics = result.summary_metrics
                print(f"\n{Colors.GREEN}Metrics:{Colors.RESET}")
                print(f"  Avg DCS: {metrics.get('avg_dcs', 'N/A')}")
                print(f"  Avg HES: {metrics.get('avg_hes', 'N/A')}")
                print(f"  Total SIS: {metrics.get('total_sis', 'N/A')}")

            # Print multi-dimensional scores if available
            if result.multi_dimensional_scores:
                md_scores = result.multi_dimensional_scores.get("scores", {})
                if md_scores:
                    print(f"\n{Colors.GREEN}Multi-Dimensional Scores:{Colors.RESET}")
                    for dim, score in sorted(md_scores.items()):
                        print(f"  {dim}: {score}/10")
                    num_citations = len(result.multi_dimensional_scores.get("citations", []))
                    print(f"  ({num_citations} citations)")

            # Print cache stats if available
            usage = result.usage_stats
            for role in ["patient", "assistant", "judge"]:
                if role not in usage:
                    continue
                role_usage = usage[role]
                # Autonomous patient nests stats under "voice"/"controller"/"planner"
                if "voice" in role_usage:
                    cache_read = sum(role_usage[k].get("total_cache_read_tokens", 0) for k in role_usage if isinstance(role_usage[k], dict))
                    cache_write = sum(role_usage[k].get("total_cache_write_tokens", 0) for k in role_usage if isinstance(role_usage[k], dict))
                    total_input = sum(role_usage[k].get("total_input_tokens", 0) for k in role_usage if isinstance(role_usage[k], dict))
                else:
                    cache_read = role_usage.get("total_cache_read_tokens", 0)
                    cache_write = role_usage.get("total_cache_write_tokens", 0)
                    total_input = role_usage.get("total_input_tokens", 0)
                if cache_read > 0 or cache_write > 0:
                    total = total_input + cache_read + cache_write
                    savings_pct = (cache_read / total * 100) if total > 0 else 0
                    print(f"  {Colors.CYAN}{role} cache: {cache_read:,} read, {cache_write:,} write ({savings_pct:.0f}% cached){Colors.RESET}")

            # Print estimated cost
            cost = result.estimated_cost or result._compute_cost()
            if cost.get("total", 0) > 0:
                print(f"\n{Colors.GREEN}Estimated cost:{Colors.RESET}")
                for k, v in cost.items():
                    if k in ("total", "patient_breakdown"):
                        continue
                    if isinstance(v, (int, float)) and v > 0:
                        print(f"  {k}: ${v:.4f}")
                breakdown = cost.get("patient_breakdown")
                if breakdown:
                    parts = ", ".join(f"{k}: ${v:.4f}" for k, v in breakdown.items() if v > 0)
                    print(f"    ({parts})")
                print(f"  {Colors.BOLD}total: ${cost['total']:.4f}{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            results.append({
                "experiment": exp,
                "error": str(e),
                "success": False
            })

        print()

    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    # Save run configuration
    run_config = {
        "run_id": run_timestamp,
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "presentation_mode": args.presentation_mode,
        "profiles": profile_ids,
        "assistants": args.assistant,
        "patient_model": args.patient_model,
        "judge_model": args.judge_model,
        "turn_counts": args.turns,
        "prompt_style": args.style,
        "schedule": args.schedule if args.mode == "scripted" else "autonomous",
        "scoring_mode": args.scoring_mode,
        "multi_dimensional_scoring": not args.skip_multi_judge,
        "multi_judge_model": args.multi_judge_model or args.judge_model if not args.skip_multi_judge else None,
        "marker_detection": args.marker_detection,
        "topic_tracking_enabled": not args.no_topic_tracking,
        "total_experiments": len(experiments),
        "successful": successful,
        "failed": failed,
    }
    if args.mode == "autonomous":
        run_config["planner_model"] = args.planner_model
        run_config["controller_model"] = args.controller_model
    import json
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{Colors.HEADER}{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  Successful: {Colors.GREEN}{successful}{Colors.RESET}")
    if failed > 0:
        print(f"  Failed: {Colors.RED}{failed}{Colors.RESET}")
    print(f"  Results saved to: {run_dir}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
