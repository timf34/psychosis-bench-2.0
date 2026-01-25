#!/usr/bin/env python3
"""
Dynamic Mania-Bench: Benchmarking LLM responses to manic/psychotic presentations.

This CLI tool runs multi-turn conversations between a simulated patient
with manic symptoms and an LLM assistant under test, evaluating the
assistant's response using DCS/HES/SIS metrics.

Usage:
    # Run with defaults (10 turns, pattern_decoder profile, GPT-4o assistant):
    python main.py

    # Specify turn count and profile:
    python main.py --turns 20 --profile pattern_decoder

    # Test multiple assistants:
    python main.py --assistant gpt-4o gpt-5-chat-latest claude

    # Run all profiles:
    python main.py --all-profiles

    # Batch experiment (all profiles x all assistants x multiple turn counts):
    python main.py --batch --turns 10 20 --assistant gpt-4o claude
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.intensity_controller import (
    IntensityController,
    get_standard_schedule,
    load_schedule_from_yaml,
)
from src.patient import PatientSimulator, PatientProfile, load_profile, load_all_profiles
from src.patient.profiles import get_builtin_profile
from src.patient.simulator import SimulatorConfig
from src.assistant import create_assistant, AssistantConfig
from src.evaluation import Scorer, JudgeLLM, calculate_conversation_metrics
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


def get_profile(profile_id: str, profiles_dir: Path) -> PatientProfile:
    """Load a patient profile by ID."""
    # Try loading from YAML file
    yaml_path = profiles_dir / f"{profile_id}.yaml"
    if yaml_path.exists():
        return load_profile(yaml_path)

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


def run_single_experiment(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    profile: PatientProfile,
    assistant_model: str,
    num_turns: int,
    prompt_style: str,
    output_dir: Path,
    enable_scoring: bool = True,
    judge_model: str = "gpt-4o-mini",
    verbose: bool = True
):
    """Run a single conversation experiment."""
    # Get intensity schedule
    schedule = get_standard_schedule()

    # Create intensity controller
    intensity_controller = IntensityController(
        total_turns=num_turns,
        schedule=schedule
    )

    # Create patient simulator
    patient_config = SimulatorConfig(model="gpt-4o", max_tokens=300)
    patient_simulator = PatientSimulator(
        client=openai_client,
        profile=profile,
        intensity_controller=intensity_controller,
        config=patient_config
    )

    # Create assistant under test
    if assistant_model.startswith("claude"):
        assistant = create_assistant(
            provider="anthropic",
            model="claude-sonnet-4-20250514" if assistant_model == "claude" else assistant_model,
            anthropic_client=anthropic_client,
            prompt_style=prompt_style
        )
    else:
        assistant = create_assistant(
            provider="openai",
            model=assistant_model,
            openai_client=openai_client,
            prompt_style=prompt_style
        )

    # Create scorer (if enabled)
    scorer = None
    if enable_scoring:
        judge_llm = JudgeLLM(
            provider="openai",
            model=judge_model,
            openai_client=openai_client
        )
        scorer = Scorer(judge_llm=judge_llm)

    # Create conversation config
    conv_config = ConversationConfig(
        num_turns=num_turns,
        save_incrementally=True,
        output_dir=output_dir,
        enable_scoring=enable_scoring,
        verbose=verbose
    )

    # Create and run conversation
    runner = ConversationRunner(
        patient_simulator=patient_simulator,
        assistant=assistant,
        intensity_controller=intensity_controller,
        scorer=scorer,
        config=conv_config
    )

    result = runner.run(assistant_prompt_style=prompt_style)
    return result


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dynamic Mania-Bench: Benchmark LLM responses to manic presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with defaults
  python main.py --turns 20 --profile pattern_decoder
  python main.py --assistant gpt-4o claude --turns 10 20
  python main.py --all-profiles --assistant gpt-5-chat-latest
  python main.py --batch --all-profiles --assistant gpt-4o claude --turns 10 20

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
        """
    )

    # Core arguments
    parser.add_argument("--turns", type=int, nargs="+", default=[10],
                        help="Number of conversation turns (default: 10)")
    parser.add_argument("--profile", type=str, nargs="+", default=["pattern_decoder"],
                        help="Patient profile ID(s) to use")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Run all available profiles")
    parser.add_argument("--assistant", type=str, nargs="+", default=["gpt-4o"],
                        help="Assistant model(s) to test")
    parser.add_argument("--style", choices=["baseline", "sycophantic", "professional", "none"],
                        default="baseline", help="Assistant prompt style")

    # Execution modes
    parser.add_argument("--batch", action="store_true",
                        help="Run all combinations of profiles x assistants x turns")

    # Scoring options
    parser.add_argument("--no-scoring", action="store_true",
                        help="Disable LLM-as-judge scoring (faster)")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Model to use as judge (default: gpt-4o-mini)")

    # API keys
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--anthropic-key", help="Anthropic API key")

    # Output
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        Colors.disable()

    # Get API keys
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    anthropic_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        print(f"{Colors.RED}Error: OpenAI API key required. Set OPENAI_API_KEY or use --openai-key{Colors.RESET}")
        sys.exit(1)

    # Check if we need Anthropic key
    needs_anthropic = any("claude" in a.lower() for a in args.assistant)
    if needs_anthropic and not anthropic_key:
        print(f"{Colors.RED}Error: Anthropic API key required for Claude models. Set ANTHROPIC_API_KEY or use --anthropic-key{Colors.RESET}")
        sys.exit(1)

    # Create clients
    openai_client = OpenAI(api_key=openai_key)
    anthropic_client = Anthropic(api_key=anthropic_key) if anthropic_key else None

    # Setup paths
    profiles_dir = Path(__file__).parent / "profiles"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get profiles to run
    if args.all_profiles:
        profile_ids = get_available_profiles(profiles_dir)
    else:
        profile_ids = args.profile

    # Build experiment matrix
    experiments = []
    for profile_id in profile_ids:
        for assistant in args.assistant:
            for turns in args.turns:
                experiments.append({
                    "profile_id": profile_id,
                    "assistant": assistant,
                    "turns": turns
                })

    # Print configuration
    print(f"\n{Colors.HEADER}{'='*60}")
    print("DYNAMIC MANIA-BENCH")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n{Colors.BOLD}Configuration:{Colors.RESET}")
    print(f"  Profiles:        {', '.join(profile_ids)}")
    print(f"  Assistants:      {', '.join(args.assistant)}")
    print(f"  Turn counts:     {', '.join(map(str, args.turns))}")
    print(f"  Prompt style:    {args.style}")
    print(f"  Scoring:         {'Enabled' if not args.no_scoring else 'Disabled'}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"\n{'='*60}\n")

    # Run experiments
    results = []
    for i, exp in enumerate(experiments):
        profile_id = exp["profile_id"]
        assistant = exp["assistant"]
        turns = exp["turns"]

        print(f"{Colors.BOLD}[{i+1}/{len(experiments)}] Running: {profile_id} + {assistant} ({turns} turns){Colors.RESET}")

        try:
            profile = get_profile(profile_id, profiles_dir)

            result = run_single_experiment(
                openai_client=openai_client,
                anthropic_client=anthropic_client,
                profile=profile,
                assistant_model=assistant,
                num_turns=turns,
                prompt_style=args.style,
                output_dir=output_dir,
                enable_scoring=not args.no_scoring,
                judge_model=args.judge_model,
                verbose=not args.quiet
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

        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            results.append({
                "experiment": exp,
                "error": str(e),
                "success": False
            })

        print()

    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{Colors.HEADER}{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  Successful: {Colors.GREEN}{successful}{Colors.RESET}")
    if failed > 0:
        print(f"  Failed: {Colors.RED}{failed}{Colors.RESET}")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
