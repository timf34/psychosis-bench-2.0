# Implementation: inspect_ai Port + Prompt Caching

## Step 1: Foundation
- [x] Add inspect-ai to requirements.txt
- [x] Create src/models.py shared helper
- [x] Verify inspect_ai imports and model initialization work

## Step 2: Port PatientSimulator
- [x] Replace OpenAI client with inspect_ai Model
- [x] Make generate_response async
- [x] Remove manual retry logic
- [x] Update token tracking to use ModelUsage (including cache fields)

## Step 3: Port AssistantAPI
- [x] Replace OpenAIAssistant + AnthropicAssistant with single Assistant class
- [x] Make generate_response async
- [x] Simplify create_assistant factory
- [x] Keep ASSISTANT_PROMPTS dict

## Step 4: Port JudgeLLM
- [x] Replace dual-provider JudgeLLM with inspect_ai Model
- [x] Make generate async
- [x] Update Scorer to be async

## Step 5: Async ConversationRunner
- [x] Make run() async
- [x] Make _run_turn() async
- [x] Make _score_turn() async
- [x] Port ConversationTopicTracker to use inspect_ai model
- [x] Port LLMMarkerDetector to use inspect_ai model

## Step 6: Update main.py
- [x] Async entry point with asyncio.run()
- [x] Accept inspect_ai model name format (provider/model)
- [x] Add --patient-model flag
- [x] Remove manual API key/client management
- [x] Update run_single_experiment

## Step 7: Prompt Caching
- [x] Split build_patient_prompt into stable/dynamic
- [x] Surface cache metrics in usage stats output
- [ ] Verify cache hits with Anthropic model (requires live API test)

## Verification
- [x] All imports work
- [x] Existing tests pass (35/35)
- [x] CLI --help output correct
- [ ] Basic run with OpenAI models works (requires API key)
- [ ] Cross-provider run works (Anthropic + OpenAI)
- [ ] Cache metrics visible in output
