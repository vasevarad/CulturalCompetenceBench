# CC-Bench

`CC-Bench` is a benchmark for evaluating cultural competence, by checking whether healthcare responses accommodate culturally associated norms and values, when personas implicitly reveal their norm preferences and have partial adherence to cultural norms. It pairs culturally aligned or misaligned personas with filtered healthcare queries, then evaluates generated responses against culture-specific recommendations.

## Repository contents

```text
data/
	norm_values/       Current culture norm/value definitions (8 JSON files)
	personas/          Persona norm-adherence profiles and descriptions by culture
	queries/           Filtered benchmark queries
system_prompts/
	checklist_evaluation.txt  Shared checklist-evaluation system prompt
evaluations/
	checklists/        Follow/avoid recommendations by culture
	batch_checklisting_batched_respones.py
```

The benchmark currently covers Afghan, Burmese, Chinese, Maori, Midwest, Nepali, Northeast, and Vietnamese cultures. Each persona JSON marks norms as `Follow`, `Avoid`, or `Neutral`; the evaluator uses only the relevant `Follow` and `Avoid` norms for each query.

## Data flow

1. A response-generation process administers `data/queries/filtered_queries_final.csv` to a persona from `data/personas/<culture>/`.
2. Generated responses are stored as JSONL records. Each record must have a `custom_id` ending in `<culture>_<persona_id>_<question_id>` and a response at `response.body.choices[0].message.content`.
3. `evaluations/batch_checklisting_batched_respones.py` joins each response to its persona profile, filtered query, and culture checklist.
4. The script emits JSONL batch requests containing the checklist recommendations for a downstream judge model.

## Generate checklist prompts

Create an environment with Python 3.10+ and install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the evaluator from the repository root:

```bash
python3 evaluations/batch_checklisting_batched_respones.py \
	--responses /path/to/generated_responses.jsonl \
	--output /path/to/checklist_prompts.jsonl
```

The query, persona, and checklist paths default to the copies in this repository. They can be overridden with `--queries`, `--personas-dir`, and `--checklists-dir`.

The output is a batch-request JSONL file for a compatible chat-completions endpoint. The script prepares prompts; it does not make network requests itself.

## Evaluate through a chat API

`scripts/evaluate_responses_with_checklist.py` uses the same persona, query, and checklist selection logic, but sends each response directly to a chat-completions API. Set the API key in the environment and run:

```bash
export OPENAI_API_KEY="..."
python3 scripts/evaluate_responses_with_checklist.py \
	--responses /path/to/generated_responses.jsonl \
	--output /path/to/evaluations.jsonl \
	--model gpt-5.2
```

For an OpenAI-compatible provider, set `OPENAI_BASE_URL` or pass `--base-url`. Each output line contains the source `custom_id`, the selected recommendations, the model's evaluation, and the model name. Results are appended and flushed after each API call, so an interrupted run retains completed records.

The evaluator loads its system prompt from `system_prompts/checklist_evaluation.txt`. The batch prompt generator uses the same file; pass `--system-prompt` to the direct evaluator to test a prompt variant.

## Collect responses one at a time

`scripts/collect_responses.py` reproduces the conversation-context request construction from the original TogetherAI batch script, but sends one chat completion at a time. It writes each response immediately in the `response.body` format consumed by the checklist evaluator and skips completed `custom_id` values when rerun:

```bash
export TOGETHER_API_KEY="..."
python3 scripts/collect_responses.py \
	--conversations-dir /path/to/simulated_health_nonhealth_conversations \
	--output /path/to/responses.jsonl
```

The filtered query CSV, model, TogetherAI base URL, cultures, and temperature are configurable with CLI options. Use `--overwrite` to request records even when their IDs already occur in the output file.
