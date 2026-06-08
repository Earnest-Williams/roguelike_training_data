# Datasets

This folder contains the JSONL datasets used to fine-tune and evaluate small LLMs on roguelike code.

The dataset is organized as chat-style instruction records. Each line in each `.jsonl` file is one complete training or evaluation example.

## Files

```text
datasets/
  README.md
  train.jsonl
  validation.jsonl
  test.jsonl
  schema.json
  splits.json
  specs/
    train_specs.jsonl
    validation_specs.jsonl
    test_specs.jsonl
  rejected/
    rejected.jsonl
  reports/
    validation_report.json
    test_report.json
````

## `train.jsonl`

The main supervised fine-tuning dataset.

This file contains the examples the model learns from directly. It includes a broad mix of roguelike programming tasks:

* complete single-file implementations
* subsystem implementations
* bug fixes
* test-writing tasks
* refactors
* code explanations

Most records are short-to-medium examples designed for Qwen3-4B, usually focused on one skill or a small combination of skills.

Example task families include:

```text
rooms and corridors
A* pathfinding
BFS flood fill
combat resolution
armor and damage clamping
inventory
save/load
field of view
monster AI
headless simulation
pytest tests
```

The training set intentionally contains repeated concepts with varied wording and constraints so the model can learn robust patterns rather than memorize one exact template.

## `validation.jsonl`

The validation set used during training.

This file is used to monitor whether the model is learning and whether training should continue. It should not be used for final scoring.

Validation examples are close to the training distribution but are generated from held-out specs, seeds, and prompt phrasings. This helps detect overfitting while still giving a realistic signal during training.

Use this file for:

* checking validation loss
* comparing checkpoints
* detecting style collapse
* detecting overfitting
* checking whether the model still follows formatting and dependency constraints

Do not tune the dataset repeatedly against the validation set until it effectively becomes part of training.

## `test.jsonl`

The final holdout evaluation set.

This file is reserved for final model evaluation. It should not be used during training, checkpoint selection, or prompt-template iteration.

The test set contains harder and more compositional tasks than the validation set. It is designed to check whether the model actually learned roguelike programming concepts instead of memorizing training patterns.

Examples may include:

```text
cellular automata cave generation with connectivity repair
shadowcasting field of view
save/load with entity IDs and inventory contents
monster AI with chase and wander states
bug fixes involving pathfinding through blocked tiles
refactors from single-file games into small packages
```

Treat this as the benchmark set.

## `schema.json`

The expected structure for each JSONL record.

Every dataset line should follow the same basic shape:

````json
{
  "id": "train_subsystem_000001",
  "split": "train",
  "metadata": {
    "domain": "roguelike",
    "language": "python",
    "difficulty": "intermediate",
    "task_type": "subsystem_implementation",
    "features": ["astar", "grid_pathfinding", "pytest"],
    "validation": {
      "compileall": true,
      "ruff": true,
      "mypy_strict": true,
      "pytest": true
    }
  },
  "messages": [
    {
      "role": "system",
      "content": "You are a precise Python coding assistant."
    },
    {
      "role": "user",
      "content": "Implement A* pathfinding for a grid-based roguelike."
    },
    {
      "role": "assistant",
      "content": "```python\n...\n```"
    }
  ]
}
````

The `messages` field is compatible with chat-style supervised fine-tuning. For Unsloth training, these messages are rendered through the Qwen3 chat template before tokenization.

## `splits.json`

Metadata describing how the dataset was split.

This file records the intended train, validation, and test proportions, plus the rules used to prevent leakage.

Example contents:

```json
{
  "train": {
    "records": 4500,
    "purpose": "supervised fine-tuning"
  },
  "validation": {
    "records": 250,
    "purpose": "checkpoint selection and overfitting checks"
  },
  "test": {
    "records": 250,
    "purpose": "final held-out evaluation"
  },
  "leakage_controls": [
    "no shared generation seeds across splits",
    "no shared rendered prompts across splits",
    "no duplicate specs across splits",
    "some feature combinations held out for validation and test"
  ]
}
```

Use this file to document the dataset split logic. It should make it clear that the split was not done by randomly shuffling near-duplicate examples.

## `specs/`

This directory contains the private or semi-private generation specifications used to create dataset records.

Specs describe what an example should contain before it is rendered into a natural-language prompt and assistant answer.

Example spec:

```json
{
  "spec_id": "astar_blocked_tiles_v1",
  "task_type": "subsystem_implementation",
  "difficulty": "intermediate",
  "features": ["astar", "blocked_tiles", "deterministic_neighbor_order"],
  "requirements": [
    "Use a Point dataclass",
    "Return path excluding start and including goal",
    "Return None when no path exists",
    "Include pytest tests"
  ]
}
```

### `specs/train_specs.jsonl`

Specs used to generate `train.jsonl`.

These specs may include many repeated concepts with varied constraints. They are safe to iterate on during dataset development.

### `specs/validation_specs.jsonl`

Specs used to generate `validation.jsonl`.

These should be held out from training. They should remain stable so validation metrics are comparable across training runs.

### `specs/test_specs.jsonl`

Specs used to generate `test.jsonl`.

These should be treated as final holdout specs. Avoid reading, editing, or optimizing against these during model development unless intentionally creating a new benchmark version.

## `rejected/rejected.jsonl`

Records that were generated but rejected by validation.

A generated sample should be rejected if it fails one or more quality checks, such as:

```text
python -m compileall
ruff check
mypy --strict
pytest
headless simulation
manual review
deduplication
```

Rejected records are kept for debugging the data generation pipeline. They should not be included in training unless they are intentionally converted into bug-fix examples.

Each rejected record should explain why it failed:

```json
{
  "id": "generated_000482",
  "task_type": "complete_implementation",
  "failed_checks": ["mypy_strict", "pytest"],
  "failure_reason": "missing return annotation and failing connectivity test",
  "source_spec_id": "rooms_corridors_connectivity_v2"
}
```

## `reports/`

This directory contains validation and evaluation reports produced by the dataset pipeline or model evaluation scripts.

### `reports/validation_report.json`

Summary of quality checks for `validation.jsonl`.

This may include:

* number of records
* task-type distribution
* difficulty distribution
* feature distribution
* static-check pass rates
* duplicate detection results
* average token length

### `reports/test_report.json`

Final benchmark results for the trained model.

This may include:

* compile success rate
* pytest success rate
* headless simulation success rate
* feature completion score
* hallucinated dependency rate
* formatting compliance
* exact pass/fail results by task type

## JSONL format

All `.jsonl` files use one JSON object per line.

Correct:

````jsonl
{"id":"train_000001","messages":[{"role":"user","content":"Write a map generator."},{"role":"assistant","content":"```python\n...\n```"}]}
{"id":"train_000002","messages":[{"role":"user","content":"Fix this combat bug."},{"role":"assistant","content":"The bug is...\n```python\n...\n```"}]}
````

Incorrect:

```json
[
  {
    "id": "train_000001"
  },
  {
    "id": "train_000002"
  }
]
```

The dataset loader expects newline-delimited JSON, not a single JSON array.

## Split policy

The dataset is split by underlying spec, not by rendered prompt.

Do not create multiple paraphrases of the same example and place one in training and one in test. That leaks the answer pattern and inflates evaluation scores.

The split should separate:

* generation specs
* random seeds
* prompt templates
* feature combinations
* generated code variants

Recommended split:

```text
train:       90%
validation:  5%
test:        5%
```

For a 5,000-record dataset:

```text
train.jsonl       4,500 records
validation.jsonl    250 records
test.jsonl          250 records
```

## Task types

Each record should have a `metadata.task_type` value.

Recommended task types:

```text
complete_implementation
subsystem_implementation
bug_fix
test_generation
refactor
code_explanation
```

Suggested distribution:

```text
complete_implementation: 35%
subsystem_implementation: 25%
bug_fix: 20%
test_generation: 10%
refactor: 7%
code_explanation: 3%
```

## Validation checks

Generated code should be accepted only after passing the relevant checks.

Minimum checks:

```bash
python -m compileall .
ruff check .
mypy --strict .
pytest
```

For complete game implementations, also run a headless simulation:

```bash
python game.py --headless --seed 123 --turns 100
```

or, for package-style samples:

```bash
python -m roguelike.main --headless --seed 123 --turns 100
```

Samples that fail validation should be moved to `rejected/rejected.jsonl`.

## Record IDs

Use stable, descriptive IDs.

Recommended format:

```text
train_complete_000001
train_subsystem_000001
train_bugfix_000001

val_complete_000001
val_subsystem_000001
val_bugfix_000001

test_complete_000001
test_subsystem_000001
test_bugfix_000001
```

IDs should not be reused after a record is changed substantially. Either create a new ID or update a version field.

## Metadata fields

Recommended metadata fields:

```json
{
  "domain": "roguelike",
  "language": "python",
  "difficulty": "beginner",
  "task_type": "complete_implementation",
  "architecture": "single_file",
  "features": ["rooms_and_corridors", "combat", "pytest"],
  "validation": {
    "compileall": true,
    "ruff": true,
    "mypy_strict": true,
    "pytest": true,
    "headless_run": true
  }
}
```

Useful optional fields:

```json
{
  "generator": {
    "spec_id": "rooms_corridors_combat_v1",
    "template_version": "0.2.0",
    "seed": 12345
  },
  "code_metrics": {
    "files": 1,
    "loc": 184,
    "classes": 6,
    "functions": 14
  }
}
```

## Training notes

For Unsloth SFT, load the JSONL files and render `messages` through the Qwen3 chat template.

Only train on assistant responses. The system and user messages should provide context, but the loss should be applied to the assistant completion.

The intended training behavior is:

```text
user gives constrained roguelike coding task
model emits typed, runnable Python
model includes tests when requested
model avoids unnecessary dependencies
model supports deterministic seeds and headless validation when requested
```

## Do not include

Avoid adding records that contain:

* broken code marked as valid
* unvalidated full implementations
* near-duplicate examples across splits
* hidden dependencies not mentioned in the prompt
* very long boilerplate-heavy repositories
* examples copied from existing tutorials or licensed projects
* records where the assistant ignores the user’s constraints

## Maintenance checklist

Before training:

```text
[ ] JSONL parses successfully
[ ] all records have unique IDs
[ ] all records have messages
[ ] all records have metadata.task_type
[ ] all code records pass validation
[ ] train/validation/test specs do not overlap
[ ] no generated prompt appears in multiple splits
[ ] rejected examples are excluded from train/validation/test
[ ] token lengths are within the target training context
[ ] final test set has not been used for checkpoint selection
```

```
```
