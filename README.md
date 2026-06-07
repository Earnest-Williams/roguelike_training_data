# Datasets

This folder contains the JSONL datasets used to fine-tune and evaluate small LLM's for roguelike code.

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
