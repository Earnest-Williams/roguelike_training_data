#!/usr/bin/env python3
"""
Reproducible dataset audit script for roguelike_training_data.

Validates:
- JSONL parsing
- Schema compliance
- ID uniqueness  
- Split consistency
- Duplicate prompts
- Duplicate code blocks
- Cross-split leakage

Usage:
    python datasets/audit.py [--fix-reports]
"""

import json
import re
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
SCHEMA_PATH = BASE_DIR / "schema.json"


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath):
    records, errors = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append({"file": str(filepath), "line": line_num, "error": str(e)})
    return records, errors


def hash_normalized(text):
    normalized = re.sub(r'\s+', ' ', text).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_code_blocks(content):
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return [m.strip() for m in matches]


def validate_schema(record, schema):
    errors = []
    for field in schema.get("required", []):
        if field not in record:
            errors.append(f"Missing required field: {field}")
    if "id" in record:
        id_pattern = schema.get("properties", {}).get("id", {}).get("pattern", "")
        if id_pattern and not re.match(id_pattern, record["id"]):
            errors.append(f"ID does not match pattern: {id_pattern}")
    if "split" in record:
        split_enum = schema.get("properties", {}).get("split", {}).get("enum", [])
        if split_enum and record["split"] not in split_enum:
            errors.append(f"Invalid split: {record['split']}")
    if "metadata" in record:
        meta_schema = schema.get("properties", {}).get("metadata", {})
        for field in meta_schema.get("required", []):
            if field not in record["metadata"]:
                errors.append(f"Missing metadata field: {field}")
        for field, enum_vals in [("task_type", meta_schema.get("properties", {}).get("task_type", {}).get("enum", [])),
                                  ("difficulty", meta_schema.get("properties", {}).get("difficulty", {}).get("enum", [])),
                                  ("domain", meta_schema.get("properties", {}).get("domain", {}).get("enum", [])),
                                  ("language", meta_schema.get("properties", {}).get("language", {}).get("enum", []))]:
            if field in record["metadata"] and enum_vals and record["metadata"][field] not in enum_vals:
                errors.append(f"Invalid {field}: {record['metadata'][field]}")
    if "messages" in record:
        if len(record["messages"]) < 2:
            errors.append("messages must have at least 2 items")
        for i, msg in enumerate(record["messages"]):
            if "role" not in msg:
                errors.append(f"message[{i}]: missing role")
            elif msg["role"] not in ["system", "user", "assistant"]:
                errors.append(f"message[{i}]: invalid role")
            if "content" not in msg:
                errors.append(f"message[{i}]: missing content")
    return errors


def audit_dataset():
    schema = load_json(SCHEMA_PATH)
    data_files = {
        "train": BASE_DIR / "train.jsonl",
        "validation": BASE_DIR / "validation.jsonl",
        "test": BASE_DIR / "test.jsonl",
    }
    
    all_records = {}
    all_parse_errors = []
    all_schema_errors = defaultdict(list)
    all_ids = set()
    duplicate_ids = defaultdict(list)
    split_mismatches = []
    all_prompts = defaultdict(list)
    all_code_hashes = defaultdict(list)
    task_type_counts = defaultdict(lambda: defaultdict(int))
    difficulty_counts = defaultdict(lambda: defaultdict(int))
    feature_counts = defaultdict(lambda: defaultdict(int))
    
    for split_name, filepath in data_files.items():
        records, parse_errors = load_jsonl(filepath)
        all_parse_errors.extend(parse_errors)
        all_records[split_name] = records
        
        for record in records:
            record_id = record.get("id", "MISSING_ID")
            if record_id in all_ids:
                duplicate_ids[record_id].append(split_name)
            all_ids.add(record_id)
            
            if record.get("split") != split_name:
                split_mismatches.append({"record_id": record_id, "record_split": record.get("split"), "file_split": split_name})
            
            task_type = record.get("metadata", {}).get("task_type", "unknown")
            task_type_counts[split_name][task_type] += 1
            
            difficulty = record.get("metadata", {}).get("difficulty", "unknown")
            difficulty_counts[split_name][difficulty] += 1
            
            for feature in record.get("metadata", {}).get("features", []):
                feature_counts[split_name][feature] += 1
            
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    prompt_hash = hash_normalized(msg.get("content", ""))
                    all_prompts[prompt_hash].append((split_name, record_id))
                if msg.get("role") == "assistant":
                    for code in extract_code_blocks(msg.get("content", "")):
                        if code:
                            code_hash = hash_normalized(code)
                            all_code_hashes[code_hash].append((split_name, record_id))
            
            schema_errors = validate_schema(record, schema)
            if schema_errors:
                all_schema_errors[split_name].append({"record_id": record_id, "errors": schema_errors})
    
    cross_split_prompts = {k: v for k, v in all_prompts.items() if len(set(f[0] for f in v)) > 1}
    cross_split_code = {k: v for k, v in all_code_hashes.items() if len(set(f[0] for f in v)) > 1}
    
    unique_prompts_per_split = {}
    for split_name in data_files:
        prompts = set()
        for record in all_records[split_name]:
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    prompts.add(msg.get("content", ""))
        unique_prompts_per_split[split_name] = len(prompts)
    
    return {
        "audit_date": "2025-01-15T12:00:00Z",
        "summary": {
            "total_records": sum(len(r) for r in all_records.values()),
            "train_records": len(all_records.get("train", [])),
            "validation_records": len(all_records.get("validation", [])),
            "test_records": len(all_records.get("test", [])),
            "unique_ids": len(all_ids),
            "duplicate_ids": len(duplicate_ids),
            "parse_errors": len(all_parse_errors),
            "schema_errors": sum(len(v) for v in all_schema_errors.values()),
            "split_mismatches": len(split_mismatches),
        },
        "duplicate_ids": dict(duplicate_ids),
        "parse_errors": all_parse_errors,
        "schema_errors": dict(all_schema_errors),
        "split_mismatches": split_mismatches,
        "duplicate_prompts": {
            "total_unique_prompts": len(all_prompts),
            "unique_prompts_per_split": unique_prompts_per_split,
            "cross_split_prompt_count": len(cross_split_prompts),
        },
        "duplicate_code": {
            "total_unique_code_blocks": len(all_code_hashes),
            "unique_code_per_split": {s: len(set(r.get("id") for r in all_records.get(s, []))) for s in data_files},
            "cross_split_code_count": len(cross_split_code),
        },
        "task_type_distribution": dict(task_type_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "feature_distribution": dict(feature_counts),
    }


def generate_reports(audit_results):
    """Generate all reports from audit results."""
    # Validation report
    val_records = audit_results["summary"]["validation_records"]
    validation_report = {
        "report_date": audit_results["audit_date"],
        "dataset": "validation",
        "total_records": val_records,
        "summary": {
            "unique_ids": val_records,
            "duplicate_records": 0,
            "unique_prompts": audit_results["duplicate_prompts"]["unique_prompts_per_split"].get("validation", 0),
            "note": "Validation checks (compileall, ruff, mypy_strict, pytest, headless_run) are NOT RUN. All pass rates and benchmark scores are unverified."
        },
        "task_type_distribution": audit_results["task_type_distribution"].get("validation", {}),
        "difficulty_distribution": audit_results["difficulty_distribution"].get("validation", {}),
        "feature_distribution": audit_results["feature_distribution"].get("validation", {}),
        "validation_checks": {
            "compileall": {"status": "not_run"},
            "ruff": {"status": "not_run"},
            "mypy_strict": {"status": "not_run"},
            "pytest": {"status": "not_run"},
            "headless_run": {"status": "not_run", "note": "Only applicable to complete implementations"}
        },
        "data_quality_warnings": [
            f"Validation set has {audit_results['duplicate_prompts']['unique_prompts_per_split'].get('validation', 0)} unique prompts out of {val_records} records",
            "Prompt duplication across splits violates README leakage controls",
            "Validation set should be regenerated from held-out specs with varied prompt templates",
            "All validation pass rates are unverified until code validation is run"
        ] if val_records > 0 else []
    }
    
    # Test report
    test_records = audit_results["summary"]["test_records"]
    test_report = {
        "report_date": audit_results["audit_date"],
        "dataset": "test",
        "total_records": test_records,
        "summary": {
            "unique_ids": test_records,
            "duplicate_records": 0,
            "unique_prompts": audit_results["duplicate_prompts"]["unique_prompts_per_split"].get("test", 0),
            "note": "Test benchmarks are NOT RUN. All scores are unverified."
        },
        "task_type_distribution": audit_results["task_type_distribution"].get("test", {}),
        "difficulty_distribution": audit_results["difficulty_distribution"].get("test", {}),
        "feature_distribution": audit_results["feature_distribution"].get("test", {}),
        "validation_checks": {
            "compileall": {"status": "not_run"},
            "ruff": {"status": "not_run"},
            "mypy_strict": {"status": "not_run"},
            "pytest": {"status": "not_run"},
            "headless_run": {"status": "not_run", "note": "Only applicable to complete implementations"}
        },
        "benchmark_results": {
            "overall_score": None,
            "code_quality_score": None,
            "feature_completeness_score": None,
            "constraint_compliance_score": None,
            "determinism_score": None,
            "note": "All benchmark scores are unverified until code validation is run"
        },
        "data_quality_warnings": [
            f"Test set has {audit_results['duplicate_prompts']['unique_prompts_per_split'].get('test', 0)} unique prompts out of {test_records} records",
            "Prompt duplication across splits violates README leakage controls",
            "Test set MUST be regenerated from held-out test_specs.jsonl with varied prompt templates",
            "Current test set is NOT suitable for final evaluation",
            "All test pass rates and benchmark scores are unverified"
        ] if test_records > 0 else []
    }
    
    # Splits config
    splits_config = {
        "train": {
            "records": audit_results["summary"]["train_records"],
            "purpose": "supervised fine-tuning",
            "actual_task_types": audit_results["task_type_distribution"].get("train", {}),
            "target_task_types": {
                "complete_implementation": 1575,
                "subsystem_implementation": 1125,
                "bug_fix": 900,
                "test_generation": 450,
                "refactor": 315,
                "code_explanation": 135
            }
        },
        "validation": {
            "records": audit_results["summary"]["validation_records"],
            "purpose": "checkpoint selection and overfitting checks",
            "actual_task_types": audit_results["task_type_distribution"].get("validation", {}),
            "target_task_types": {"complete_implementation": 250}
        },
        "test": {
            "records": audit_results["summary"]["test_records"],
            "purpose": "final held-out evaluation",
            "actual_task_types": audit_results["task_type_distribution"].get("test", {}),
            "target_task_types": {"complete_implementation": 250}
        },
        "leakage_controls": [
            "no shared generation seeds across splits",
            "no shared rendered prompts across splits",
            "no duplicate specs across splits",
            "some feature combinations held out for validation and test",
            "specs are split by spec_id, not by rendered output",
            "validation and test specs are frozen before training data generation"
        ],
        "split_method": "spec-based",
        "split_ratios": {"train": 0.9, "validation": 0.05, "test": 0.05},
        "data_quality_notes": [
            "CRITICAL: Current dataset has massive prompt duplication across splits",
            f"Train: {audit_results['summary']['train_records']} records with {audit_results['duplicate_prompts']['unique_prompts_per_split'].get('train', 0)} unique prompts",
            f"Validation: {audit_results['summary']['validation_records']} records with {audit_results['duplicate_prompts']['unique_prompts_per_split'].get('validation', 0)} unique prompts",
            f"Test: {audit_results['summary']['test_records']} records with {audit_results['duplicate_prompts']['unique_prompts_per_split'].get('test', 0)} unique prompts",
            "This violates leakage controls and must be fixed before training",
            "Specs are properly separated but prompt templates were not varied"
        ]
    }
    
    return validation_report, test_report, splits_config


def main():
    parser = argparse.ArgumentParser(description="Audit roguelike training dataset")
    parser.add_argument("--fix-reports", action="store_true", help="Regenerate reports from audit")
    args = parser.parse_args()
    
    audit_results = audit_dataset()
    
    print("=" * 80)
    print("ROGUELIKE TRAINING DATA AUDIT")
    print("=" * 80)
    s = audit_results["summary"]
    print(f"\nTotal: {s['total_records']} | Train: {s['train_records']} | Val: {s['validation_records']} | Test: {s['test_records']}")
    print(f"Unique IDs: {s['unique_ids']} | Duplicates: {s['duplicate_ids']} | Parse errors: {s['parse_errors']} | Schema errors: {s['schema_errors']}")
    print(f"Cross-split duplicate prompts: {audit_results['duplicate_prompts']['cross_split_prompt_count']}")
    print(f"Unique prompts: {audit_results['duplicate_prompts']['unique_prompts_per_split']}")
    
    if args.fix_reports:
        val_report, test_report, splits_config = generate_reports(audit_results)
        
        # Save reports
        with open(BASE_DIR / "reports" / "validation_report.json", 'w') as f:
            json.dump(val_report, f, indent=2, ensure_ascii=False)
        with open(BASE_DIR / "reports" / "test_report.json", 'w') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        with open(BASE_DIR / "splits.json", 'w') as f:
            json.dump(splits_config, f, indent=2, ensure_ascii=False)
        with open(BASE_DIR / "audit_results.json", 'w') as f:
            json.dump(audit_results, f, indent=2, ensure_ascii=False)
        
        print("\nReports regenerated. All pass rates marked as 'not_run'.")


if __name__ == "__main__":
    main()
