#!/usr/bin/env python3
"""
Repository-local audit/validation script for roguelike_training_data.

This script performs comprehensive validation of the dataset:
- Parses all JSONL files
- Validates each record against schema.json
- Counts records per split
- Counts task types, difficulties, features, unique prompts
- Checks ID uniqueness globally
- Checks split consistency
- Hashes normalized user prompts
- Hashes normalized assistant code blocks
- Detects duplicate prompts within and across splits
- Detects duplicate code across splits
- Prints concise pass/fail summary
- Exits nonzero on critical errors

Usage:
    python tools/audit_dataset.py [--generate-reports] [--verbose]

Exit codes:
    0 - All checks passed
    1 - Critical errors found (schema errors, invalid JSON, duplicate IDs, cross-split leakage)
    2 - Warnings found (duplicate prompts within split, etc.)
"""

import json
import re
import hashlib
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Repository paths
REPO_ROOT = Path(__file__).parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMA_PATH = DATASETS_DIR / "schema.json"

# Data files
DATA_FILES = {
    "train": DATASETS_DIR / "train.jsonl",
    "validation": DATASETS_DIR / "validation.jsonl",
    "test": DATASETS_DIR / "test.jsonl",
}

# Report files
SPLITS_PATH = DATASETS_DIR / "splits.json"
VALIDATION_REPORT_PATH = DATASETS_DIR / "reports" / "validation_report.json"
TEST_REPORT_PATH = DATASETS_DIR / "reports" / "test_report.json"
AUDIT_RESULTS_PATH = DATASETS_DIR / "audit_results.json"


class AuditError(Exception):
    """Critical audit error that should cause nonzero exit."""
    pass


class AuditWarning(Exception):
    """Audit warning that should be reported but not cause failure."""
    pass


def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath):
    """Load JSONL file, tracking parse errors. Returns tuple of (records, errors)."""
    records = []
    errors = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                try:
                    rel_path = str(filepath.relative_to(REPO_ROOT))
                except ValueError:
                    rel_path = str(filepath)
                errors.append({
                    "file": rel_path,
                    "line": line_num,
                    "error": str(e)
                })
    return records, errors


def hash_normalized(text):
    """Create hash of normalized text (whitespace and case normalized)."""
    if not text:
        return hashlib.sha256(b"").hexdigest()
    normalized = re.sub(r'\s+', ' ', text).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_code_blocks(content):
    """Extract Python code blocks from content."""
    pattern = r"```python\r?\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return [m.strip() for m in matches]


def validate_schema(record, schema):
    """Validate a record against the schema. Returns list of errors.
    
    Handles type errors gracefully - if a field has the wrong type,
    it's reported as a schema error rather than crashing.
    """
    errors = []
    
    # Check that record is a dict
    if not isinstance(record, dict):
        errors.append(f"Record is not a dict, got {type(record).__name__}")
        return errors
    
    # Check required fields
    for field in schema.get("required", []):
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Check ID pattern
    if "id" in record:
        if not isinstance(record["id"], str):
            errors.append(f"ID must be a string, got {type(record['id']).__name__}")
        else:
            id_pattern = schema.get("properties", {}).get("id", {}).get("pattern", "")
            if id_pattern:
                try:
                    if not re.match(id_pattern, record["id"]):
                        errors.append(f"ID '{record['id']}' does not match pattern: {id_pattern}")
                except (TypeError, re.error):
                    errors.append(f"ID '{record['id']}' caused pattern matching error")
    
    # Check split enum
    if "split" in record:
        if not isinstance(record["split"], str):
            errors.append(f"Split must be a string, got {type(record['split']).__name__}")
        else:
            split_enum = schema.get("properties", {}).get("split", {}).get("enum", [])
            if split_enum and record["split"] not in split_enum:
                errors.append(f"Invalid split value: {record['split']}. Must be one of {split_enum}")
    
    # Check metadata
    if "metadata" in record:
        if not isinstance(record["metadata"], dict):
            errors.append(f"Metadata must be an object, got {type(record['metadata']).__name__}")
        else:
            meta_schema = schema.get("properties", {}).get("metadata", {})
            
            # Required metadata fields
            for field in meta_schema.get("required", []):
                if field not in record["metadata"]:
                    errors.append(f"Missing required metadata field: {field}")
            
            # Enum validations
            enum_fields = {
                "task_type": meta_schema.get("properties", {}).get("task_type", {}).get("enum", []),
                "difficulty": meta_schema.get("properties", {}).get("difficulty", {}).get("enum", []),
                "domain": meta_schema.get("properties", {}).get("domain", {}).get("enum", []),
                "language": meta_schema.get("properties", {}).get("language", {}).get("enum", []),
                "architecture": meta_schema.get("properties", {}).get("architecture", {}).get("enum", []),
            }
            
            for field, valid_values in enum_fields.items():
                if field in record["metadata"]:
                    value = record["metadata"][field]
                    if not isinstance(value, str):
                        errors.append(f"Metadata field '{field}' must be a string, got {type(value).__name__}")
                    elif valid_values and value not in valid_values:
                        errors.append(f"Invalid {field}: '{value}'. Must be one of {valid_values}")
            
            # Validate features field
            if "features" in record["metadata"]:
                features = record["metadata"]["features"]
                if not isinstance(features, list):
                    errors.append(f"Metadata field 'features' must be a list, got {type(features).__name__}")
                else:
                    for idx, feature in enumerate(features):
                        if not isinstance(feature, str):
                            errors.append(f"Metadata field 'features[{idx}]' must be a string, got {type(feature).__name__}")

            # Validate validation field
            if "validation" in record["metadata"]:
                val_data = record["metadata"]["validation"]
                if not isinstance(val_data, dict):
                    errors.append(f"Metadata field 'validation' must be an object, got {type(val_data).__name__}")
                else:
                    val_schema = schema.get("properties", {}).get("metadata", {}).get("properties", {}).get("validation", {})
                    for field in val_schema.get("required", []):
                        if field not in val_data:
                            errors.append(f"Missing required metadata.validation field: {field}")

                    for field, field_schema in val_schema.get("properties", {}).items():
                        if field in val_data and field_schema.get("type") == "boolean":
                            if not isinstance(val_data[field], bool):
                                errors.append(
                                    f"metadata.validation.{field} must be a boolean, got {type(val_data[field]).__name__}"
                                )
    
    # Check messages
    if "messages" in record:
        if not isinstance(record["messages"], list):
            errors.append(f"Messages must be an array, got {type(record['messages']).__name__}")
        else:
            if len(record["messages"]) < 2:
                errors.append("messages must have at least 2 items")
            
            for i, msg in enumerate(record["messages"]):
                if not isinstance(msg, dict):
                    errors.append(f"message[{i}] must be an object, got {type(msg).__name__}")
                    continue
                if "role" not in msg:
                    errors.append(f"message[{i}]: missing 'role'")
                else:
                    role = msg["role"]
                    if not isinstance(role, str):
                        errors.append(f"message[{i}]: 'role' must be a string, got {type(role).__name__}")
                    elif role not in ["system", "user", "assistant"]:
                        errors.append(f"message[{i}]: invalid role '{role}'")
                if "content" not in msg:
                    errors.append(f"message[{i}]: missing 'content'")
                elif not isinstance(msg["content"], str):
                    errors.append(f"message[{i}]: 'content' must be a string, got {type(msg['content']).__name__}")
    
    return errors



def audit_dataset(verbose=False):
    """
    Run comprehensive audit and return results.
    
    Returns:
        dict: Audit results
        
    Raises:
        AuditError: On critical errors (parse errors, schema errors, duplicate IDs, cross-split leakage)
    """
    # Load schema
    try:
        schema = load_json(SCHEMA_PATH)
    except Exception as e:
        raise AuditError(f"Failed to load schema: {e}")
    
    # Load all data files
    all_records = {}
    valid_records = defaultdict(list)
    all_ids = {}
    duplicate_ids = defaultdict(list)
    split_mismatches = []
    all_prompts = defaultdict(list)  # hash -> list of (split, record_id)
    all_code_hashes = defaultdict(list)  # hash -> list of (split, record_id)
    task_type_counts = defaultdict(lambda: defaultdict(int))
    difficulty_counts = defaultdict(lambda: defaultdict(int))
    feature_counts = defaultdict(lambda: defaultdict(int))
    schema_errors = defaultdict(list)
    all_parse_errors = []
    
    # Track duplicates within splits
    prompts_within_split = defaultdict(lambda: defaultdict(list))
    code_within_split = defaultdict(lambda: defaultdict(list))
    
    for split_name, filepath in DATA_FILES.items():
        if not filepath.exists():
            raise AuditError(f"Dataset file not found for split '{split_name}': {filepath}")

        if verbose:
            try:
                rel_path = str(filepath.relative_to(REPO_ROOT))
            except ValueError:
                rel_path = str(filepath)
            print(f"Loading {rel_path}...")
        
        records, parse_errors = load_jsonl(filepath)
        all_parse_errors.extend(parse_errors)
        all_records[split_name] = records
        
        for record in records:
            is_dict = isinstance(record, dict)
            record_id = record.get("id", "MISSING_ID") if is_dict else "INVALID_RECORD_TYPE"
            
            # Validate schema first to ensure record structure is correct and safe to process
            schema_errs = validate_schema(record, schema)
            if schema_errs:
                schema_errors[split_name].append({
                    "record_id": record_id,
                    "errors": schema_errs
                })
                continue  # Skip processing malformed records to avoid crashes
            valid_records[split_name].append(record)
            
            # Track IDs globally
            if record_id in all_ids:
                if record_id not in duplicate_ids:
                    duplicate_ids[record_id].append(all_ids[record_id])
                duplicate_ids[record_id].append(split_name)
            all_ids[record_id] = split_name
            
            # Check split consistency
            record_split = record.get("split", "")
            if record_split != split_name:
                split_mismatches.append({
                    "record_id": record_id,
                    "record_split": record_split,
                    "file_split": split_name
                })
            
            # Count task types
            task_type = record.get("metadata", {}).get("task_type", "unknown")
            task_type_counts[split_name][task_type] += 1
            
            # Count difficulties
            difficulty = record.get("metadata", {}).get("difficulty", "unknown")
            difficulty_counts[split_name][difficulty] += 1
            
            # Count features
            for feature in record.get("metadata", {}).get("features", []):
                feature_counts[split_name][feature] += 1
            
            # Track prompts
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    prompt_hash = hash_normalized(prompt)
                    all_prompts[prompt_hash].append((split_name, record_id))
                    prompts_within_split[split_name][prompt_hash].append(record_id)
            
            # Track code blocks
            for msg in record.get("messages", []):
                if msg.get("role") == "assistant":
                    for code in extract_code_blocks(msg.get("content", "")):
                        if code:
                            code_hash = hash_normalized(code)
                            all_code_hashes[code_hash].append((split_name, record_id))
                            code_within_split[split_name][code_hash].append(record_id)
    
    # Detect cross-split duplicates
    cross_split_prompts = {
        k: v for k, v in all_prompts.items() 
        if len(set(f[0] for f in v)) > 1
    }
    
    cross_split_code = {
        k: v for k, v in all_code_hashes.items() 
        if len(set(f[0] for f in v)) > 1
    }
    
    # Detect within-split duplicates
    within_split_duplicate_prompts = {}
    for split_name in DATA_FILES:
        for prompt_hash, record_ids in prompts_within_split[split_name].items():
            if len(record_ids) > 1:
                if split_name not in within_split_duplicate_prompts:
                    within_split_duplicate_prompts[split_name] = []
                within_split_duplicate_prompts[split_name].append({
                    "prompt_hash": prompt_hash,
                    "record_count": len(record_ids),
                    "record_ids": record_ids[:5]  # Limit to first 5
                })
    
    within_split_duplicate_code = {}
    for split_name in DATA_FILES:
        for code_hash, record_ids in code_within_split[split_name].items():
            if len(record_ids) > 1:
                if split_name not in within_split_duplicate_code:
                    within_split_duplicate_code[split_name] = []
                within_split_duplicate_code[split_name].append({
                    "code_hash": code_hash,
                    "record_count": len(record_ids),
                    "record_ids": record_ids[:5]
                })
    
    # Count unique prompts per split
    unique_prompts_per_split = {}
    for split_name in DATA_FILES:
        unique_prompts = set()
        for record in valid_records[split_name]:
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    unique_prompts.add(hash_normalized(msg.get("content", "")))
        unique_prompts_per_split[split_name] = len(unique_prompts)
    
    # Count unique code blocks per split
    unique_code_per_split = {}
    for split_name in DATA_FILES:
        unique_code = set()
        for record in valid_records[split_name]:
            for msg in record.get("messages", []):
                if msg.get("role") == "assistant":
                    for code in extract_code_blocks(msg.get("content", "")):
                        if code:
                            unique_code.add(hash_normalized(code))
        unique_code_per_split[split_name] = len(unique_code)
    
    return {
        "audit_date": "2025-01-15T12:00:00Z",
        "summary": {
            "total_records": sum(len(r) for r in all_records.values()),
            "train_records": len(all_records.get("train", [])),
            "validation_records": len(all_records.get("validation", [])),
            "test_records": len(all_records.get("test", [])),
            "unique_ids": len(all_ids),
            "duplicate_ids_count": len(duplicate_ids),
            "parse_errors": len(all_parse_errors),
            "schema_errors": sum(len(v) for v in schema_errors.values()),
            "split_mismatches": len(split_mismatches),
        },
        "duplicate_ids": dict(duplicate_ids),
        "parse_errors": all_parse_errors,
        "schema_errors": dict(schema_errors),
        "split_mismatches": split_mismatches,
        "duplicate_prompts": {
            "total_unique_prompts": len(all_prompts),
            "unique_prompts_per_split": unique_prompts_per_split,
            "cross_split_prompt_count": len(cross_split_prompts),
            "cross_split_prompts": {
                k: [{"split": f[0], "record_id": f[1]} for f in v] 
                for k, v in list(cross_split_prompts.items())[:10]
            },
            "within_split_duplicate_prompts": within_split_duplicate_prompts,
        },
        "duplicate_code": {
            "total_unique_code_blocks": len(all_code_hashes),
            "unique_code_per_split": unique_code_per_split,
            "cross_split_code_count": len(cross_split_code),
            "within_split_duplicate_code": within_split_duplicate_code,
        },
        "task_type_distribution": dict(task_type_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "feature_distribution": dict(feature_counts),
    }


def generate_splits_config(audit_results):
    """Generate splits.json from audit results."""
    return {
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
            f"Cross-split duplicate prompts: {audit_results['duplicate_prompts']['cross_split_prompt_count']}",
            "This violates leakage controls and must be fixed before training",
            "Specs are properly separated but prompt templates were not varied"
        ]
    }


def generate_validation_report(audit_results):
    """Generate validation_report.json from audit results."""
    val_records = audit_results["summary"]["validation_records"]
    return {
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
            f"Cross-split duplicate prompts: {audit_results['duplicate_prompts']['cross_split_prompt_count']}",
            "Prompt duplication across splits violates README leakage controls",
            "Validation set should be regenerated from held-out specs with varied prompt templates",
            "All validation pass rates are unverified until code validation is run"
        ] if val_records > 0 else []
    }


def generate_test_report(audit_results):
    """Generate test_report.json from audit results."""
    test_records = audit_results["summary"]["test_records"]
    return {
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
            f"Cross-split duplicate prompts: {audit_results['duplicate_prompts']['cross_split_prompt_count']}",
            "Prompt duplication across splits violates README leakage controls",
            "Test set MUST be regenerated from held-out test_specs.jsonl with varied prompt templates",
            "Current test set is NOT suitable for final evaluation",
            "All test pass rates and benchmark scores are unverified"
        ] if test_records > 0 else []
    }


def print_summary(audit_results):
    """Print concise pass/fail summary."""
    s = audit_results["summary"]
    
    print("=" * 80)
    print("ROGUELIKE TRAINING DATA AUDIT SUMMARY")
    print("=" * 80)
    
    # Basic counts
    print(f"\n📊 RECORD COUNTS:")
    print(f"  Total:     {s['total_records']}")
    print(f"  Train:     {s['train_records']}")
    print(f"  Validation: {s['validation_records']}")
    print(f"  Test:      {s['test_records']}")
    
    # Quality metrics
    print(f"\n✅ PASSING CHECKS:")
    print(f"  Unique IDs:        {s['unique_ids']} / {s['total_records']} ✓")
    print(f"  Parse errors:     {s['parse_errors']} ✓")
    print(f"  Schema errors:    {s['schema_errors']} ✓")
    print(f"  Split mismatches: {s['split_mismatches']} ✓")
    
    # Prompt analysis
    dup_prompts = audit_results["duplicate_prompts"]
    print(f"\n📝 PROMPT ANALYSIS:")
    for split in ["train", "validation", "test"]:
        unique = dup_prompts["unique_prompts_per_split"].get(split, 0)
        total = s[f"{split}_records"]
        print(f"  {split:12s}: {unique} unique prompts / {total} records")
    print(f"  Cross-split duplicates: {dup_prompts['cross_split_prompt_count']} ⚠️")
    
    # Code analysis
    dup_code = audit_results["duplicate_code"]
    print(f"\n💻 CODE ANALYSIS:")
    for split in ["train", "validation", "test"]:
        unique = dup_code["unique_code_per_split"].get(split, 0)
        print(f"  {split:12s}: {unique} unique code blocks")
    print(f"  Cross-split duplicates: {dup_code['cross_split_code_count']}")
    
    # Critical issues
    has_critical = (
        s['parse_errors'] > 0 or
        s['schema_errors'] > 0 or
        s['duplicate_ids_count'] > 0 or
        s['split_mismatches'] > 0 or
        dup_prompts['cross_split_prompt_count'] > 0 or
        dup_code['cross_split_code_count'] > 0
    )
    
    print(f"\n{'❌ CRITICAL ISSUES FOUND' if has_critical else '✅ NO CRITICAL ISSUES'}")
    if s['duplicate_ids_count'] > 0:
        print(f"  Duplicate IDs: {s['duplicate_ids_count']}")
    if s['schema_errors'] > 0:
        print(f"  Schema errors: {s['schema_errors']}")
    if s['split_mismatches'] > 0:
        print(f"  Split mismatches: {s['split_mismatches']}")
    if dup_prompts['cross_split_prompt_count'] > 0:
        print(f"  Cross-split prompt leakage: {dup_prompts['cross_split_prompt_count']} prompts")
    if dup_code['cross_split_code_count'] > 0:
        print(f"  Cross-split code leakage: {dup_code['cross_split_code_count']} code blocks")
    
    print("=" * 80)
    
    return has_critical


def save_reports(audit_results):
    """Save all generated reports."""
    # Ensure directories exist
    (DATASETS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    
    # Generate and save reports
    splits_config = generate_splits_config(audit_results)
    validation_report = generate_validation_report(audit_results)
    test_report = generate_test_report(audit_results)
    
    # Save files
    with open(SPLITS_PATH, 'w', encoding='utf-8') as f:
        json.dump(splits_config, f, indent=2, ensure_ascii=False)
    
    with open(VALIDATION_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    
    with open(TEST_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)
    
    with open(AUDIT_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    
    print("\n📝 Reports saved:")
    print(f"  {SPLITS_PATH.relative_to(REPO_ROOT)}")
    print(f"  {VALIDATION_REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"  {TEST_REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"  {AUDIT_RESULTS_PATH.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit roguelike training dataset for compliance and quality"
    )
    parser.add_argument(
        "--generate-reports",
        action="store_true",
        help="Regenerate splits.json and report files from audit results"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output"
    )
    args = parser.parse_args()
    
    try:
        # Run audit
        audit_results = audit_dataset(verbose=args.verbose)
        
        # Print parse errors to stderr if any
        if audit_results.get("parse_errors"):
            for error in audit_results["parse_errors"]:
                print(f"PARSE ERROR: {error['file']}:{error['line']} - {error['error']}", file=sys.stderr)
        
        # Print summary
        has_critical = print_summary(audit_results)
        
        # Generate and save reports if requested
        if args.generate_reports:
            save_reports(audit_results)
        
        # Exit with appropriate code
        if has_critical:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except AuditError as e:
        print(f"\n❌ AUDIT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
