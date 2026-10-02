"""Prepare BFCL tasks and reproducible, disjoint train/eval/test splits."""

import json
import random
from pathlib import Path
from typing import Any, Optional

from bfcl_eval.constants.eval_config import PROMPT_PATH
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.utils import parse_test_category_argument, populate_test_cases_with_predefined_functions

TEST_FILE_MAPPING = {
    "simple": "BFCL_v4_simple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
    "parallel": "BFCL_v4_parallel.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "java": "BFCL_v4_java.json",
    "javascript": "BFCL_v4_javascript.json",
    "live_simple": "BFCL_v4_live_simple.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
    "live_parallel": "BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
    "live_irrelevance": "BFCL_v4_live_irrelevance.json",
    "live_relevance": "BFCL_v4_live_relevance.json",
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}


def bfcl_task_preprocess(
    test_categories: Optional[list[str]] = None,
    train_ratio: float = 0.5,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.4,
    random_seed: int = 42,
    output_dir: str = "",
    enable_shuffle: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Preprocess BFCL test cases and split into train / eval / test subsets.

    Args:
        test_categories: List of test categories or collection names ('all',
            'multi_turn', etc.). If None, process all categories.
        train_ratio: Fraction in [0, 1] for the train split.
        eval_ratio:  Fraction in [0, 1] for the eval split (mid-training eval).
        test_ratio:  Fraction in [0, 1] for the held-out test split.
            train_ratio + eval_ratio + test_ratio must sum to <= 1.0;
            any leftover fraction is dropped.
        random_seed: Seed for deterministic shuffling (default 42).
        enable_shuffle: Shuffle before splitting (default True).
        output_dir: Output directory; emits two files per category.
    Returns:
        Dict containing the three subsets: {'train': [...], 'eval': [...], 'test': [...]}
    """
    if not (0.0 <= train_ratio <= 1.0 and 0.0 <= eval_ratio <= 1.0 and 0.0 <= test_ratio <= 1.0):
        raise ValueError(
            f"split ratios must each lie in [0, 1]; got train={train_ratio}, eval={eval_ratio}, test={test_ratio}"
        )
    total = train_ratio + eval_ratio + test_ratio
    if total > 1.0 + 1e-9:
        raise ValueError(f"train_ratio + eval_ratio + test_ratio must be <= 1.0; got {total}")

    def load_selected_test_cases(categories: list[str]):
        all_test_entries_by_category = {}

        try:
            test_categories_resolved = parse_test_category_argument(categories)
        except Exception as e:
            print(f"Error: Invalid test categories - {e}")
            return {}

        print(f"Selected test categories: {test_categories_resolved}")

        for category in test_categories_resolved:
            if category in TEST_FILE_MAPPING:
                test_file_path = TEST_FILE_MAPPING[category]
                test_entries = load_file(PROMPT_PATH / test_file_path)
                print(f"Loaded {len(test_entries)} test cases from {category}")
                if category not in all_test_entries_by_category:
                    all_test_entries_by_category[category] = []
                all_test_entries_by_category[category].extend(test_entries)

        return all_test_entries_by_category

    random.seed(random_seed)

    if test_categories is None:
        test_categories = ["all"]

    all_test_cases_by_category = load_selected_test_cases(test_categories)

    if not all_test_cases_by_category:
        print("Warning: No test cases found")
        return {"train": [], "eval": [], "test": []}

    total_cases = sum(len(cases) for cases in all_test_cases_by_category.values())
    print(f"Loaded {total_cases} test cases in total across {len(all_test_cases_by_category)} categories")

    all_processed_cases = []
    processed_cases_by_category = {}

    for category, test_cases in all_test_cases_by_category.items():
        print(f"Processing category: {category}")

        category_processed_cases = populate_test_cases_with_predefined_functions(test_cases)
        processed_cases_by_category[category] = category_processed_cases
        all_processed_cases.extend(category_processed_cases)
        print(f"Successfully processed {len(category_processed_cases)} test cases for {category}")

    print(f"Successfully processed {len(all_processed_cases)} test cases in total")

    if enable_shuffle:
        random.shuffle(all_processed_cases)
    n_total = len(all_processed_cases)
    train_size = int(n_total * train_ratio)
    eval_size = int(n_total * eval_ratio)
    test_size = int(n_total * test_ratio)
    train_cases = all_processed_cases[:train_size]
    eval_cases = all_processed_cases[train_size : train_size + eval_size]
    test_cases = all_processed_cases[train_size + eval_size : train_size + eval_size + test_size]
    print(
        f"Data split complete (seed={random_seed}, shuffle={enable_shuffle}): "
        f"{len(train_cases)} train, {len(eval_cases)} eval, {len(test_cases)} test "
        f"(of {n_total} total)"
    )

    result = {"train": train_cases, "eval": eval_cases, "test": test_cases}

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        test_categories_str = "_".join(test_categories)

        full_jsonl_path = output_path / f"{test_categories_str}_processed.jsonl"
        with open(full_jsonl_path, "w", encoding="utf-8") as f:
            for case in all_processed_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"Full dataset saved to: {full_jsonl_path}")

        split_ids = {
            "train": [case.get("id", idx) for idx, case in enumerate(train_cases)],
            "eval": [case.get("id", idx) for idx, case in enumerate(eval_cases)],
            "test": [case.get("id", idx) for idx, case in enumerate(test_cases)],
        }

        split_ids_path = output_path / f"{test_categories_str}_split_ids.json"
        with open(split_ids_path, "w", encoding="utf-8") as f:
            json.dump(split_ids, f, ensure_ascii=False, indent=2)
        print(f"Split IDs saved to: {split_ids_path}")

    return result


if __name__ == "__main__":
    single_categories = [
        "all",
        "all_scoring",
        "multi_turn",
        "single_turn",
        "live",
        "non_live",
        "non_python",
        "python",
        "multi_turn_base",
    ]

    union_categories = [["multi_turn_base", "multi_turn_long_context"]]

    runs: list[list[str]] = [[name] for name in single_categories] + union_categories

    for categories in runs:
        result = bfcl_task_preprocess(
            test_categories=categories,
            train_ratio=0.5,
            eval_ratio=0.1,
            test_ratio=0.4,
            random_seed=42,
            enable_shuffle=True,
            output_dir="./bfcl_data",
        )

        print("-" * 50)
        print("Processing complete!")
        for key in ("train", "eval", "test"):
            if result[key]:
                print(f"{key.capitalize()} samples: {len(result[key])}")
