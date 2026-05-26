"""
Moment-Video accuracy evaluation with an OpenRouter LLM judge.

Use this after an inference script has produced a CSV containing ModelAnswer.
The script combines the two benchmark evaluation paths:
1) closed multiple-choice rows use ClosedEvalPass when available;
2) open-ended rows are judged by an LLM through OpenRouter against the Answer
   reference;
3) aggregate accuracy is reported overall and by answer type, category,
   subcategory, and task type.

Usage example:
python scripts/eval_llm_judge_openrouter.py \
  --input-csv result/output/result.csv \
  --output-dir result/judge \
  --workers 8
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

csv.field_size_limit(10_000_000)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("accuracy_judge_csv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"judge_run_{ts}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Log file: %s", log_path)
    return logger


def build_judge_prompt(gold_answer: str, model_answer: str) -> str:
    return f"""
Evaluate whether the model answer is semantically consistent with the gold answer.

Judging rules:
1. Judge meaning, not wording.
2. Accept paraphrases, synonyms, abbreviations, and equivalent naming variants.
3. Mark as consistent if the model answer fully covers the gold answer's meaning, even with extra harmless details.
4. Mark as inconsistent if it misses a key point, contradicts the gold answer, or changes important facts (entity, action, order, quantity, identity, or existence).
5. Mark as inconsistent if the model answer is too vague to support the same meaning.
6. If uncertain, choose consistent only when a reasonable reader would conclude full semantic coverage.

Return JSON only, with no markdown and no extra text:
{{"is_consistent": true/false, "reason": "one-sentence explanation"}}

Gold answer:
{gold_answer}

Model answer:
{model_answer}
""".strip()


def parse_judge_response(text: str) -> Tuple[Optional[bool], str]:
    if not text:
        return None, "empty response"

    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            val = data.get("is_consistent")
            reason = str(data.get("reason", "")).strip()
            if isinstance(val, bool):
                return val, reason or "parsed from JSON"
    except json.JSONDecodeError:
        pass

    m = re.search(r'"is_consistent"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", "parsed with regex fallback"

    return None, "cannot parse judge response"


def judge_consistency_once(
    api_key: str,
    model: str,
    gold_answer: str,
    model_answer: str,
    timeout: int,
) -> Tuple[Optional[bool], str]:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    prompt = build_judge_prompt(gold_answer, model_answer)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict and fair evaluator. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        timeout=timeout,
    )

    text = ""
    if resp and resp.choices and resp.choices[0].message:
        text = (resp.choices[0].message.content or "").strip()
    return parse_judge_response(text)


def judge_row_with_retry(
    row_idx: int,
    api_key: str,
    model: str,
    gold_answer: str,
    model_answer: str,
    timeout: int,
    max_retries: int,
    retry_base_seconds: float,
) -> Dict[str, object]:
    start = time.perf_counter()
    last_reason = ""

    for attempt in range(1, max_retries + 1):
        try:
            val, reason = judge_consistency_once(
                api_key=api_key,
                model=model,
                gold_answer=gold_answer,
                model_answer=model_answer,
                timeout=timeout,
            )
            return {
                "row_idx": row_idx,
                "is_consistent": val,
                "reason": reason,
                "attempts": attempt,
                "elapsed": time.perf_counter() - start,
            }
        except Exception as exc:
            last_reason = f"judge error: {exc}"
            if attempt < max_retries:
                time.sleep(retry_base_seconds * (2 ** (attempt - 1)))

    return {
        "row_idx": row_idx,
        "is_consistent": None,
        "reason": last_reason or "judge failed",
        "attempts": max_retries,
        "elapsed": time.perf_counter() - start,
    }


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
        f.seek(0)
        delimiter = "\t" if ("\t" in header_line and "," not in header_line) else ","
        reader = csv.DictReader(
            f,
            delimiter=delimiter,
            quotechar='"',
            doublequote=True,
            strict=False,
        )
        raw_headers = list(reader.fieldnames or [])
        headers = [h.strip() if isinstance(h, str) else h for h in raw_headers]
        rows = []
        for raw_row in reader:
            normalized: Dict[str, str] = {}
            for key, value in raw_row.items():
                norm_key = key.strip() if isinstance(key, str) else key
                normalized[norm_key] = value if value is not None else ""
            rows.append(normalized)
    if not headers:
        raise ValueError(f"CSV has no header: {csv_path}")
    return headers, rows


def resolve_fields(headers: List[str], gold_field_arg: Optional[str], pred_field_arg: Optional[str]) -> Tuple[str, str]:
    if len(headers) < 2:
        raise ValueError("CSV must contain at least 2 columns.")

    if gold_field_arg:
        if gold_field_arg not in headers:
            raise ValueError(f"--gold-field not found in CSV header: {gold_field_arg}")
        gold_field = gold_field_arg
    elif "Answer" in headers:
        gold_field = "Answer"
    else:
        gold_field = headers[-2]

    if pred_field_arg:
        if pred_field_arg not in headers:
            raise ValueError(f"--pred-field not found in CSV header: {pred_field_arg}")
        pred_field = pred_field_arg
    elif "ModelAnswer" in headers:
        pred_field = "ModelAnswer"
    else:
        pred_field = headers[-1]

    if gold_field == pred_field:
        raise ValueError(f"gold/pred field resolved to the same column: {gold_field}")

    return gold_field, pred_field


def parse_row_range(total_data_rows: int, start_row: int, end_row: Optional[int]) -> Tuple[int, int]:
    if total_data_rows <= 0:
        raise ValueError("CSV has no data rows.")

    data_start_excel = max(2, start_row)
    data_end_excel = total_data_rows + 1 if end_row is None else min(end_row, total_data_rows + 1)

    if data_start_excel > data_end_excel:
        raise ValueError(f"Invalid row range: start={data_start_excel}, end={data_end_excel}")

    return data_start_excel - 2, data_end_excel - 2


def is_closed_question_row(row: Dict[str, str]) -> bool:
    question_stage = (row.get("QuestionStage") or "").strip().lower()
    answer_type = (row.get("AnswerType") or "").strip().lower()
    return question_stage == "closed" or answer_type == "closed"


def parse_closed_eval_pass(value: str) -> Optional[bool]:
    """Parse the boolean value from ClosedEvalPass."""
    text = (value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def write_csv_rows(csv_path: Path, headers: List[str], rows: List[Dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=headers,
            delimiter=",",
            quoting=csv.QUOTE_ALL,
            doublequote=True,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def init_metric_counter() -> Dict[str, int]:
    return {"evaluated": 0, "correct": 0, "undecidable": 0}


def update_metric_counter(container: Dict[str, Dict[str, int]], key: str, verdict: Optional[bool]) -> None:
    bucket = container.setdefault(key, init_metric_counter())
    bucket["evaluated"] += 1
    if verdict is None:
        bucket["undecidable"] += 1
    elif verdict:
        bucket["correct"] += 1


def update_group_stats(
    row: Dict[str, str],
    verdict: Optional[bool],
    answer_type_stats: Dict[str, Dict[str, int]],
    category_stats: Dict[str, Dict[str, int]],
    subclass_stats: Dict[str, Dict[str, int]],
    question_type_stats: Dict[str, Dict[str, int]],
) -> None:
    answer_type_key = "closed" if is_closed_question_row(row) else "open"
    category_key = (row.get("Category") or "").strip() or "UNKNOWN"
    subclass_key = (row.get("Subclass") or "").strip() or "UNKNOWN"
    raw_qt = (row.get("QuestionType") or "").strip().upper()
    if raw_qt == "REASONING":
        question_type_key = "TR"
    elif raw_qt in {"TO", "TC", "AD", "TR"}:
        question_type_key = raw_qt
    else:
        question_type_key = "UNKNOWN"

    update_metric_counter(answer_type_stats, answer_type_key, verdict)
    update_metric_counter(category_stats, category_key, verdict)
    update_metric_counter(subclass_stats, subclass_key, verdict)
    update_metric_counter(question_type_stats, question_type_key, verdict)


def format_metric_line(name: str, metric: Dict[str, int]) -> str:
    evaluated = metric["evaluated"]
    correct = metric["correct"]
    undecidable = metric["undecidable"]
    acc = (correct / evaluated) if evaluated > 0 else 0.0
    return (
        f"{name}: "
        f"Evaluated={evaluated}, Correct={correct}, Undecidable={undecidable}, Accuracy={acc:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Moment-Video accuracy with ClosedEvalPass and an OpenRouter LLM judge."
    )
    parser.add_argument(
        "--input-csv",
        default="result/output/result.csv",
        help="Path to result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/judge",
        help="Directory to save judged CSV and summary.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5-mini",
        help="Judge model on OpenRouter.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Single request timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry times when judge call fails.")
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=2.0,
        help="Exponential backoff base seconds.",
    )
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers for text judging.")
    parser.add_argument("--start-row", type=int, default=2, help="Excel-style start row, including header as row 1.")
    parser.add_argument("--end-row", type=int, default=None, help="Excel-style end row, inclusive.")
    parser.add_argument(
        "--gold-field",
        default=None,
        help="Gold answer column name. Default: Answer, else fallback to second-last column.",
    )
    parser.add_argument(
        "--pred-field",
        default=None,
        help="Model answer column name. Default: ModelAnswer, else fallback to last column.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set environment variable OPENROUTER_API_KEY first.")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    headers, rows = read_csv_rows(input_path)
    if len(headers) < 2:
        raise ValueError("CSV must contain at least 2 columns (gold answer + model answer).")
    start_idx, end_idx = parse_row_range(len(rows), args.start_row, args.end_row)

    gold_field, pred_field = resolve_fields(headers, args.gold_field, args.pred_field)
    logger.info("Using columns: gold=%s, pred=%s", gold_field, pred_field)
    logger.info("Using row range: Excel rows %d-%d", start_idx + 2, end_idx + 2)

    if "JudgeIsConsistent" not in headers:
        headers.append("JudgeIsConsistent")
    if "JudgeReason" not in headers:
        headers.append("JudgeReason")

    total_rows = len(rows)
    evaluated = 0
    correct = 0
    undecidable = 0
    answer_type_stats: Dict[str, Dict[str, int]] = {}
    category_stats: Dict[str, Dict[str, int]] = {}
    subclass_stats: Dict[str, Dict[str, int]] = {}
    question_type_stats: Dict[str, Dict[str, int]] = {}

    jobs = []
    for idx in range(start_idx, end_idx + 1):
        row = rows[idx]
        gold_text = (row.get(gold_field) or "").strip()
        pred_text = (row.get(pred_field) or "").strip()
        closed_eval_val = parse_closed_eval_pass(row.get("ClosedEvalPass", ""))
        is_closed = is_closed_question_row(row)

        if is_closed and closed_eval_val is not None:
            evaluated += 1
            row["JudgeIsConsistent"] = "1" if closed_eval_val else "0"
            row["JudgeReason"] = (row.get("ClosedEvalReason") or "").strip() or "used ClosedEvalPass"
            if closed_eval_val:
                correct += 1
            update_group_stats(
                row=row,
                verdict=closed_eval_val,
                answer_type_stats=answer_type_stats,
                category_stats=category_stats,
                subclass_stats=subclass_stats,
                question_type_stats=question_type_stats,
            )
            logger.info(
                "row=%d done verdict=%s source=ClosedEvalPass reason=%s",
                idx + 2,
                row["JudgeIsConsistent"],
                row["JudgeReason"],
            )
            continue

        if not gold_text:
            row["JudgeIsConsistent"] = "N/A"
            row["JudgeReason"] = "gold answer empty"
            continue

        evaluated += 1

        if not pred_text:
            row["JudgeIsConsistent"] = "0"
            row["JudgeReason"] = "model answer empty"
            update_group_stats(
                row=row,
                verdict=False,
                answer_type_stats=answer_type_stats,
                category_stats=category_stats,
                subclass_stats=subclass_stats,
                question_type_stats=question_type_stats,
            )
            logger.info("row=%d done verdict=0 attempts=0 elapsed=0.00s reason=model answer empty", idx + 2)
            continue

        jobs.append((idx, gold_text, pred_text))

    logger.info(
        "Prepared rows: total=%d, evaluated=%d, to_judge=%d, workers=%d",
        total_rows,
        evaluated,
        len(jobs),
        max(1, args.workers),
    )

    results: Dict[int, Dict[str, object]] = {}

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(jobs), desc="Judging", unit="row")
    elif len(jobs) > 0:
        logger.info("tqdm not installed, fallback to plain logging progress.")

    done_count = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                judge_row_with_retry,
                idx,
                api_key,
                args.model,
                gold_text,
                pred_text,
                args.timeout,
                args.max_retries,
                args.retry_base_seconds,
            ): idx
            for idx, gold_text, pred_text in jobs
        }

        for future in as_completed(future_map):
            result = future.result()
            idx = int(result["row_idx"])
            results[idx] = result
            done_count += 1

            val = result["is_consistent"]
            verdict_text = "N/A" if val is None else ("1" if val else "0")
            logger.info(
                "row=%d done verdict=%s attempts=%d elapsed=%.2fs reason=%s",
                idx + 2,
                verdict_text,
                int(result["attempts"]),
                float(result["elapsed"]),
                str(result["reason"]),
            )

            if progress is not None:
                progress.update(1)
            elif done_count % 10 == 0 or done_count == len(jobs):
                logger.info("Progress: %d/%d", done_count, len(jobs))

    if progress is not None:
        progress.close()

    for idx in sorted(results.keys()):
        result = results[idx]
        row = rows[idx]
        val = result["is_consistent"]
        reason = str(result["reason"])

        if val is None:
            undecidable += 1
            row["JudgeIsConsistent"] = "N/A"
            row["JudgeReason"] = reason or "cannot parse judge result"
            update_group_stats(
                row=row,
                verdict=None,
                answer_type_stats=answer_type_stats,
                category_stats=category_stats,
                subclass_stats=subclass_stats,
                question_type_stats=question_type_stats,
            )
        elif bool(val):
            correct += 1
            row["JudgeIsConsistent"] = "1"
            row["JudgeReason"] = reason
            update_group_stats(
                row=row,
                verdict=True,
                answer_type_stats=answer_type_stats,
                category_stats=category_stats,
                subclass_stats=subclass_stats,
                question_type_stats=question_type_stats,
            )
        else:
            row["JudgeIsConsistent"] = "0"
            row["JudgeReason"] = reason
            update_group_stats(
                row=row,
                verdict=False,
                answer_type_stats=answer_type_stats,
                category_stats=category_stats,
                subclass_stats=subclass_stats,
                question_type_stats=question_type_stats,
            )

    accuracy = (correct / evaluated) if evaluated > 0 else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = output_dir / f"accuracy_judged_{ts}.csv"
    output_txt = output_dir / f"accuracy_summary_{ts}.txt"

    write_csv_rows(output_csv, headers, rows)

    summary_lines = [
        f"InputCSV: {input_path}",
        f"OutputCSV: {output_csv}",
        f"Model: {args.model}",
        "Temperature: 0",
        f"Workers: {max(1, args.workers)}",
        f"TotalDataRows: {total_rows}",
        f"SelectedExcelRows: {start_idx + 2}-{end_idx + 2}",
        f"EvaluatedRows: {evaluated}",
        f"CorrectRows: {correct}",
        f"UndecidableRows: {undecidable}",
        f"Accuracy: {accuracy:.4f}",
        "",
        "[AnswerType Accuracy]",
        format_metric_line("closed", answer_type_stats.get("closed", init_metric_counter())),
        format_metric_line("open", answer_type_stats.get("open", init_metric_counter())),
        "",
        "[Category Accuracy]",
    ]
    for key in sorted(category_stats.keys()):
        summary_lines.append(format_metric_line(key, category_stats[key]))
    summary_lines.append("")
    summary_lines.append("[Subclass Accuracy]")
    for key in sorted(subclass_stats.keys()):
        summary_lines.append(format_metric_line(key, subclass_stats[key]))
    summary_lines.append("")
    summary_lines.append("[QuestionType Accuracy]")
    summary_lines.append(format_metric_line("TO", question_type_stats.get("TO", init_metric_counter())))
    summary_lines.append(format_metric_line("TC", question_type_stats.get("TC", init_metric_counter())))
    summary_lines.append(format_metric_line("AD", question_type_stats.get("AD", init_metric_counter())))
    summary_lines.append(format_metric_line("TR", question_type_stats.get("TR", init_metric_counter())))
    if "UNKNOWN" in question_type_stats:
        summary_lines.append(format_metric_line("UNKNOWN", question_type_stats["UNKNOWN"]))
    summary_lines.append("")
    summary = "\n".join(summary_lines)
    output_txt.write_text(summary, encoding="utf-8")

    logger.info("Saved judged CSV: %s", output_csv)
    logger.info("Saved summary: %s", output_txt)
    print(summary)


if __name__ == "__main__":
    main()
