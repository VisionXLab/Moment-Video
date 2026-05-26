"""
Moment-Video inference with native video input.

Use this entry point for OpenAI-compatible endpoints that accept a video_url
message item. Raw videos are passed to the endpoint, while decoding and frame
sampling happen inside the provider or model serving stack.

Representative model routing:
- closed-source native-video pipelines: Seed-series and Gemini-series models,
  when accessed through a provider-specific or OpenAI-compatible video endpoint.
- open-source native-video pipelines: Qwen-series and InternVL-series models
  served locally by vLLM.
- sampled-frame / multi-image pipelines: use scripts/inference_sampled_frames.py.

Default sampling policy:
- For locally served vLLM models, pair this script with launch_vllm_server.sh
  using VIDEO_FPS=1 and VIDEO_NUM_FRAMES=64.
- This Python client sends native video_url inputs; vLLM or the API provider
  performs video decoding and frame sampling.

Local vLLM example:
python scripts/inference_native_video.py \
  --input-csv data/annotation_all.csv \
  --video-root data/videos \
  --output-dir result/output \
  --model Qwen3-VL-4B-Instruct \
  --base-url http://127.0.0.1:8085/v1

API endpoint example:
python scripts/inference_native_video.py \
  --input-csv data/annotation_all.csv \
  --video-root data/videos \
  --output-dir result/output_api \
  --model provider/video-native-model \
  --base-url https://api.example.com/v1 \
  --api-key $VIDEO_API_KEY

Pipeline:
1) locate videos as <video-root>/<Category>/<Subclass>/<Index>.mp4;
2) normalize video resolution before sending;
3) encode the normalized MP4 as a video_url data URL;
4) retry with a smaller side length on context or encoder-cache overflow.

Closed multiple-choice rows are evaluated with shuffled option orders. Open
questions write the raw model response to ModelAnswer for later LLM-as-judge
evaluation.
"""

import argparse
import base64
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

csv.field_size_limit(10_000_000)

MAX_SIDE_ELAPSED_SECONDS = 500.0

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model.replace("/", "_")).strip("_")


def setup_logger(output_dir: Path, model: str) -> logging.Logger:
    logger = logging.getLogger("batch_eval_video_native")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = sanitize_model_name(model)
    log_path = output_dir / f"run_video_{model_tag}_{ts}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info("Log file: %s", log_path)
    return logger


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
        headers = [h.strip() if isinstance(h, str) else h for h in list(reader.fieldnames or [])]
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


def ensure_model_answer_header(headers: List[str], rows: List[Dict[str, str]]) -> List[str]:
    out = list(headers)
    if "ModelAnswer" not in out:
        out.append("ModelAnswer")
        for row in rows:
            row["ModelAnswer"] = row.get("ModelAnswer", "")
    return out


def ensure_closed_eval_headers(headers: List[str], rows: List[Dict[str, str]]) -> List[str]:
    out = list(headers)
    for col in ["ClosedEvalPass", "ClosedEvalReason", "ClosedEvalMeta"]:
        if col not in out:
            out.append(col)
            for row in rows:
                row[col] = row.get(col, "")
    return out


def parse_row_range(total_data_rows: int, start_row: int, end_row: Optional[int]) -> Tuple[int, int]:
    if total_data_rows <= 0:
        raise ValueError("CSV has no data rows.")
    data_start_excel = max(2, start_row)
    data_end_excel = (total_data_rows + 1) if end_row is None else min(end_row, total_data_rows + 1)
    if data_start_excel > data_end_excel:
        raise ValueError(f"Invalid row range: start={data_start_excel}, end={data_end_excel}")
    return data_start_excel - 2, data_end_excel - 2


def is_closed_question_row(row: Dict[str, str]) -> bool:
    question_stage = (row.get("QuestionStage") or "").strip().lower()
    answer_type = (row.get("AnswerType") or "").strip().lower()
    return question_stage == "closed" or answer_type == "closed"


def parse_mcq_question(question: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    matches = list(re.finditer(r"\(([a-zA-Z])\)\s*", question))
    if len(matches) < 2:
        return None
    stem = question[: matches[0].start()].strip()
    options: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        label = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        text = question[start:end].strip()
        if not text:
            return None
        options.append((label, text))
    if len({k for k, _ in options}) != len(options):
        return None
    return stem, options


def parse_gold_option_label(answer: str) -> Optional[str]:
    text = (answer or "").strip().lower()
    if not text:
        return None
    m = re.search(r"\(([a-z])\)", text)
    if m:
        return m.group(1)
    cleaned = re.sub(r"[^a-z]", "", text)
    if len(cleaned) == 1:
        return cleaned
    return None


def clean_model_answer(answer: str) -> str:
    text = (answer or "").strip()
    boxed_match = re.fullmatch(r"<\|begin_of_box\|>\s*(.*?)\s*<\|end_of_box\|>", text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    return text


def extract_option_label(model_answer: str, valid_labels: List[str]) -> Optional[str]:
    text = (model_answer or "").strip().lower()
    if not text:
        return None
    patterns = [
        r"\(\s*([a-z])\s*\)",
        r"<\|begin_of_box\|>\s*\(?\s*([a-z])\s*\)?\s*<\|end_of_box\|>",
        r"^\s*\(?\s*([a-z])\s*\)?\s*\.?\s*$",
        r"\b(?:option|answer)\s*(?:is)?\s*[:：]?\s*\(?\s*([a-z])\s*\)?",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        label = m.group(1).lower()
        if label in valid_labels:
            return label
    return None


def build_mcq_permutations(question: str, gold_label: str) -> Optional[Tuple[List[dict], List[Tuple[str, str]]]]:
    parsed = parse_mcq_question(question)
    if not parsed:
        return None
    stem, options = parsed
    old_labels = [k for k, _ in options]
    if gold_label not in old_labels:
        return None
    option_count = len(options)
    if option_count < 2 or option_count > 26:
        return None
    new_labels = [chr(ord("a") + i) for i in range(option_count)]
    gold_idx = old_labels.index(gold_label)
    variants: List[dict] = []
    for target_idx in range(option_count):
        perm = list(range(option_count))
        perm[target_idx], perm[gold_idx] = perm[gold_idx], perm[target_idx]
        option_parts = []
        for new_idx, old_idx in enumerate(perm):
            option_parts.append(f"({new_labels[new_idx]}) {options[old_idx][1]}")
        question_text = (stem + " " if stem else "") + "  ".join(option_parts)
        variants.append({"question": question_text, "new_labels": list(new_labels), "perm": perm})
    return variants, options


def transcode_video_for_vl(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    max_video_side: int,
    crf: int,
) -> None:
    vf = (
        "scale="
        f"if(gt(iw\\,ih)\\,min(iw\\,{max_video_side})\\,-2):"
        f"if(gt(iw\\,ih)\\,-2\\,min(ih\\,{max_video_side})),"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {(proc.stderr or '').strip()}")


def make_video_data_url(
    video_path: Path,
    max_video_side: int,
) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg first.")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if max_video_side <= 0:
        raise ValueError(f"max_video_side must be > 0, got {max_video_side}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out = tmp / "norm.mp4"
        transcode_video_for_vl(ffmpeg, video_path, out, max_video_side, crf=23)
        b64 = base64.standard_b64encode(out.read_bytes()).decode("ascii")
        return f"data:video/mp4;base64,{b64}"


def ask_with_video(
    client: OpenAI,
    model: str,
    question: str,
    video_data_url: str,
    timeout: int,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video_url", "video_url": {"url": video_data_url}},
                ],
            }
        ],
        timeout=timeout,
    )
    if not completion or not completion.choices:
        return ""
    return (completion.choices[0].message.content or "").strip()


def is_context_limit_error(error_text: str) -> bool:
    t = (error_text or "").lower()
    return (
        ("input length" in t and "maximum context length" in t and "exceeds" in t)
        or (
            "decoder prompt" in t
            and "maximum model length" in t
            and ("longer than" in t or "no smaller than" in t)
        )
    )


def is_encoder_cache_limit_error(error_text: str) -> bool:
    t = (error_text or "").lower()
    return (
        "video item with length" in t
        and "encoder cache size" in t
        and "limit-mm-per-prompt" in t
    )


def process_row_job(
    idx: int,
    row: Dict[str, str],
    question_field: str,
    video_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    max_video_side: int,
    fallback_video_side: int,
    timeout: int,
    max_retries: int,
    retry_base_seconds: float,
) -> Dict[str, str]:
    display_row = idx + 2
    category = (row.get("Category") or "").strip()
    subclass = (row.get("Subclass") or "").strip()
    index = (row.get("Index") or "").strip()
    question = (row.get(question_field) or "").strip()
    video_path = video_root / category / subclass / f"{index}.mp4"
    client = OpenAI(base_url=base_url, api_key=api_key)

    current_side = max_video_side
    video_data_url = make_video_data_url(
        video_path=video_path,
        max_video_side=current_side,
    )
    input_mode = "video_native"

    def single_run_answer(prompt: str) -> Tuple[bool, str, int, float]:
        nonlocal current_side, video_data_url, input_mode
        last_error = ""
        total_elapsed = 0.0
        side_elapsed = 0.0
        side_attempt = 0
        total_attempts = 0

        def fallback_to_smaller_side(reason: str) -> None:
            nonlocal current_side, video_data_url, input_mode, side_elapsed, side_attempt
            current_side = fallback_video_side
            video_data_url = make_video_data_url(
                video_path=video_path,
                max_video_side=current_side,
            )
            input_mode = f"video_native|side_fallback:{reason}"
            side_elapsed = 0.0
            side_attempt = 0

        while True:
            if side_attempt >= max_retries:
                return False, last_error, total_attempts, total_elapsed

            side_attempt += 1
            total_attempts += 1
            t0 = time.time()
            try:
                answer = ask_with_video(
                    client=client,
                    model=model,
                    question=prompt,
                    video_data_url=video_data_url,
                    timeout=timeout,
                )
                elapsed = time.time() - t0
                total_elapsed += elapsed
                if not answer or not answer.strip():
                    raise RuntimeError("empty model response")
                return True, answer, total_attempts, total_elapsed
            except Exception as exc:
                elapsed = time.time() - t0
                total_elapsed += elapsed
                side_elapsed += elapsed
                last_error = str(exc)
                if (
                    (is_context_limit_error(last_error) or is_encoder_cache_limit_error(last_error))
                    and current_side > fallback_video_side
                    and fallback_video_side > 0
                ):
                    reason = "encoder_cache_limit" if is_encoder_cache_limit_error(last_error) else "context_limit"
                    fallback_to_smaller_side(reason)
                    continue

                if side_elapsed > MAX_SIDE_ELAPSED_SECONDS:
                    if current_side > fallback_video_side and fallback_video_side > 0:
                        fallback_to_smaller_side("elapsed_limit")
                        continue
                    return (
                        False,
                        (
                            f"side={current_side} exceeded {MAX_SIDE_ELAPSED_SECONDS:.0f}s "
                            f"after {side_attempt} attempts: {last_error}"
                        ),
                        total_attempts,
                        total_elapsed,
                    )

                if side_attempt < max_retries:
                    time.sleep(retry_base_seconds * (2 ** (side_attempt - 1)))

        return False, last_error, total_attempts, total_elapsed

    if is_closed_question_row(row):
        gold_label = parse_gold_option_label(row.get("Answer", ""))
        permutation_result = build_mcq_permutations(question, gold_label or "")
        if not gold_label or not permutation_result:
            ok, payload, attempts, elapsed = single_run_answer(question)
            meta = {
                "input_mode": input_mode,
                "max_video_side": current_side,
            }
            if ok:
                return {
                    "idx": str(idx),
                    "ModelAnswer": payload,
                    "ClosedEvalPass": "N/A",
                    "ClosedEvalReason": "closed parse failed, fallback single-run",
                    "ClosedEvalMeta": json.dumps(meta, ensure_ascii=False),
                    "log_level": "info",
                    "log_message": f"row={display_row} success attempt={attempts} elapsed={elapsed:.2f}s",
                }
            return {
                "idx": str(idx),
                "ModelAnswer": f"ERROR: {payload}",
                "ClosedEvalPass": "N/A",
                "ClosedEvalReason": "closed parse failed, fallback single-run",
                "ClosedEvalMeta": json.dumps(meta, ensure_ascii=False),
                "log_level": "warning",
                "log_message": f"row={display_row} failed attempts={attempts} elapsed={elapsed:.2f}s error={payload}",
            }

        variants, original_options = permutation_result
        trials = []
        for trial_idx, variant in enumerate(variants, start=1):
            trial_prompt = (
                f"{variant['question']}\n\n"
                "Reply with only one option label in parentheses, such as (a). "
                "Do not output any explanation."
            )
            ok, payload, _, _ = single_run_answer(trial_prompt)
            if not ok:
                reason = f"trial {trial_idx} failed: {payload}"
                return {
                    "idx": str(idx),
                    "ModelAnswer": f"ERROR: {reason}",
                    "ClosedEvalPass": "N/A",
                    "ClosedEvalReason": reason,
                    "ClosedEvalMeta": json.dumps(
                        {
                            "input_mode": input_mode,
                            "max_video_side": current_side,
                            "trials": trials,
                        },
                        ensure_ascii=False,
                    ),
                    "log_level": "warning",
                    "log_message": f"row={display_row} closed-eval failed: {reason}",
                }
            pred_new_label = extract_option_label(payload, variant["new_labels"])
            if pred_new_label is None:
                reason = f"trial {trial_idx} cannot parse option label: {payload}"
                return {
                    "idx": str(idx),
                    "ModelAnswer": f"ERROR: {reason}",
                    "ClosedEvalPass": "N/A",
                    "ClosedEvalReason": reason,
                    "ClosedEvalMeta": json.dumps(
                        {
                            "input_mode": input_mode,
                            "max_video_side": current_side,
                            "trials": trials,
                        },
                        ensure_ascii=False,
                    ),
                    "log_level": "warning",
                    "log_message": f"row={display_row} closed-eval failed: {reason}",
                }
            perm = variant["perm"]
            pred_new_idx = variant["new_labels"].index(pred_new_label)
            pred_old_idx = perm[pred_new_idx]
            pred_old_label = original_options[pred_old_idx][0]
            trials.append(
                {
                    "trial": trial_idx,
                    "pred_new_label": pred_new_label,
                    "pred_old_label": pred_old_label,
                    "is_correct": pred_old_label == gold_label,
                    "raw_answer": payload,
                }
            )

        correct_cnt = sum(1 for t in trials if t["is_correct"])
        total_cnt = len(trials)
        pass_all = total_cnt > 0 and correct_cnt == total_cnt
        pred_old_labels = [t["pred_old_label"] for t in trials]
        canonical_label = max(set(pred_old_labels), key=pred_old_labels.count) if pred_old_labels else ""
        return {
            "idx": str(idx),
            "ModelAnswer": f"({canonical_label})" if canonical_label else "ERROR: no closed-eval trials",
            "ClosedEvalPass": "1" if pass_all else "0",
            "ClosedEvalReason": f"closed permutation eval: {correct_cnt}/{total_cnt} correct",
            "ClosedEvalMeta": json.dumps(
                {
                    "input_mode": input_mode,
                    "max_video_side": current_side,
                    "trials": trials,
                },
                ensure_ascii=False,
            ),
            "log_level": "info",
            "log_message": (
                f"row={display_row} closed-eval pass={'1' if pass_all else '0'} "
                f"correct={correct_cnt}/{total_cnt}"
            ),
        }

    ok, payload, attempts, elapsed = single_run_answer(question)
    if ok:
        return {
            "idx": str(idx),
            "ModelAnswer": clean_model_answer(payload),
            "ClosedEvalPass": "N/A",
            "ClosedEvalReason": "not a closed question",
            "ClosedEvalMeta": json.dumps(
                {
                    "input_mode": input_mode,
                    "max_video_side": current_side,
                },
                ensure_ascii=False,
            ),
            "log_level": "info",
            "log_message": f"row={display_row} success attempt={attempts} elapsed={elapsed:.2f}s",
        }
    return {
        "idx": str(idx),
        "ModelAnswer": f"ERROR: {payload}",
        "ClosedEvalPass": "N/A",
        "ClosedEvalReason": "not a closed question",
        "ClosedEvalMeta": json.dumps(
            {
                "input_mode": input_mode,
                "max_video_side": current_side,
            },
            ensure_ascii=False,
        ),
        "log_level": "warning",
        "log_message": f"row={display_row} failed attempts={attempts} elapsed={elapsed:.2f}s error={payload}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate Moment-Video via native video_url input.")
    parser.add_argument("--input-csv", required=True, help="Path to the input annotation CSV.")
    parser.add_argument("--video-root", required=True, help="Root directory containing input videos.")
    parser.add_argument("--output-dir", required=True, help="Directory to save result CSV and logs.")
    parser.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8085/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY") or "EMPTY")
    parser.add_argument("--max-video-side", "--max-frame-side", dest="max_video_side", type=int, default=1024)
    parser.add_argument(
        "--fallback-video-side",
        "--fallback-max-frame-side",
        dest="fallback_video_side",
        type=int,
        default=512,
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--start-row", type=int, default=2)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("--api-key must be non-empty.")
    if not args.base_url:
        raise ValueError("--base-url must be non-empty.")
    if args.max_video_side <= 0:
        raise ValueError("--max-video-side must be > 0.")
    if args.fallback_video_side <= 0:
        raise ValueError("--fallback-video-side must be > 0.")

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, args.model)

    headers, rows = read_csv_rows(input_csv)
    headers = ensure_model_answer_header(headers, rows)
    headers = ensure_closed_eval_headers(headers, rows)

    question_field = "Question" if "Question" in headers else headers[-2] if len(headers) >= 2 else headers[-1]
    start_idx, end_idx = parse_row_range(len(rows), args.start_row, args.end_row)
    video_root = Path(args.video_root)

    total = end_idx - start_idx + 1
    workers = max(1, args.workers)
    logger.info("Start processing row-index %d-%d, total=%d", start_idx, end_idx, total)
    logger.info("Model: %s | Workers: %d", args.model, workers)
    logger.info("Endpoint: %s", args.base_url)
    logger.info(
        "Video-native config: max_video_side=%d, fallback_side=%d",
        args.max_video_side,
        args.fallback_video_side,
    )

    jobs: List[Tuple[int, Dict[str, str]]] = []
    for idx in range(start_idx, end_idx + 1):
        row = rows[idx]
        display_row = idx + 2
        category = (row.get("Category") or "").strip()
        subclass = (row.get("Subclass") or "").strip()
        index = (row.get("Index") or "").strip()
        question = (row.get(question_field) or "").strip()
        if not category or not subclass or not index or not question:
            row["ModelAnswer"] = "ERROR: missing required fields in CSV row"
            row["ClosedEvalPass"] = "N/A"
            row["ClosedEvalReason"] = "missing required fields"
            row["ClosedEvalMeta"] = ""
            logger.warning("row=%d missing required fields", display_row)
            continue
        video_path = video_root / category / subclass / f"{index}.mp4"
        if not video_path.exists():
            row["ModelAnswer"] = f"ERROR: video not found: {video_path}"
            row["ClosedEvalPass"] = "N/A"
            row["ClosedEvalReason"] = "video not found"
            row["ClosedEvalMeta"] = ""
            logger.warning("row=%d video missing: %s", display_row, video_path)
            continue
        jobs.append((idx, row))

    logger.info("Prepared video-native jobs: %d", len(jobs))

    progress = tqdm(total=len(jobs), desc="Evaluating-Video", unit="row") if tqdm is not None and jobs else None
    done_jobs = 0
    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    process_row_job,
                    idx,
                    row,
                    question_field,
                    video_root,
                    args.base_url,
                    args.api_key,
                    args.model,
                    args.max_video_side,
                    args.fallback_video_side,
                    args.timeout,
                    args.max_retries,
                    args.retry_base_seconds,
                ): idx
                for idx, row in jobs
            }
            total_jobs = len(future_map)
            pending = set(future_map.keys())
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                if not done:
                    logger.info("Heartbeat: completed=%d/%d pending=%d", done_jobs, total_jobs, len(pending))
                    continue
                for future in done:
                    idx = future_map[future]
                    row = rows[idx]
                    try:
                        result = future.result()
                        row["ModelAnswer"] = result["ModelAnswer"]
                        row["ClosedEvalPass"] = result["ClosedEvalPass"]
                        row["ClosedEvalReason"] = result["ClosedEvalReason"]
                        row["ClosedEvalMeta"] = result["ClosedEvalMeta"]
                        if result.get("log_message"):
                            if result.get("log_level") == "warning":
                                logger.warning(result["log_message"])
                            else:
                                logger.info(result["log_message"])
                    except Exception as exc:
                        err = str(exc)
                        row["ModelAnswer"] = f"ERROR: {err}"
                        row["ClosedEvalPass"] = "N/A"
                        row["ClosedEvalReason"] = "worker crashed"
                        row["ClosedEvalMeta"] = ""
                        logger.exception("row=%d worker exception: %s", idx + 2, err)
                    done_jobs += 1
                    if progress is not None:
                        progress.update(1)
                    elif done_jobs % 10 == 0 or done_jobs == total_jobs:
                        logger.info("Progress: %d/%d", done_jobs, total_jobs)

    if progress is not None:
        progress.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_tag = sanitize_model_name(args.model)
    final_path = output_dir / f"result_video_{model_tag}_{timestamp}.csv"
    write_csv_rows(final_path, headers, rows)
    logger.info("Saved final result: %s", final_path)


if __name__ == "__main__":
    main()
