"""
Moment-Video inference with sampled-frame image input.

Use this entry point when Moment-Video is evaluated by first converting each
video into sampled frames and then sending those frames as multiple image_url
items. It is useful for multi-image API adapters and for controlled frame-rate
ablations where the benchmark protocol fixes FPS and the maximum number of
frames.

Representative model routing:
- closed-source sampled-frame / multi-image pipeline: GPT-series models through
  an API adapter that accepts sampled frames.
- open-source sampled-frame / multi-image pipeline: Kimi-series models through
  an OpenRouter-style image input path.
- native-video pipelines: use scripts/inference_native_video.py.

Default sampling policy:
- This script samples at 1 FPS and caps each video at 64 frames by default.
- For local vLLM or API adapters, make sure the model endpoint can fetch the
  image_url values served by this client. A 127.0.0.1 media server is suitable
  for local endpoints; cloud APIs need a reachable media host or adapter.

Local vLLM / OpenAI-compatible example:
python scripts/inference_sampled_frames.py \
  --input-csv data/annotation_all.csv \
  --video-root data/videos \
  --output-dir result/output_frames \
  --model Kimi-2.6 \
  --base-url http://127.0.0.1:8085/v1

API endpoint example:
python scripts/inference_sampled_frames.py \
  --input-csv data/annotation_all.csv \
  --video-root data/videos \
  --output-dir result/output_api_frames \
  --model provider/multi-image-model \
  --base-url https://api.example.com/v1 \
  --api-key $SAMPLED_FRAMES_API_KEY \
  --media-host <public-or-adapter-reachable-host>

Pipeline:
1) sample each source video at --sample-fps;
2) uniformly reduce to --max-sampled-frames if needed;
3) resize sampled frames while preserving aspect ratio;
4) serve JPEG frames from a temporary local HTTP server;
5) send the ordered frames as image_url items to the model.

Closed multiple-choice rows are evaluated with shuffled option orders. Open
questions write the raw model response to ModelAnswer for later LLM-as-judge
evaluation.
"""

import argparse
import csv
import functools
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from decord import VideoReader, cpu
from PIL import Image

from inference_native_video import (
    MAX_SIDE_ELAPSED_SECONDS,
    build_mcq_permutations,
    clean_model_answer,
    ensure_closed_eval_headers,
    ensure_model_answer_header,
    extract_option_label,
    is_closed_question_row,
    is_context_limit_error,
    is_encoder_cache_limit_error,
    parse_gold_option_label,
    parse_row_range,
    read_csv_rows,
    sanitize_model_name,
    setup_logger,
    write_csv_rows,
)

csv.field_size_limit(10_000_000)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


def start_media_server(media_dir: Path, host: str, port: int) -> Tuple[ThreadingHTTPServer, str]:
    handler = functools.partial(QuietHTTPRequestHandler, directory=str(media_dir))
    server = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = server.server_address
    public_host = "127.0.0.1" if actual_host in {"", "0.0.0.0"} else actual_host
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://{public_host}:{actual_port}"


def resize_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.BICUBIC)


def choose_frame_indices(
    total_frames: int,
    avg_fps: float,
    sample_fps: float,
    max_sampled_frames: int,
) -> List[int]:
    if total_frames <= 0:
        return []
    if avg_fps > 0:
        step = max(1, round(avg_fps / sample_fps))
        frame_indices = list(range(0, total_frames, step))
    else:
        frame_indices = list(range(total_frames))
    if len(frame_indices) > max_sampled_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_sampled_frames, dtype=int).tolist()
    return frame_indices


def make_sampled_frame_urls(
    video_path: Path,
    media_dir: Path,
    media_base_url: str,
    max_video_side: int,
    sample_fps: float,
    max_sampled_frames: int,
) -> Tuple[List[str], Dict[str, object]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if max_video_side <= 0:
        raise ValueError(f"max_video_side must be > 0, got {max_video_side}")
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps}")
    if max_sampled_frames <= 0:
        raise ValueError(f"max_sampled_frames must be > 0, got {max_sampled_frames}")

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"No frames found in video: {video_path}")

    avg_fps = float(vr.get_avg_fps())
    frame_indices = choose_frame_indices(total_frames, avg_fps, sample_fps, max_sampled_frames)
    if not frame_indices:
        raise ValueError(f"No frame indices sampled from video: {video_path}")

    frame_dir_name = uuid.uuid4().hex
    frame_dir = media_dir / frame_dir_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames = vr.get_batch(frame_indices).asnumpy()
    frame_urls = []
    total_bytes = 0
    for order, frame in enumerate(frames):
        image = Image.fromarray(frame).convert("RGB")
        image = resize_keep_aspect(image, max_video_side)
        frame_name = f"frame_{order:04d}.jpg"
        frame_path = frame_dir / frame_name
        image.save(frame_path, format="JPEG", quality=90, optimize=True)
        total_bytes += frame_path.stat().st_size
        frame_urls.append(f"{media_base_url}/{frame_dir_name}/{frame_name}")

    video_time = total_frames / avg_fps if avg_fps > 0 else 0.0
    frame_times = [idx / avg_fps if avg_fps > 0 else 0.0 for idx in frame_indices]
    meta = {
        "source_fps": avg_fps,
        "source_total_frames": total_frames,
        "source_duration": video_time,
        "requested_sample_fps": sample_fps,
        "max_sampled_frames": max_sampled_frames,
        "sampled_frames": len(frame_indices),
        "sampled_indices": frame_indices,
        "sampled_times": [round(t, 3) for t in frame_times],
        "max_video_side": max_video_side,
        "served_frame_bytes": total_bytes,
    }
    return frame_urls, meta


def ask_with_frame_urls(
    client: OpenAI,
    model: str,
    question: str,
    frame_urls: List[str],
    timeout: int,
    max_tokens: int,
) -> str:
    framed_question = (
        f"The following {len(frame_urls)} images are frames sampled from a video "
        "in chronological order. Answer the question based on these frames.\n\n"
        f"{question}"
    )
    content = [{"type": "text", "text": framed_question}]
    content.extend(
        {"type": "image_url", "image_url": {"url": frame_url}}
        for frame_url in frame_urls
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        timeout=timeout,
    )
    if not completion or not completion.choices:
        return ""
    return (completion.choices[0].message.content or "").strip()


def process_row_job(
    idx: int,
    row: Dict[str, str],
    question_field: str,
    video_root: Path,
    media_dir: Path,
    media_base_url: str,
    base_url: str,
    api_key: str,
    model: str,
    max_video_side: int,
    fallback_video_side: int,
    sample_fps: float,
    max_sampled_frames: int,
    timeout: int,
    max_tokens: int,
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
    frame_urls, frame_meta = make_sampled_frame_urls(
        video_path=video_path,
        media_dir=media_dir,
        media_base_url=media_base_url,
        max_video_side=current_side,
        sample_fps=sample_fps,
        max_sampled_frames=max_sampled_frames,
    )
    input_mode = "http_image_urls|client_sampled_frames"

    def single_run_answer(prompt: str) -> Tuple[bool, str, int, float]:
        nonlocal current_side, frame_urls, frame_meta, input_mode
        last_error = ""
        total_elapsed = 0.0
        side_elapsed = 0.0
        side_attempt = 0
        total_attempts = 0

        def fallback_to_smaller_side(reason: str) -> None:
            nonlocal current_side, frame_urls, frame_meta, input_mode, side_elapsed, side_attempt
            current_side = fallback_video_side
            frame_urls, frame_meta = make_sampled_frame_urls(
                video_path=video_path,
                media_dir=media_dir,
                media_base_url=media_base_url,
                max_video_side=current_side,
                sample_fps=sample_fps,
                max_sampled_frames=max_sampled_frames,
            )
            input_mode = f"http_image_urls|client_sampled_frames|side_fallback:{reason}"
            side_elapsed = 0.0
            side_attempt = 0

        while True:
            if side_attempt >= max_retries:
                return False, last_error, total_attempts, total_elapsed

            side_attempt += 1
            total_attempts += 1
            t0 = time.time()
            try:
                answer = ask_with_frame_urls(
                    client=client,
                    model=model,
                    question=prompt,
                    frame_urls=frame_urls,
                    timeout=timeout,
                    max_tokens=max_tokens,
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

    def base_meta() -> Dict[str, object]:
        return {
            "input_mode": input_mode,
            "frame_urls": frame_urls,
            "max_video_side": current_side,
            "sampling": frame_meta,
        }

    if is_closed_question_row(row):
        gold_label = parse_gold_option_label(row.get("Answer", ""))
        permutation_result = build_mcq_permutations(question, gold_label or "")
        if not gold_label or not permutation_result:
            ok, payload, attempts, elapsed = single_run_answer(question)
            if ok:
                return {
                    "idx": str(idx),
                    "ModelAnswer": payload,
                    "ClosedEvalPass": "N/A",
                    "ClosedEvalReason": "closed parse failed, fallback single-run",
                    "ClosedEvalMeta": json.dumps(base_meta(), ensure_ascii=False),
                    "log_level": "info",
                    "log_message": f"row={display_row} success attempt={attempts} elapsed={elapsed:.2f}s",
                }
            return {
                "idx": str(idx),
                "ModelAnswer": f"ERROR: {payload}",
                "ClosedEvalPass": "N/A",
                "ClosedEvalReason": "closed parse failed, fallback single-run",
                "ClosedEvalMeta": json.dumps(base_meta(), ensure_ascii=False),
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
                meta = base_meta()
                meta["trials"] = trials
                return {
                    "idx": str(idx),
                    "ModelAnswer": f"ERROR: {reason}",
                    "ClosedEvalPass": "N/A",
                    "ClosedEvalReason": reason,
                    "ClosedEvalMeta": json.dumps(meta, ensure_ascii=False),
                    "log_level": "warning",
                    "log_message": f"row={display_row} closed-eval failed: {reason}",
                }
            pred_new_label = extract_option_label(payload, variant["new_labels"])
            if pred_new_label is None:
                reason = f"trial {trial_idx} cannot parse option label: {payload}"
                meta = base_meta()
                meta["trials"] = trials
                return {
                    "idx": str(idx),
                    "ModelAnswer": f"ERROR: {reason}",
                    "ClosedEvalPass": "N/A",
                    "ClosedEvalReason": reason,
                    "ClosedEvalMeta": json.dumps(meta, ensure_ascii=False),
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
        meta = base_meta()
        meta["trials"] = trials
        return {
            "idx": str(idx),
            "ModelAnswer": f"({canonical_label})" if canonical_label else "ERROR: no closed-eval trials",
            "ClosedEvalPass": "1" if pass_all else "0",
            "ClosedEvalReason": f"closed permutation eval: {correct_cnt}/{total_cnt} correct",
            "ClosedEvalMeta": json.dumps(meta, ensure_ascii=False),
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
            "ClosedEvalMeta": json.dumps(base_meta(), ensure_ascii=False),
            "log_level": "info",
            "log_message": f"row={display_row} success attempt={attempts} elapsed={elapsed:.2f}s",
        }
    return {
        "idx": str(idx),
        "ModelAnswer": f"ERROR: {payload}",
        "ClosedEvalPass": "N/A",
        "ClosedEvalReason": "not a closed question",
        "ClosedEvalMeta": json.dumps(base_meta(), ensure_ascii=False),
        "log_level": "warning",
        "log_message": f"row={display_row} failed attempts={attempts} elapsed={elapsed:.2f}s error={payload}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluate Moment-Video with sampled frames sent as image_url inputs."
    )
    parser.add_argument("--input-csv", required=True, help="Path to the input annotation CSV.")
    parser.add_argument("--video-root", required=True, help="Root directory containing input videos.")
    parser.add_argument("--output-dir", required=True, help="Directory to save result CSV and logs.")
    parser.add_argument("--model", default="Kimi-2.6")
    parser.add_argument("--base-url", default="http://127.0.0.1:8085/v1")
    parser.add_argument("--api-key", default=os.getenv("SAMPLED_FRAMES_API_KEY") or "EMPTY")
    parser.add_argument("--max-video-side", "--max-frame-side", dest="max_video_side", type=int, default=1024)
    parser.add_argument(
        "--fallback-video-side",
        "--fallback-max-frame-side",
        dest="fallback_video_side",
        type=int,
        default=512,
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-sampled-frames", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--start-row", type=int, default=2)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--media-host", default="127.0.0.1")
    parser.add_argument("--media-port", type=int, default=0)
    parser.add_argument("--keep-media-cache", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("--api-key must be non-empty.")
    if not args.base_url:
        raise ValueError("--base-url must be non-empty.")
    if args.max_video_side <= 0:
        raise ValueError("--max-video-side must be > 0.")
    if args.fallback_video_side <= 0:
        raise ValueError("--fallback-video-side must be > 0.")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be > 0.")
    if args.max_sampled_frames <= 0:
        raise ValueError("--max-sampled-frames must be > 0.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0.")

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

    media_tmp = tempfile.TemporaryDirectory(prefix="sampled_frame_media_")
    media_dir = Path(media_tmp.name)
    media_server, media_base_url = start_media_server(media_dir, args.media_host, args.media_port)

    try:
        total = end_idx - start_idx + 1
        workers = max(1, args.workers)
        logger.info("Start processing row-index %d-%d, total=%d", start_idx, end_idx, total)
        logger.info("Model: %s | Workers: %d", args.model, workers)
        logger.info("Endpoint: %s", args.base_url)
        logger.info("Media server: %s -> %s", media_base_url, media_dir)
        logger.info(
            "Sampled-frame config: sample_fps=%.3f, max_sampled_frames=%d, max_frame_side=%d, fallback_side=%d",
            args.sample_fps,
            args.max_sampled_frames,
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

        logger.info("Prepared sampled-frame jobs: %d", len(jobs))

        progress = tqdm(total=len(jobs), desc="Evaluating-Sampled-Frames", unit="row") if tqdm is not None and jobs else None
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
                        media_dir,
                        media_base_url,
                        args.base_url,
                        args.api_key,
                        args.model,
                        args.max_video_side,
                        args.fallback_video_side,
                        args.sample_fps,
                        args.max_sampled_frames,
                        args.timeout,
                        args.max_tokens,
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
        final_path = output_dir / f"result_sampled_frames_{model_tag}_{timestamp}.csv"
        write_csv_rows(final_path, headers, rows)
        logger.info("Saved final result: %s", final_path)

        if args.keep_media_cache:
            keep_dir = output_dir / f"sampled_frame_media_{timestamp}"
            shutil.copytree(media_dir, keep_dir, dirs_exist_ok=True)
            logger.info("Saved media cache: %s", keep_dir)

    finally:
        media_server.shutdown()
        media_server.server_close()
        media_tmp.cleanup()


if __name__ == "__main__":
    main()
