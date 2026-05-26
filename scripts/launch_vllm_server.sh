#!/bin/bash
set -euo pipefail

# Launch an OpenAI-compatible vLLM server for video-native Moment-Video inference.
#
# Use this for locally deployed open-source models whose serving stack accepts raw
# videos through video_url and performs server-side decoding / frame sampling.
# Representative models include Qwen-series and InternVL-series models. After the
# server is ready, run scripts/inference_native_video.py.
#
# Usage example:
# MODEL_DIR=models/Qwen3.5-27B ALLOWED_LOCAL_MEDIA_PATH=data/videos bash scripts/launch_vllm_server.sh

: "${MODEL_DIR:?Please set MODEL_DIR, for example: MODEL_DIR=models/Qwen3.5-27B}"
: "${ALLOWED_LOCAL_MEDIA_PATH:?Please set ALLOWED_LOCAL_MEDIA_PATH, for example: ALLOWED_LOCAL_MEDIA_PATH=data/videos}"

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.5-27B}

LOG_DIR=${LOG_DIR:-./logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/vllm_${SERVED_MODEL_NAME}_fps8_len32768.log}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8085}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
VIDEO_FPS=${VIDEO_FPS:-1}
VIDEO_NUM_FRAMES=${VIDEO_NUM_FRAMES:-64}

# Frame budget notes:
# - Default / main setting: keep VIDEO_FPS=1 and VIDEO_NUM_FRAMES=64.
#   This is the setting expected by scripts/inference_native_video.py examples.
# - 8 FPS ablation for Qwen-series models: set VIDEO_FPS=8 and
#   VIDEO_NUM_FRAMES=-1 so vLLM samples according to FPS.
# - 8 FPS ablation for non-Qwen models: set VIDEO_FPS=8 and
#   VIDEO_NUM_FRAMES=110, or another explicit frame cap required by the model.

mkdir -p "$LOG_DIR"

echo "Using MODEL_DIR=$MODEL_DIR"
echo "Using SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "Using HOST=$HOST"
echo "Using PORT=$PORT"
echo "Using MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "Using VIDEO_FPS=$VIDEO_FPS"
echo "Using VIDEO_NUM_FRAMES=$VIDEO_NUM_FRAMES"
echo "Using ALLOWED_LOCAL_MEDIA_PATH=$ALLOWED_LOCAL_MEDIA_PATH"
echo "Using vLLM: $(which vllm)"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES

vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --port "$PORT" \
  --host "$HOST" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --limit-mm-per-prompt '{"video":1}' \
  --media-io-kwargs "{\"video\":{\"fps\":$VIDEO_FPS,\"num_frames\":$VIDEO_NUM_FRAMES}}" \
  2>&1 | tee -a "$LOG_FILE"

# After the vLLM server is ready, run the experiment in another terminal:
# python scripts/inference_native_video.py \
#   --input-csv data/annotation_all.csv \
#   --video-root data/videos \
#   --output-dir result/output \
#   --model "$SERVED_MODEL_NAME" \
#   --base-url "http://$HOST:$PORT/v1"
