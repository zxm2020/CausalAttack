#!/usr/bin/env bash
# Upload the zip to a CUDA server, put the personal SDK token in key.txt, then bash run_server.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -z "${MODELSCOPE_SDK_TOKEN:-}" ]]; then
    if [[ ! -s key.txt ]]; then
        printf '%s\n' 'No ModelScope key: write your personal SDK token into ./key.txt (one line), chmod 600 key.txt.' >&2
        exit 2
    fi
    if [[ "$(stat -c '%a' key.txt)" != "600" ]]; then
        printf '%s\n' 'For security: chmod 600 key.txt' >&2
        exit 2
    fi
    IFS= read -r MODELSCOPE_SDK_TOKEN < key.txt || [[ -n "$MODELSCOPE_SDK_TOKEN" ]]
    MODELSCOPE_SDK_TOKEN="${MODELSCOPE_SDK_TOKEN%$'\r'}"
    export MODELSCOPE_SDK_TOKEN
fi
if [[ -z "$MODELSCOPE_SDK_TOKEN" ]]; then
    echo 'The API token is empty.' >&2; exit 2
fi
if ! command -v python >/dev/null 2>&1; then
    echo 'Activate your Python 3.10/3.11 + CUDA PyTorch environment first.' >&2; exit 2
fi
python -m pip install -r requirements.txt
python - <<'PY'
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU unavailable. Activate a CUDA-enabled PyTorch environment before running. No API calls have been made.')
print('GPU:', torch.cuda.get_device_name(0))
PY
python run_all.py "$@"
