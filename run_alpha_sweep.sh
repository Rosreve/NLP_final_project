#!/usr/bin/env bash
# Grid search CombinedRetreiver --alpha and --beta: run pipeline, then eval with a distinct metrics filename.
# Gamma is fixed as 1 - alpha - beta in run.py, so skip pairs with alpha + beta > 1.
# run.py always writes output/predictions.json, so eval must run immediately after each run.py.

set -euo pipefail # exit on error

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # get the root directory of the project
cd "$ROOT" # change to the root directory

# activate the virtual environment
VENV_ACTIVATE="${ROOT}/.venv/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  printf 'Missing venv activate script: %s\n' "${VENV_ACTIVATE}" >&2
  exit 1
fi
# shellcheck source=/dev/null
. "${VENV_ACTIVATE}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" # set the python path

PRED="${ROOT}/output/predictions.json" # set the predictions file (output of run.py)
GT="${ROOT}/output/dev-groundtruth.json" # set the ground truth file (ground truth for the dev set)

# Edit these lists as needed (space-separated floats are fine via array).
read -r -a ALPHAS <<< "${ALPHAS:-0.25 0.5 0.75 1.0}"
read -r -a BETAS <<< "${BETAS:-0.0 0.3 0.6}"

for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    if ! python -c "import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); sys.exit(0 if a+b<=1.0+1e-9 else 1)" "${alpha}" "${beta}"; then
      printf 'Skipping alpha=%s beta=%s (alpha+beta > 1)\n' "${alpha}" "${beta}" >&2
      continue
    fi
    printf '\n========== alpha=%s beta=%s ==========\n' "${alpha}" "${beta}"
    python "${ROOT}/run.py" --alpha "${alpha}" --beta "${beta}"

    out_name="eval_metrics_alpha_${alpha}_beta_${beta}.json"
    python "${ROOT}/eval.py" \
      --predictions "${PRED}" \
      --groundtruth "${GT}" \
      --output_filename "${out_name}"
  done
done

printf '\nDone. Metrics under output/eval_results/\n'
