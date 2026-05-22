#!/usr/bin/env bash
# Parallel obs-12 retrain launcher (PPO x3 + SAC x3 + Rainbow x3, 3 seeds each).
#
# Run ONLY AFTER rebuilding Builds/BoatSailing_Mac.app in Unity with
# VectorObservationSize=12 (restores the previously-truncated weather-vane z).
#
# Rolling pool: at most MAX runs concurrent (default 6, tuned for the 64GB
# Mac mini M4 Pro = 14-core CPU / 10 performance cores). Each run = 1 headless
# Unity sim + 1 trainer ~= 1.5 cores; 6 fits the 10 P-cores with headroom.
#
# Usage:
#   bash code/training/run_retrain_obs12.sh            # MAX=6
#   MAX=9 bash code/training/run_retrain_obs12.sh       # push all 9
# Logs: code/results/<run-id>/train_stdout.log
set -uo pipefail
cd "$(dirname "$0")/.."   # -> code/

ENV_NOEXT="Builds/BoatSailing_Mac"   # mlagents + rainbow auto-append .app
MAX="${MAX:-6}"

if [ ! -d "${ENV_NOEXT}.app" ]; then
  echo "ERROR: ${ENV_NOEXT}.app missing. Build it in Unity Editor first." >&2
  exit 1
fi
mkdir -p results

launch() {
  local algo="$1" seed="$2" port="$3" rid log
  rid="${algo}_seed_${seed}"
  mkdir -p "results/${rid}"
  log="results/${rid}/train_stdout.log"
  echo "[launch] ${rid} (port ${port}) -> ${log}"
  case "$algo" in
    ppo)
      conda run -n mlagents mlagents-learn Config/moveToGoal.yaml \
        --run-id="${rid}" --seed="${seed}" --base-port="${port}" \
        --env="${ENV_NOEXT}" --no-graphics --force >"${log}" 2>&1 ;;
    sac)
      conda run -n mlagents mlagents-learn Config/sac_boat.yaml \
        --run-id="${rid}" --seed="${seed}" --base-port="${port}" \
        --env="${ENV_NOEXT}" --no-graphics --force >"${log}" 2>&1 ;;
    rainbow)
      conda run -n tianshou python training/rainbow_boat.py \
        --seed="${seed}" --run-id="${rid}" --total-steps 5000000 \
        --binary "${ENV_NOEXT}" >"${log}" 2>&1 ;;
  esac
}

# algo seed base-port  (ml-agents uses base-port; rainbow uses worker_id=seed off 5005)
JOBS=(
  "ppo 0 5010"     "ppo 1 5011"     "ppo 2 5012"
  "sac 0 5020"     "sac 1 5021"     "sac 2 5022"
  "rainbow 0 5005" "rainbow 1 5006" "rainbow 2 5007"
)

for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do sleep 10; done
  # shellcheck disable=SC2086
  launch $j &
  sleep 8   # stagger Unity port binding to avoid race
done
wait
echo "[done] all 9 runs finished. Next: re-run frozen-policy eval + regenerate figures."
