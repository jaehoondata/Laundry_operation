#!/usr/bin/env bash
# Run several gate-test configs at once on the lab GPU (one cloth uses ~0.5 GB and leaves the
# GPU mostly idle, so parallel processes finish a sweep much sooner). Run inside tmux:
#
#   bash sim/style3d_gate/run_sweep.sh SCRIPT [COMMON_ARGS...] -- NAME "ARGS" [NAME "ARGS" ...]
#
#   bash sim/style3d_gate/run_sweep.sh fold_test.py --solver vbd --no-video -- \
#       dt1 "--sim-dt 1e-3" tke4 "--table-ke 1e4" r2 "--contact-radius 2e-3"
#
# Each config writes to outputs/style3d_gate/<SWEEP>/<NAME>/ (SWEEP defaults to sweep_<time>)
# with its log next to it as <NAME>.log. MAX_JOBS caps how many run at once (default: all).
set -uo pipefail
cd "$(dirname "$0")/../.."

script=sim/style3d_gate/$1
shift
common=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do common+=("$1"); shift; done
[ $# -gt 0 ] && shift  # drop "--"
if [ $# -eq 0 ] || [ $(($# % 2)) -ne 0 ]; then
    echo "usage: $0 SCRIPT [COMMON_ARGS...] -- NAME \"ARGS\" [NAME \"ARGS\" ...]" >&2
    exit 1
fi

sweep=outputs/style3d_gate/${SWEEP:-sweep_$(date +%m%d_%H%M)}
max_jobs=${MAX_JOBS:-$(($# / 2))}
mkdir -p "$sweep"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate newton17

names=()
while [ $# -gt 0 ]; do
    name=$1 cfg=$2
    shift 2
    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do sleep 2; done  # no `wait -n` in bash 3.2
    # shellcheck disable=SC2086  # cfg is a space-separated arg list on purpose
    python "$script" "${common[@]}" $cfg --out "$sweep/$name" > "$sweep/$name.log" 2>&1 &
    echo "started $name ($cfg) pid $!"
    names+=("$name")
done
echo "watch: tail -n 3 $sweep/*.log"
wait

echo
for name in "${names[@]}"; do
    printf '%-16s %s\n' "$name" "$(grep -h "trials stable" "$sweep/$name.log" || echo "no summary (crashed? see $sweep/$name.log)")"
done
echo "results in $sweep/"
