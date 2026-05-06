#!/usr/bin/env bash
# Fetch the zodiacal benchmarking test-case sets from
# https://github.com/OrbitalCommons/zodiacal-test-cases.
#
# Usage:
#   scripts/fetch_test_cases.sh                 # clone/update sibling ../zodiacal-test-cases
#   scripts/fetch_test_cases.sh --dest <path>   # clone/update at <path>
#   scripts/fetch_test_cases.sh --ssh           # use git@github.com remote (default: https)
#
# Exits 0 once a checkout exists and is up to date with origin/main.
set -euo pipefail

REMOTE_HTTPS="https://github.com/OrbitalCommons/zodiacal-test-cases.git"
REMOTE_SSH="git@github.com:OrbitalCommons/zodiacal-test-cases.git"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${repo_root}/../zodiacal-test-cases"
remote="${REMOTE_HTTPS}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            dest="$2"
            shift 2
            ;;
        --ssh)
            remote="${REMOTE_SSH}"
            shift
            ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -d "${dest}/.git" ]]; then
    echo "Updating existing checkout at ${dest}"
    git -C "${dest}" fetch --quiet origin
    git -C "${dest}" checkout --quiet main
    git -C "${dest}" pull --ff-only --quiet
else
    echo "Cloning ${remote} -> ${dest}"
    git clone --quiet "${remote}" "${dest}"
fi

echo
echo "Available sets in ${dest}:"
for d in "${dest}"/set*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    n="$(find "$d" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')"
    printf '  %-24s  %s cases\n' "$name" "$n"
done

cat <<EOF

Run the bench harness against a set, e.g.:
  cargo run -p zodiacal-tools --release -- bench-bundle \\
      --bundle-path <bundle.zdcl> \\
      --test-cases-dir ${dest}/set1-legacy
EOF
