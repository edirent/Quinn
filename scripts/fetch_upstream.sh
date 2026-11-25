#!/usr/bin/env bash
set -euo pipefail
mkdir -p .upstream
fetch() {
    local name="$1" url="$2" revision
    revision=$(awk -v key="$name:" '$1 == key {print $2}' configs/upstream.yaml)
    test -n "$revision"
    if [[ ! -d ".upstream/$name/.git" ]]; then
        git init -q ".upstream/$name"
    fi
    if ! git -C ".upstream/$name" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git -C ".upstream/$name" fetch --depth 1 "$url" "$revision"
    fi
    git -C ".upstream/$name" checkout -q --detach "$revision"
}
fetch kernelbench https://github.com/ScalingIntelligence/KernelBench.git
fetch kernelgym https://github.com/hkust-nlp/KernelGYM.git
