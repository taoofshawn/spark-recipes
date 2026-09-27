#!/usr/bin/env bash
# glm-v53-flash-4x-noswitch — add docker log rotation to every node's daemon.json,
# preserving existing per-node settings (nvidia runtime on ranks 0-1, containerd-
# snapshotter=false on ranks 2-3). Site-level host change, like the snapshotter fix.
# Applied cluster-wide so all ranks keep the same logging behavior.
set -euo pipefail

for n in spark-0f0b spark-6d14 spark-6d90 spark-6d24; do
  echo "=== $n"
  ssh -o BatchMode=yes "$n" 'bash -s' <<'REMOTE'
set -euo pipefail
if sudo -n grep -q nvidia-container-runtime /etc/docker/daemon.json 2>/dev/null; then
  sudo -n tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "log-driver": "json-file",
    "log-opts": { "max-size": "256m", "max-file": "4" }
}
JSON
else
  sudo -n tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
    "features": { "containerd-snapshotter": false },
    "log-driver": "json-file",
    "log-opts": { "max-size": "256m", "max-file": "4" }
}
JSON
fi
sudo -n python3 -c "import json; json.load(open('/etc/docker/daemon.json'))" && echo "daemon.json valid"
REMOTE
done
