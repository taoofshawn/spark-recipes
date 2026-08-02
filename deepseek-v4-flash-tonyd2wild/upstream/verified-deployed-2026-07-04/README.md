Byte-exact files pulled from the live 2026-07-04 deployment (Asusi head).
docker-compose.dspark.yml = compose as run (image=probe-c-p2b, patch3 bind-mount, spec=5 hardcoded, port 8888).
env.dspark = node/network vars (NOTE: its SPECULATIVE_CONFIG/MTP_NUM_TOKENS=3 are DEAD — the compose hardcodes spec=5).
patch3-scheduler.py = roady001 Patch 3 scheduler.py (bind-mounted; also in recipe/overlay/ for stage-c builds).
See ../DEFAULT-CONFIG.md for the full canonical snapshot.
