# deepseek-v4-flash-aiden - 1m token context

This is a docker compose file for starting vLLM with DeepSeek-V4-Flash-0731 on a 2-node DGX Spark cluster.  
The repository should be cloned to both nodes and started on the second node first. 

## Prep (both nodes)
```bash
cd ~/spark-recipes/deepseek-v4-flash-aiden/
docker pull $(grep image docker-compose.yml | awk '{print $NF}')
HF_MODEL=$(grep "MODEL_PATH:" docker-compose.yml | awk '{print $NF}')
HF_REVISION=$(grep "MODEL_REVISION:" docker-compose.yml | awk '{print $NF}')
hf download $HF_MODEL --revision $HF_REVISION
```

## Run
```bash
# Node 1 (follower): start first
docker compose --env-file .env --env-file .env.node1 up -d

# Node 0 (leader): start about 30s after
docker compose --env-file .env --env-file .env.node0 up -d
```

## Reference
The [discussion thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268) for this configuration

## files

| File | Purpose |
|---|---|
| `.env` | Shared config — must be customized |
| `.env.node0` | Per-node overrides for node 0 (leader) |
| `.env.node1` | Per-node overrides for node 1 (follower) |
| `docker-compose.yml` | compose file |


## Finding interface names for .env

### **Ethernet ports** (used for control-plane traffic): These will be used for `ETH_IF` and `ETH_IF2`

```bash
# List all Ethernet interfaces — pick the ones that are "BROADCAST" with link up:
ip addr show | grep -E '^[0-9]+: en'
```
These ports will change depending on if you used the left or right QSFP ports to connect the sparks to each other.

### **RoCE/InfiniBand ports** (used for NCCL data-plane traffic): These will be used for `IB_PORTS` (comma separated)

```bash
# Show RoCE interface names and link state:
ibdev2netdev

# Alternative: list RDMA devices directly:
rdma link show
```

### On DGX Spark each Ethernet port has a matching RoCE port on the same physical port.
Typical mapping:

| Ethernet (`ip addr`) | RoCE (`rdma link show`) |
|---|---|
| `enp1s0f0np0` | `rocep1s0f0` |
| `enP2p1s0f0np0` | `roceP2p1s0f0` |

Update the three variables in `.env` to match your system:

| .env variable | Typical value | What it controls |
|---|---|---|
| `ETH_IF` | `enp1s0f0np0` | GLOO, MPI, TP control-plane traffic |
| `ETH_IF2` | `enP2p1s0f0np0` | Second Ethernet port (for multi-interface NCCL socket) |
| `IB_PORTS` | `rocep1s0f0,roceP2p1s0f0` | NCCL InfiniBand data-plane traffic |
