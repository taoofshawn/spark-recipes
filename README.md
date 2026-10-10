
# collection of recipes and scripts for starting models on a 4-node dgx spark switchless ring

- active lanes: `glm-v53-flash-4x-noswitch` (FP8) and `glm-v53-flash-nvfp4-4x-noswitch` (NVFP4) — switching: [LANE-SWITCHING.md](LANE-SWITCHING.md)
- everything else is archived under `.archived/` (2-node era); in-progress work on branches

# links:
- [lane switching runbook](LANE-SWITCHING.md): switching the GLM-5.3-Flash FP8/NVFP4 lanes on the 4-node switchless ring (start here for that task)
- [discussion forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719): the source for most/all of this information
- [dgx spark hardware](https://marketplace.nvidia.com/en-us/developer/dgx-spark/)
- [connectx-7 cable](https://docs.nvidia.com/sync/latest/cluster-assistant.html#check-network-performance) : model `NJAAKK-N911`- interconnect 2 sparks

# tips

### copy large stuff between sparks
```bash
# get model dir
ls -al ~/.cache/huggingface/hub/ # get model dir
MODEL=models--Intel--GLM-5.3-Flash-W4A16-AutoRound

# rclone for fast data transfer (use cx7 link for transfer)
rclone sync  ~/.cache/huggingface/hub/$MODEL/ spark-6d14:.cache/huggingface/hub/$MODEL/ \
  --copy-links \
  --multi-thread-streams=64 \
  --transfers=32 \
  --checkers=32 \
  --progress \
  --exclude "*.incomplete"

# rsync to fix permissions
rsync -av --delete --exclude '*.incomplete' ~/.cache/huggingface/hub/$MODEL/ spark-6d14:~/.cache/huggingface/hub/$MODEL/
