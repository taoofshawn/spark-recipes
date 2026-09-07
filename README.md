
# collection of recipes and scripts for starting models on a 2-node dgx spark cluster

- stable models will be in main
- in-progress recipes will be in branches

# links:
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
  --multi-thread-streams=32 \
  --transfers=16 \
  --checkers=16 \
  --progress \
  --exclude "*.incomplete"

# rsync to fix permissions
rsync -av --delete --exclude '*.incomplete' ~/.cache/huggingface/hub/$MODEL/ spark-6d14:~/.cache/huggingface/hub/$MODEL/
