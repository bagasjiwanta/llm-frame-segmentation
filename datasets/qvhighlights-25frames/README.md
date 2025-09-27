---
license: mit
task_categories:
  - video-text-to-text
language:
  - en
pretty_name: "QVHighlights - 25 Frames - 3 Frame Variations"
configs:
  - config_name: p1
    data_files:
      - split: train
        path: p1/train_v1.json
      - split: val
        path: p1/val_v1.json
      - split: test
        path: p1/test.json
  - config_name: p2
    data_files:
      - split: train
        path: p2/train_v1.json
      - split: val
        path: p2/val_v1.json
      - split: test
        path: p2/test.json
---

# QVHighlights 

- Sets: train and val. Test set will be available soon.
- Number of frames per video: 25 frames.
- Frame variations: 3
- Answer format: 25 frame segmentation mask (e.g. 1010100101100101011110011)
- Images are in videos.tar.gz (no inside folders).
- Each image is noted by `{video name}_frame{frame:03d}_var{var:d}.jpg` where var2 is the middle one. The variations are separated 0.125 seconds or around 7 frames for 30fps videos (most are 30fps).
- The subset name `p{n}` is the n^th prompt style. So, p1 = prompt style 1, p2 = prompt style 2, etc.
- The suffix name v1 means the original. In v2, the scores that is 0.0 are re-weighted using nearby tokens with small weight so the scores turns to be around 0.0 to 0.2 (less harsh for the soft losses). This change does not really change the overall distribution as they are only a handful of corrections.

## To Download

### Install dependencies

```bash
pip install huggingface-hub[cli]
hf auth login
```

### Download

```bash
hf download jwnt4/qvhighlights-25frames \
  --repo-type dataset \
  --local-dir qvhighlights-25frames
```

### Extract

Set the amount of thread with `pigz -p`

```bash
apt install pigz pv
cd qvhighlights-25frames
pv videos.tar.gz | pigz -dc -p 2 | tar -x
rm videos.tar.gz
```