# ComfyUI FaceID

Offline face identity scoring for ComfyUI. Answers one question with a number:

> Is the person in this image the same person as in my reference photos?

Built for checking character / person LoRAs: generate a batch, score every image
against a few reference photos, see instantly which seeds drifted off-model.

- **No cloud calls.** Everything runs locally on ONNX Runtime.
- **No `insightface` package.** The official ONNX models are used directly, so a
  broken or uncompiled `insightface` install cannot break this node.
- **No extra models beyond one 288 MB folder** (`buffalo_l`).

---

## How it works

1. **SCRFD** (`det_10g.onnx`) finds faces and 5 landmarks.
2. The landmarks are aligned to the standard **ArcFace 112×112** template.
3. **ArcFace** (`w600k_r50.onnx`) turns the aligned crop into a 512-d embedding.
4. Two embeddings are compared with **cosine similarity** (`1.0` = identical).

Small faces (full-body shots) are automatically re-analysed on a 2× upscale, so
the 112×112 crop is built from more real pixels instead of a blurry upscale.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/j7708git/comfyui-face-id.git
```

Install the one dependency in the **ComfyUI Python environment** (skip if you
already have `onnxruntime` or `onnxruntime-gpu`):

```bash
pip install onnxruntime>=1.16.0
```

Restart ComfyUI. The nodes appear under the **FaceID** category.

### Models

The pack needs the `buffalo_l` ONNX files:

```
ComfyUI/models/insightface/models/buffalo_l/
├─ det_10g.onnx
└─ w600k_r50.onnx
```

If you already use IPAdapter FaceID or ReActor, they are probably there already.
Otherwise:

```bash
python custom_nodes/comfyui-face-id/tools/download_models.py
```

The script also accepts `--dir` and falls back to `~/.insightface/models/buffalo_l`.

---

## Nodes

### FaceID Reference

Averages one or more reference photos into a single identity vector.

| Input | Default | Notes |
|---|---|---|
| `image` | – | Batch of photos of the same person (3–5 works well) |
| `det_size` | 1280 | Detector resolution; raise it if faces are small |
| `face_index` | -1 | `-1` = largest face in the photo |
| `min_face_px` | 80 | Below this, the image is re-analysed on a 2× upscale |

Outputs: `ref` (FACE_REF), `used` (how many photos had a usable face),
`info` (text, including a suggested threshold when you feed 2+ photos).

### FaceID Score

| Input | Default | Notes |
|---|---|---|
| `ref` | – | From FaceID Reference |
| `image` | – | The image to check |
| `threshold` | 0.30 | Cosine similarity needed for PASS |
| `det_size` | 1280 | |
| `face_index` | -1 | `-1` = score every face, best match wins |
| `min_face_px` | 80 | |
| `draw` | true | Draws the box and the score on the output image |

Outputs: `image` (annotated), `score`, `distance` (`1 - score`), `verdict`
(`PASS` / `DRIFT` / `NO_FACE`), `is_same` (BOOLEAN), `info` (text).

### FaceID Batch Score

Same as above for a whole batch, plus an optional CSV report written to
`ComfyUI/output/faceid/scores_<timestamp>.csv`.

Outputs: `images` (annotated batch), `report` (text table), `mean_score`,
`min_score`, `fails` (count of anything that is not PASS).

---

## Wiring

```
Load Image ─┐
Load Image ─┼─► ImageBatch ─► FaceID Reference ──┐
Load Image ─┘                    (ref)           │
                                                 ▼
Your LoRA workflow ─► Save Image ─► FaceID Score ─┬─► score / is_same
                                                  └─► ImpactConditionalBranch
```

`is_same` can drive any boolean switch (Impact Pack's `ImpactConditionalBranch`,
for example) to route rejects somewhere else.

ComfyUI graphs have no loops, so "regenerate until the score is high enough"
has to be done by hand or by an external script that calls the API.

---

**Multi-face images.** Every detected face is embedded and the **highest**
similarity wins, so the question answered is "is this person anywhere in this
image". That keeps group shots, couples and small-in-frame characters working.
Keep reference photos to **one person per photo** - the reference node uses the
largest face of each photo.

---

## Calibrating the threshold

The widely quoted `0.68` cosine distance comes from large, sharp, real-face
photos. Generated images at 100 px faces score lower even when they are the
same person, so **calibrate on your own data**:

1. Feed 3–5 reference photos into `FaceID Reference`. The `info` output shows
   their self-similarity and a suggested threshold.
2. Score a few images you are sure are correct. Take the lowest score and stop
   a bit below it.

Measured with `buffalo_l` on real photos and SDXL/MOSS character-LoRA output:

| Relation | Score |
|---|---|
| An image against itself | 1.00 |
| Same person, real photos (different years and angles) | 0.51 – 0.58 |
| Same person, generated at ~100 px faces | 0.30 – 0.66 |
| Different people, real photos | 0.00 – 0.15, occasionally up to 0.28 |

The separation is real but not huge, so treat the score as a **ranking and
coarse filter**, not a verdict. `0.30` is a reasonable starting threshold; the
`0.25 – 0.35` band is where you should look at the images yourself.

---

## What it can and cannot do

**Works well**

- Real-photo faces, front to 3/4 profile, 60 px and larger.
- Ranking seeds / strengths: "which of these 8 images drifted".
- Finding the one broken image in a batch of 50.

**Unreliable**

- Faces smaller than ~40 px, heavy occlusion, extreme profile, motion blur.
- Stylised and anime faces. `buffalo_l` is trained on real faces; it often finds
  nothing on illustration output. Use a CLIP/DINOv2 image similarity instead.
- "Same character" judged by outfit, hair colour or silhouette. That is not
  facial geometry; a whole-image embedding is a different metric.

`NO_FACE` is reported separately and never counted as a PASS, because "could not
measure" and "different person" are different problems.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not find det_10g.onnx` | Run `python custom_nodes/comfyui-face-id/tools/download_models.py`, then restart ComfyUI |
| `onnxruntime is required` | `pip install onnxruntime` in the ComfyUI Python environment |
| Everything is `NO_FACE` | Raise `det_size` to 1536/2048, or check whether the image is illustration output (see above) |
| Scores look random | Feed 3–5 clean reference photos, one person per photo, face at least 60 px |
| Very slow | A CPU-only `onnxruntime` runs ~0.3–0.6 s per image; install `onnxruntime-gpu` to use CUDA |

Sanity check without starting ComfyUI:

```bash
python custom_nodes/comfyui-face-id/tools/selftest.py path/to/a/face/photo.jpg
```

It verifies the models load and that an image scores `1.00` against itself.

---

## Repository layout

```
comfyui-face-id/
├─ __init__.py            node registration
├─ core.py                detection, alignment, embedding, similarity
├─ nodes.py               the three ComfyUI nodes
├─ requirements.txt       onnxruntime only
├─ tools/download_models.py   fetch buffalo_l
├─ tools/selftest.py      quick sanity check
└─ LICENSE                MIT (code only)
```

Model weights are **not** included in this repository. `models/`, `*.onnx`,
`*.safetensors` and `output/` are git-ignored on purpose.

---

## Privacy

- No network access at runtime. No telemetry, no uploads.
- The optional `tools/download_models.py` is the only thing that touches the
  network, and it only downloads the public model archive.
- Everything is written under `ComfyUI/output/faceid/`.

---

## License

- **This code:** MIT, see `LICENSE`.
- **InsightFace model weights (`buffalo_l`):** released by the InsightFace
  authors under **CC BY-NC 4.0 — non-commercial use only**. The download script
  fetches them from the official release page; they are not part of this
  repository. If you need a commercially licensed alternative, the
  [AuraFace](https://huggingface.co/fal/AuraFace-v1) models are Apache-2.0 and
  use the same embedding layout.

---

## 中文快速開始

1. 把這個資料夾放到 `ComfyUI\custom_nodes\`，重開 ComfyUI。
2. 確認 `ComfyUI\models\insightface\models\buffalo_l\` 裡有 `det_10g.onnx` 和
   `w600k_r50.onnx`（裝過 IPAdapter FaceID 的話通常已經有了）。
3. 在節點搜尋 `FaceID`，接成：參考照 → `FaceID Reference` → `FaceID Score`。
4. 先跑幾張你確定正確的圖，把 `threshold` 調到它們都通過為止。

分數是餘弦相似度，1.0 代表完全相同。臉太小的話把 `det_size` 調到 1536 或 2048。
