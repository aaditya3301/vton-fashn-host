# Virtual Try-On Experiments — Modal Deployments

This repository contains **two** serverless virtual try-on (VTON) deployments on [Modal](https://modal.com), built and tested for evaluation purposes:

1. **[FASHN VTON v1.5](#part-1-fashn-vton-v15)** — Apache 2.0 licensed, **commercially usable**, maskless pipeline
2. **[OOTDiffusion (half-body)](#part-2-ootdiffusion-half-body)** — CC BY-NC-SA 4.0, **non-commercial only**, mask-based pipeline

Both were deployed on Modal with auto-scaling A10G GPUs and benchmarked head-to-head against Google's Vertex AI Virtual Try-On API.

> **TL;DR after running both**: Open-source VTON quality is below paid APIs (Vertex VTO). FASHN is the better open-source option for commercial use. OOTD is older and quality is mixed on non-studio inputs. See [Honest Comparison](#honest-comparison) at the bottom.

---

## Table of Contents

- [Why Two Deployments](#why-two-deployments)
- [Part 1: FASHN VTON v1.5](#part-1-fashn-vton-v15)
- [Part 2: OOTDiffusion (Half-Body)](#part-2-ootdiffusion-half-body)
- [How to Improve (Both)](#how-to-improve-both)
- [Honest Comparison](#honest-comparison)
- [Honest Limitations](#honest-limitations)
- [License Summary](#license-summary)
- [Credits](#credits)

---

## Why Two Deployments

We tested both to answer one question:

> **Can we self-host an open-source VTON model that's good enough to replace Google's Vertex VTO API at $0.06/image?**

Both deployments share the same:

- Modal serverless setup
- A10G GPU (24 GB)
- Synchronous HTTP endpoint pattern
- Cost tracking
- Client testing scripts

They differ in:

- The underlying model
- License (commercial vs. non-commercial)
- Whether they need explicit masks (FASHN is maskless, OOTD is mask-based)
- Quality on real-world inputs

---

# Part 1: FASHN VTON v1.5

> ✅ **License**: Apache 2.0 — fully commercially usable.

## What FASHN Does

Given a person image and a garment image, FASHN VTON v1.5 generates a photorealistic image of the person wearing that garment. It works directly in pixel space and is **maskless** — no segmentation step required. Just three inputs: `person_image`, `garment_image`, and a `category` (`tops`, `bottoms`, or `one-pieces`).

**Pipeline**:

```
person.jpg + garment.jpg + category
              ↓
        FASHN VTON v1.5
        (single-stage, maskless)
              ↓
          output PNG
```

## FASHN Architecture

```
┌──────────────────────────────────────────┐
│  Client (laptop / future Next.js)        │
└──────────────┬───────────────────────────┘
               │ HTTPS POST
               ↓
┌──────────────────────────────────────────┐
│  Modal Web Endpoint (auto-scales)        │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Modal GPU Container (A10G, 24GB)        │
│  - FASHN model loaded once at boot       │
│  - Stays warm 2 min between requests     │
│  - Scales to zero when idle              │
└──────────────┬───────────────────────────┘
               ↓
       Output PNG (base64) returned
```

## FASHN Tech Stack

| Layer | Technology |
|---|---|
| **Diffusion model** | FASHN VTON v1.5 (1.2B params, flow-matching) |
| **Compute** | Modal serverless |
| **GPU** | NVIDIA A10G (24 GB VRAM) |
| **Framework** | PyTorch + Hugging Face |
| **Pose detection** | DWPose (built into FASHN) |
| **API framework** | FastAPI (auto-loaded by Modal) |
| **Client** | Python `requests` |

**Container image** (built once, cached forever):

- Debian Slim + Python 3.11
- PyTorch 2.4.0
- FASHN repo cloned + installed in editable mode (`pip install -e .`)
- `onnxruntime-gpu` for accelerated pose detection
- `fastapi[standard]` for the web endpoint
- ~2 GB of model weights (TryOnModel + DWPose + human parser)

## FASHN Project Structure

```
fashn-modal/
├── modal_app.py        # Modal deployment
├── client.py           # HTTP client for testing
├── requirements.txt    # Local Python deps
├── test_images/
│   ├── person.jpg
│   └── garment.jpg
├── cost_log.jsonl      # Auto-generated per-call cost log
└── result.png          # Latest generated try-on
```

## FASHN Setup

### 1. Create the project folder

```bash
mkdir fashn-modal && cd fashn-modal
mkdir test_images
cp /path/to/person.jpg test_images/person.jpg
cp /path/to/garment.jpg test_images/garment.jpg
```

### 2. Create `requirements.txt`

```txt
modal>=0.64.0
requests>=2.31.0
Pillow>=10.0.0
```

### 3. Set up Python env and install Modal

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Authenticate with Modal

```bash
modal token new
modal profile current   # verify
```

### 5. Set Modal spending cap (CRITICAL)

Go to [modal.com/settings/billing](https://modal.com/settings/billing) → Spending Limits → set **$10/month**.

Without this, a runaway loop in your code could burn $50+/hour.

### 6. Save `modal_app.py` and `client.py`

(Use the FASHN versions from your project; they reference FASHN VTON v1.5.)

## FASHN Running the App

### First run (one-off, no deploy)

```bash
modal run modal_app.py --person-path test_images/person.jpg --garment-path test_images/garment.jpg --out result.png
```

**First time**: ~5-10 minutes (downloads PyTorch + clones FASHN + downloads ~2GB weights). Subsequent runs: ~30s cold start + ~12-20s inference.

### Deploy persistent endpoint

```bash
modal deploy modal_app.py
```

Output:

```
✓ Created web endpoint => https://aaditya3301--fashn-vton-vtonservice-web.modal.run
```

Copy the URL into `client.py`'s `MODAL_URL`.

## FASHN Testing the API

### Single call

```bash
python client.py test_images/person.jpg test_images/garment.jpg tops result.png 20 1.5 42
```

Args: `<person> <garment> <category> <out> <timesteps> <guidance> <seed>`

Categories: `tops` | `bottoms` | `one-pieces`.

### From Python

```python
import base64, requests
from pathlib import Path

person = base64.b64encode(Path("test_images/person.jpg").read_bytes()).decode()
garment = base64.b64encode(Path("test_images/garment.jpg").read_bytes()).decode()

r = requests.post(
    "https://aaditya3301--fashn-vton-vtonservice-web.modal.run",
    json={
        "person_b64": person,
        "garment_b64": garment,
        "category": "tops",
        "num_timesteps": 20,
        "guidance_scale": 1.5,
        "seed": 42,
    },
    timeout=300,
)
Path("result.png").write_bytes(base64.b64decode(r.json()["image_b64"]))
```

### From curl

```bash
PERSON_B64=$(base64 -w 0 test_images/person.jpg)
GARMENT_B64=$(base64 -w 0 test_images/garment.jpg)

curl -X POST https://aaditya3301--fashn-vton-vtonservice-web.modal.run \
  -H "Content-Type: application/json" \
  -d "{\"person_b64\":\"$PERSON_B64\",\"garment_b64\":\"$GARMENT_B64\",\"category\":\"tops\",\"num_timesteps\":20,\"guidance_scale\":1.5,\"seed\":42}" \
  | jq -r '.image_b64' | base64 -d > result.png
```

## FASHN Parameter Tuning

| Parameter | Range | Default | What it does |
|---|---|---|---|
| `num_timesteps` | 10-50 | **20** | More steps = potentially sharper. Diminishing returns past 25. |
| `guidance_scale` | 1.0-3.0 | **1.5** | Higher = stricter prompt adherence. Above 2.5 often produces artifacts. |
| `seed` | -1 (random) or any int | **42** | Different seeds → different fabric drape interpretations. |
| `category` | tops/bottoms/one-pieces | **tops** | Required. Map your garment correctly. |

**Recommended starting combos**:

| Goal | timesteps | guidance | seed |
|---|---|---|---|
| Best quality (default) | 20 | 1.5 | 42 |
| Faster | 15 | 1.5 | -1 |
| Strictest match | 25 | 2.0 | 42 |
| Try multiple variants | 20 | 1.5 | 1, 7, 42, 100 |

## FASHN Cost Monitoring

Every call returns:

- `cost_usd_inference_only` — pure GPU compute time
- `cost_usd_realistic` — includes amortized 2-min idle window
- `cost_inr_realistic` — INR equivalent

Local cost summary:

```bash
python client.py --cost
```

Modal dashboard (authoritative): [modal.com/apps](https://modal.com/apps) → click `fashn-vton`.

**Realistic cost expectations**:

| Usage pattern | Cost per image |
|---|---|
| Sporadic testing | $0.03-0.06 |
| Bursty (10 calls/2 min) | $0.005-0.010 |
| Sustained (50+/hour) | $0.002-0.004 |

## FASHN Stopping & Restarting

```bash
# Stop billing immediately
modal app stop fashn-vton

# List apps
modal app list

# Fully delete
modal app delete fashn-vton

# Come back later (image cached, ~10s)
modal deploy modal_app.py
```

## FASHN Troubleshooting

| Error | Fix |
|---|---|
| `Web endpoint Functions require FastAPI to be installed` | Add `"fastapi[standard]"` to `pip_install` in `modal_app.py` |
| `ModuleNotFoundError: No module named 'PIL'` (locally) | `pip install Pillow` |
| Output is rotated 90° | Phone EXIF orientation. Use `ImageOps.exif_transpose()` on inputs (already in client) |
| Output looks unrelated to garment | Wrong param names. FASHN uses `num_timesteps` and `guidance_scale`, not `num_inference_steps` |
| All outputs look identical with random seeds | Param wasn't reaching the pipeline. Verify in server logs |
| `403 Forbidden` from endpoint | Wrong Modal workspace. Run `modal profile current` |

---

# Part 2: OOTDiffusion (Half-Body)

> ⚠️ **License Notice**: OOTDiffusion is licensed under **CC BY-NC-SA 4.0** (non-commercial only). This deployment is intended strictly for personal testing, research, and learning. **Do not use for any commercial product or revenue-generating service.**

## What OOTD Does

Given a photo of a person and a photo of a garment (preferably a flat-lay), OOTDiffusion returns a generated image of that person wearing that garment. Unlike FASHN, it requires multiple preprocessing stages (pose detection + body parsing + mask generation) before the actual diffusion step.

**Pipeline**:

```
person.jpg + garment.jpg
        ↓
   OpenPose (keypoint detection)
        ↓
   Human Parsing (body part segmentation)
        ↓
   Mask Generation (where to inpaint)
        ↓
   OOTDiffusion (latent diffusion try-on)
        ↓
    output PNG
```

## OOTD Architecture

```
┌──────────────────────────────────────────┐
│  Client (laptop / future Next.js)        │
└──────────────┬───────────────────────────┘
               │ HTTPS POST
               ↓
┌──────────────────────────────────────────┐
│  Modal Web Endpoint (auto-scales)        │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Modal GPU Container (A10G, 24GB)        │
│  - OOTD model loaded once at boot        │
│  - Stays warm 2 min between requests     │
│  - Scales to zero when idle              │
└──────────────┬───────────────────────────┘
               ↓
       Output PNG (base64) returned
```

## OOTD Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Diffusion model** | OOTDiffusion (half-body) | Best open-source VTON quality, March 2024 |
| **Compute** | Modal serverless | Auto-scaling GPUs, billed per-second |
| **GPU** | NVIDIA A10G (24GB VRAM) | Sufficient for OOTD's ~10GB weights |
| **Framework** | PyTorch 2.0.1 + diffusers 0.24.0 | Pinned to OOTD's tested versions |
| **Pose detection** | DWPose / OpenPose (ONNX) | Used internally by OOTD |
| **Body parsing** | FashnHumanParser | OOTD's mask generation |
| **API framework** | FastAPI (auto-loaded by Modal) | HTTP endpoint with auto-docs |
| **Client** | Python `requests` | Calls the deployed endpoint |

**Container image** (built once, cached forever):

- Debian Slim + Python 3.10
- `git`, `git-lfs`, OpenGL libs (for OpenCV)
- PyTorch 2.0.1 + CUDA 11.8
- All of OOTD's exact dependencies (numpy 1.24.4, scipy 1.10.1, etc.)
- huggingface_hub pinned to 0.20.3 (compatibility fix)
- ~21GB of model weights (OOTD checkpoints + CLIP ViT-Large)

## OOTD Project Structure

```
ootd-modal/
├── modal_app.py        # Modal deployment (image build + GPU class + endpoints)
├── client.py           # HTTP client for testing the deployed endpoint
├── sweep_test.py       # Parameter sweep (runs multiple configs in one go)
├── requirements.txt    # Local Python deps (just for the client)
├── test_images/
│   ├── person.jpg      # Input person photo
│   └── garment.jpg     # Input garment photo (flat-lay preferred)
├── cost_log.jsonl      # Auto-generated per-call cost log
└── result.png          # Latest generated try-on image
```

## OOTD Setup

### 1. Create the project folder

```bash
mkdir ootd-modal && cd ootd-modal
mkdir test_images
cp /path/to/your/person.jpg test_images/person.jpg
cp /path/to/your/garment.jpg test_images/garment.jpg
```

### 2. Create `requirements.txt`

```txt
modal>=0.64.0
requests>=2.31.0
Pillow>=10.0.0
```

### 3. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Authenticate with Modal

```bash
modal token new
modal profile current
```

### 5. Set Modal spending limits (CRITICAL)

Go to [modal.com/settings/billing](https://modal.com/settings/billing) → Spending Limits → set monthly cap to **$10**.

### 6. Get the code

Save `modal_app.py`, `client.py`, and `sweep_test.py` from the project root.

## OOTD Running the App

### First run: build + run once (no deploy)

```bash
modal run modal_app.py --person-path test_images/person.jpg --garment-path test_images/garment.jpg --out result.png
```

**What happens**:

1. Modal builds the container image (FIRST TIME ONLY: ~15-25 minutes)
   - Downloads PyTorch + CUDA
   - Clones OOTDiffusion repo
   - Pulls 21GB of model weights from HuggingFace
2. Boots a GPU container (~30-60s cold start)
3. Loads model into VRAM (~10-15s)
4. Runs inference (~12-20s)
5. Saves result locally as `result.png`
6. Container shuts down

Subsequent runs: ~30s cold start + ~15s inference = **~45s end-to-end**.

### Deploy persistent endpoint

```bash
modal deploy modal_app.py
```

Output:

```
✓ Created web endpoint => https://aaditya3301--ootd-vton-ootdservice-web.modal.run
```

Copy that URL into `client.py`'s `MODAL_URL`.

## OOTD Testing the API

### Single call

```bash
python client.py test_images/person.jpg test_images/garment.jpg result.png 20 2.0 -1
```

Args: `<person> <garment> <out> <steps> <scale> <seed>`

### Parameter sweep (find best params)

```bash
python sweep_test.py test_images/person.jpg test_images/garment.jpg
```

Generates **11 outputs** with different scale/seed/steps combos. Cost: ~$0.40-0.50 total.

### From curl

```bash
PERSON_B64=$(base64 -w 0 test_images/person.jpg)
GARMENT_B64=$(base64 -w 0 test_images/garment.jpg)

curl -X POST https://aaditya3301--ootd-vton-ootdservice-web.modal.run \
  -H "Content-Type: application/json" \
  -d "{\"person_b64\":\"$PERSON_B64\",\"garment_b64\":\"$GARMENT_B64\",\"num_steps\":20,\"image_scale\":2.0,\"seed\":-1}" \
  | jq -r '.image_b64' | base64 -d > result.png
```

### From Python

```python
import base64, requests
from pathlib import Path

person = base64.b64encode(Path("test_images/person.jpg").read_bytes()).decode()
garment = base64.b64encode(Path("test_images/garment.jpg").read_bytes()).decode()

r = requests.post(
    "https://aaditya3301--ootd-vton-ootdservice-web.modal.run",
    json={"person_b64": person, "garment_b64": garment, "num_steps": 20, "image_scale": 2.0, "seed": -1},
    timeout=300,
)
Path("result.png").write_bytes(base64.b64decode(r.json()["image_b64"]))
```

## OOTD Parameter Tuning

OOTD exposes three knobs:

| Parameter | Range | Default | What it does |
|---|---|---|---|
| `num_steps` | 10-50 | **20** | More steps = potentially sharper, slower, more expensive. Diminishing returns past 25. |
| `image_scale` | 1.0-5.0 | **2.0** | How strictly the output adheres to the garment. Higher = stricter, can over-fit. |
| `seed` | -1 (random) or any int | **-1** | Reproducibility. Different seeds give different drape interpretations. |

**Recommended starting combos**:

| Goal | steps | scale | seed |
|---|---|---|---|
| Best quality | 25 | 2.5 | run multiple seeds (1, 7, 42, 100) |
| Fastest | 15 | 2.0 | -1 |
| Strictest garment match | 20 | 3.0 | 42 |
| Most natural drape | 30 | 1.5 | 42 |

## OOTD Cost Monitoring

Every call returns:

- `cost_usd_inference_only` — pure GPU compute time
- `cost_usd_realistic` — includes amortized 2-min idle window
- `cost_inr_realistic` — INR equivalent

Local cost summary:

```bash
python client.py --cost
```

Modal dashboard: [modal.com/apps](https://modal.com/apps) → click `ootd-vton`.

**Realistic cost expectations**:

| Usage pattern | Cost per image |
|---|---|
| Sporadic testing | $0.04-0.08 |
| Bursty (10 calls/2 min) | $0.005-0.012 |
| Sustained (50+/hour) | $0.003-0.005 |

## OOTD Stopping & Restarting

```bash
# Stop
modal app stop ootd-vton
modal app list

# Fully delete
modal app delete ootd-vton

# Restart later
cd ootd-modal
modal deploy modal_app.py
```

## OOTD Troubleshooting

| Error | Fix |
|---|---|
| `git: 'lfs' is not a git command` | Add `"git-lfs"` to `apt_install` |
| `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` | Pin `huggingface_hub==0.20.3` |
| `ModuleNotFoundError: No module named 'matplotlib'` (or similar) | Use OOTD's exact `requirements.txt` versions |
| `ModuleNotFoundError: No module named 'ootd.utils_ootd'` | Add `sys.path.insert(0, "/ootd/run")` and `from utils_ootd import get_mask_location` |
| CUDA out of memory | Switch GPU to `L40S` (48GB) or `A100-40GB` |
| Bad output quality / artifacts | Use flat-lay garment images. OOTD trained on studio photos, struggles with casual ones |
| Endpoint timeout | Increase `timeout=600` in both server and client |
| Image build timeout (>30 min) | Re-run `modal run` — Modal resumes from cached layers |

---

## How to Improve (Both)

### Quality improvements

1. **Add face restoration post-processing** (`gfpgan` or `codeformer`)
   - Add ~1-2s to inference, ~free
   - Big quality lift for portrait shots
2. **Multi-pass with seed selection**
   - Generate 4 samples, score with CLIP similarity, return best
   - Costs 4x but gets best-of-N quality
3. **Pre-process inputs**
   - Background removal on garment with `rembg`
   - Auto-crop person to standard half-body framing
4. **For OOTD specifically**: switch to full-body model
   - Use `OOTDiffusionDC` instead of `OOTDiffusionHD`
   - Supports `upperbody`, `lowerbody`, and `dress` categories

### Performance improvements

1. **Reduce cold start**
   - Compile model with `torch.compile()` once
   - Could drop cold start from 60s → 20s
2. **Lower idle window**
   - Change `scaledown_window=120` to `30` for sporadic traffic
   - Cuts realistic cost per call by ~75%
3. **Batch inference**
   - Process multiple person+garment pairs per request
   - Modal supports this with `@modal.batched()`

### Production readiness

1. **Authentication** — API key check, Modal Secrets, rate-limiting
2. **Async processing** — webhook-based job queue, better UX for slow inference
3. **Image storage** — upload to R2/S3, return signed URL (smaller responses)
4. **Monitoring** — Sentry for errors, PostHog for usage
5. **Frontend** — Next.js + shadcn UI with progress indicators

---

## Honest Comparison

After running both deployments and comparing to Google Vertex VTO on the same person + garment pair:

| Criterion | FASHN VTON v1.5 | OOTDiffusion (HD) | Vertex VTO (paid) |
|---|---|---|---|
| **Quality** | Good (~75-80% of Vertex) | Mixed (highly input-dependent) | Best |
| **Speed** | ~12-20s | ~30-40s | ~43s |
| **License** | ✅ Apache 2.0 (commercial) | ❌ CC BY-NC-SA 4.0 (no commercial) | Paid API |
| **Setup difficulty** | Easy | Hard (dep hell) | Trivial (just API) |
| **Cost (sporadic)** | $0.03-0.06/image | $0.04-0.08/image | $0.06/image |
| **Cost (sustained)** | $0.002-0.004/image | $0.003-0.005/image | $0.06/image |
| **Maskless** | ✅ Yes | ❌ No | ✅ Yes |
| **Indian/casual photos** | OK | Struggles | Best |
| **Recommended for production** | ✅ Yes (with caveats) | ❌ No (license) | ✅ Yes |

**Verdict**:

- **Building a commercial product?** → Use **Vertex VTO** at $0.06/image. Quality is best, no license risk, no infra to manage. The cost savings of self-hosting only matter at very high sustained volume.
- **Want to self-host commercially?** → Use **FASHN VTON v1.5** (Apache 2.0). Quality is close enough to Vertex for most use cases, infra cost is real but manageable.
- **Just learning / personal projects?** → **OOTDiffusion** is fine for this. Don't put it on a SaaS.

---

## Honest Limitations

After extensive testing of both stacks:

1. **Open-source quality < paid APIs**: Both FASHN and OOTD consistently produce lower quality than Vertex VTO for the same inputs. Period.
2. **Not all photos work**: Both models trained largely on Western models, studio lighting, frontal poses. They struggle with:
   - Casual / candid photos
   - Non-frontal poses
   - Heavy shadows or backlight
   - Non-Western body types and clothing styles
3. **Garment images matter more than parameters**: A bad garment input (on-model, busy background) breaks the output regardless of param tuning. Use flat-lay or ghost mannequin images.
4. **Cost isn't always cheaper**: Sporadic testing costs ~$0.04-0.06/call (real, including idle). Vertex VTO is $0.06/call. Savings only kick in at high sustained traffic.
5. **OOTD license blocks commercial use**: Cannot be used in any product where money changes hands.
6. **Cold start matters**: First request after idle can take 60-90s. Plan UX accordingly (loading states, async jobs).

**For commercial production, use Vertex AI VTO or FASHN's paid API. Self-hosted FASHN is a viable option for cost reduction at scale, but plan for the engineering overhead.**

---

## License Summary

| Component | License | Commercial use? |
|---|---|---|
| This repo's deployment code | MIT | ✅ Yes |
| FASHN VTON v1.5 model | Apache 2.0 | ✅ Yes |
| OOTDiffusion model | CC BY-NC-SA 4.0 | ❌ No |
| Modal platform | Modal's ToS | ✅ Yes (per their pricing) |

By using this repository, you agree to all underlying licenses, especially OOTD's non-commercial restriction.

---

## Credits

- **FASHN AI**: Dan Bochman, Aya Bochman — for releasing VTON v1.5 under Apache 2.0 ([repo](https://github.com/fashn-AI/fashn-vton-1.5))
- **OOTDiffusion**: Yuhao Xu, Tao Gu, Weifeng Chen, Chengcai Chen, Xiao-i Research ([paper](https://arxiv.org/abs/2403.01779))
- **Modal**: Best-in-class serverless GPU platform
- **Hugging Face**: For hosting model weights freely
- **OpenAI**: For CLIP ViT-Large used internally by OOTD
- **DWPose / OpenPose**: For pose detection used in both pipelines

---

## Status

| Project | Build | Inference | Production-ready |
|---|---|---|---|
| FASHN | ✅ Working | ✅ Working | ⚠️ Yes (commercial-safe), with quality caveat |
| OOTD | ✅ Working | ⚠️ Mixed quality | ❌ No (license blocks commercial use) |
