"""
FASHN VTON v1.5 deployed on Modal.
- A10G GPU, synchronous, open endpoint
- Model stays warm for 2 min between requests
- Auto-scales to zero (you only pay for active GPU time)
"""
import io
import time
import base64
from pathlib import Path

import modal

# ---- Pricing constants (A10G on Modal) ----
GPU_TYPE = "A10G"
PRICE_GPU_PER_SECOND = 1.10 / 3600  # $1.10/hr → per second

# ---- Build the container image ----
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "wget")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "pillow",
        "numpy",
        "huggingface_hub",
        "fastapi[standard]",
    )
    .run_commands(
        "git clone https://github.com/fashn-AI/fashn-vton-1.5.git /fashn",
        "cd /fashn && pip install -e .",
        "pip uninstall -y onnxruntime && pip install onnxruntime-gpu",
        "cd /fashn && python scripts/download_weights.py --weights-dir /fashn/weights",
    )
)

app = modal.App("fashn-vton")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    scaledown_window=120,
    timeout=300,
    memory=8192,
)
class VTONService:
    @modal.enter()
    def load_model(self):
        """Runs once when container starts. Keeps model in GPU memory."""
        import sys
        sys.path.insert(0, "/fashn/src")
        from fashn_vton import TryOnPipeline

        print("[boot] Loading FASHN VTON v1.5...")
        t0 = time.time()
        self.pipeline = TryOnPipeline(weights_dir="/fashn/weights")
        print(f"[boot] Model loaded in {time.time() - t0:.1f}s")

    @modal.method()
    def generate(
        self,
        person_b64: str,
        garment_b64: str,
        category: str = "tops",
        num_timesteps: int = 20,
        guidance_scale: float = 1.5,
        seed: int = 42,
    ) -> dict:
        """Run virtual try-on. Returns base64 PNG + cost data."""
        from PIL import Image, ImageOps

        t_start = time.time()

        # Decode + fix EXIF rotation
        person_bytes = base64.b64decode(person_b64)
        garment_bytes = base64.b64decode(garment_b64)
        person = ImageOps.exif_transpose(Image.open(io.BytesIO(person_bytes))).convert("RGB")
        garment = ImageOps.exif_transpose(Image.open(io.BytesIO(garment_bytes))).convert("RGB")

        t_decoded = time.time()

        # Direct call with EXACT FASHN param names (verified from Kaggle code)
        result = self.pipeline(
            person,
            garment,
            category=category,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        t_inference_done = time.time()

        # Encode output
        out_buf = io.BytesIO()
        result.images[0].save(out_buf, format="PNG", optimize=True)
        out_b64 = base64.b64encode(out_buf.getvalue()).decode()

        t_total = time.time() - t_start
        cost_usd = t_total * PRICE_GPU_PER_SECOND

        print(f"[gen] timesteps={num_timesteps} guidance={guidance_scale} "
              f"seed={seed} inference={t_inference_done - t_decoded:.2f}s "
              f"total={t_total:.2f}s cost=${cost_usd:.5f}")

        return {
            "image_b64": out_b64,
            "cost_usd": round(cost_usd, 5),
            "cost_inr": round(cost_usd * 83.5, 4),
            "params": {
                "num_timesteps": num_timesteps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "category": category,
            },
            "timing": {
                "decode_sec": round(t_decoded - t_start, 2),
                "inference_sec": round(t_inference_done - t_decoded, 2),
                "total_sec": round(t_total, 2),
            },
        }

    @modal.fastapi_endpoint(method="POST")
    def web(self, payload: dict) -> dict:
        """Open HTTP endpoint. POST JSON with image data + optional params."""
        person_b64 = payload.get("person_b64")
        garment_b64 = payload.get("garment_b64")
        category = payload.get("category", "tops")
        num_timesteps = int(payload.get("num_timesteps", 20))
        guidance_scale = float(payload.get("guidance_scale", 1.5))
        seed = int(payload.get("seed", 42))

        if not person_b64 or not garment_b64:
            return {"error": "person_b64 and garment_b64 required"}

        if category not in ("tops", "bottoms", "one-pieces"):
            return {"error": f"category must be tops|bottoms|one-pieces, got {category}"}

        return self.generate.local(
            person_b64, garment_b64, category,
            num_timesteps, guidance_scale, seed,
        )


# ---- Local CLI test ----
@app.local_entrypoint()
def main(
    person_path: str,
    garment_path: str,
    category: str = "tops",
    out: str = "result.png",
    timesteps: int = 20,
    guidance: float = 1.5,
    seed: int = 42,
):
    """Run from terminal:
    modal run modal_app.py --person-path test_images/person.jpg --garment-path test_images/garment.jpg
    """
    person_b64 = base64.b64encode(Path(person_path).read_bytes()).decode()
    garment_b64 = base64.b64encode(Path(garment_path).read_bytes()).decode()

    service = VTONService()
    result = service.generate.remote(
        person_b64, garment_b64, category, timesteps, guidance, seed,
    )

    Path(out).write_bytes(base64.b64decode(result["image_b64"]))
    print(f"\n✅ Saved: {out}")
    print(f"💰 Cost: ${result['cost_usd']} (~₹{result['cost_inr']})")
    print(f"⏱️  Total: {result['timing']['total_sec']}s "
          f"(inference: {result['timing']['inference_sec']}s)")
    print(f"⚙️  Params: {result['params']}")