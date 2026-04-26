"""Test client — calls your deployed Modal endpoint."""
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

# Your Modal URL (replace with the one Modal printed after deploy)
MODAL_URL = "https://aaditya3301--fashn-vton-vtonservice-web.modal.run"

# Defaults — these match the Kaggle-quality run
DEFAULT_TIMESTEPS = 20
DEFAULT_GUIDANCE = 1.5
DEFAULT_SEED = 42


def load_and_fix_orientation(path: str) -> bytes:
    """Open image, apply EXIF rotation, return JPEG bytes."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def tryon(
    person_path: str,
    garment_path: str,
    category: str = "tops",
    out: str = "result.png",
    timesteps: int = DEFAULT_TIMESTEPS,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int = DEFAULT_SEED,
):
    person_bytes = load_and_fix_orientation(person_path)
    garment_bytes = load_and_fix_orientation(garment_path)

    person_b64 = base64.b64encode(person_bytes).decode()
    garment_b64 = base64.b64encode(garment_bytes).decode()

    print(f"📤 Sending request to {MODAL_URL}")
    print(f"⚙️  Params: timesteps={timesteps} guidance={guidance} seed={seed} category={category}")

    t0 = time.time()
    response = requests.post(
        MODAL_URL,
        json={
            "person_b64": person_b64,
            "garment_b64": garment_b64,
            "category": category,
            "num_timesteps": timesteps,
            "guidance_scale": guidance,
            "seed": seed,
        },
        timeout=300,
    )
    elapsed = time.time() - t0
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        print(f"❌ Server error: {data['error']}")
        return

    Path(out).write_bytes(base64.b64decode(data["image_b64"]))

    print(f"\n✅ Saved: {out}")
    print(f"💰 GPU cost: ${data['cost_usd']} (~₹{data['cost_inr']})")
    print(f"⏱️  Server time: {data['timing']['total_sec']}s "
          f"(inference: {data['timing']['inference_sec']}s)")
    print(f"⏱️  Round trip:  {elapsed:.2f}s (includes upload/download)")

    # Append to cost log
    log_entry = {
        "timestamp": time.time(),
        "out": out,
        "cost_usd": data["cost_usd"],
        "cost_inr": data["cost_inr"],
        "inference_sec": data["timing"]["inference_sec"],
        "total_sec": data["timing"]["total_sec"],
        "params": data["params"],
    }
    with open("cost_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print("📝 Logged to cost_log.jsonl")


def show_total_cost():
    """Print accumulated cost across all calls."""
    log_file = Path("cost_log.jsonl")
    if not log_file.exists():
        print("No cost log yet.")
        return
    rows = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    total_usd = sum(r["cost_usd"] for r in rows)
    avg_time = sum(r["total_sec"] for r in rows) / len(rows)
    print(f"\n📊 Cost Summary ({len(rows)} calls):")
    print(f"   Total: ${total_usd:.4f} (~₹{total_usd * 83.5:.2f})")
    print(f"   Avg per call: ${total_usd / len(rows):.5f}")
    print(f"   Avg time: {avg_time:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--cost":
        show_total_cost()
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python client.py <person.jpg> <garment.jpg> [category] [out.png] [timesteps] [guidance] [seed]")
        print("  python client.py --cost     (show total cost summary)")
        sys.exit(1)

    args = sys.argv[1:]
    person = args[0]
    garment = args[1]
    category = args[2] if len(args) > 2 else "tops"
    out = args[3] if len(args) > 3 else "result.png"
    timesteps = int(args[4]) if len(args) > 4 else DEFAULT_TIMESTEPS
    guidance = float(args[5]) if len(args) > 5 else DEFAULT_GUIDANCE
    seed = int(args[6]) if len(args) > 6 else DEFAULT_SEED

    tryon(person, garment, category, out, timesteps, guidance, seed)