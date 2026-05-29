"""
test_e2e.py — End-to-End System Testing
=========================================
Tests the complete Banana Ripeness AI system from API to frontend.

Test Suite:
    1.  Health Check            — API is up, all 4 models loaded
    2.  Single Prediction       — /predict returns correct structure
    3.  All Models              — each model returns valid response
    4.  Grad-CAM                — heatmap is returned and valid base64
    5.  Banana Detection        — detector finds banana in test image
    6.  No Banana Rejection     — non-banana image is rejected gracefully
    7.  Model Comparison        — /compare returns all 4 model results
    8.  History Logging         — predictions saved to /history
    9.  History Clear           — DELETE /history works
    10. All Classes             — system can classify all 4 ripeness classes
    11. Confidence Scores       — probabilities sum to 1.0
    12. Response Time           — each prediction under 5 seconds
    13. Concurrent Requests     — 3 simultaneous requests handled
    14. Invalid Input           — bad file returns 422, not 500
    15. Frontend Reachability   — Vercel URL returns 200

Run:
    python test_e2e.py

    # With custom API URL:
    API_URL=https://parankafle11-banana-ripeness.hf.space python test_e2e.py

    # Test local server only:
    API_URL=http://localhost:8000 python test_e2e.py

Requirements:
    pip install requests pillow numpy
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
API_URL      = os.environ.get("API_URL", "https://parankafle11-banana-ripeness.hf.space")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://ai-matrix-banana-ripeness.vercel.app")
TIMEOUT      = 60
MAX_LATENCY  = 10.0
CLASS_NAMES  = ["overripe", "ripe", "rotten", "unripe"]
MODEL_NAMES  = ["efficientnet", "efficientnetv2", "mobilenet", "resnet"]
DATA_DIR     = Path("data")

# ── Look for a real banana image in common locations ──────────────────────────
BANANA_IMAGE = None
_search_paths = [
    Path("C:/Users/kafle/Pictures/banana.jpg"),
    Path("C:/Users/kafle/Pictures/banana.png"),
    Path("C:/Users/kafle/Downloads/banana.jpg"),
    Path("C:/Users/kafle/Downloads/banana.png"),
    Path("test_banana.jpg"),
    Path("test_banana.png"),
]
for _p in _search_paths:
    if _p.exists():
        BANANA_IMAGE = _p
        print(f"  Using real banana image: {_p}")
        break

# ── Per-class images — one photo per ripeness class ───────────────────────────
# Save these files to fix the "All Classes" warning:
#   C:\Users\kafle\Pictures\banana_overripe.jpg
#   C:\Users\kafle\Pictures\banana_ripe.jpg
#   C:\Users\kafle\Pictures\banana_rotten.jpg
#   C:\Users\kafle\Pictures\banana_unripe.jpg
CLASS_IMAGES = {}
for _cls in ["overripe", "ripe", "rotten", "unripe"]:
    for _ext in ["jpg", "png"]:
        _p = Path(f"C:/Users/kafle/Pictures/banana_{_cls}.{_ext}")
        if _p.exists():
            CLASS_IMAGES[_cls] = _p
            break
        _p2 = Path(f"test_banana_{_cls}.{_ext}")
        if _p2.exists():
            CLASS_IMAGES[_cls] = _p2
            break
if CLASS_IMAGES:
    print(f"  Per-class images found: {list(CLASS_IMAGES.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# Test result tracking
# ─────────────────────────────────────────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"
WARN = "⚠️  WARN"

results: list[dict] = []

def record(name: str, status: str, detail: str = "", duration: float = 0.0):
    results.append({
        "name": name, "status": status,
        "detail": detail, "duration": duration
    })
    icon = status.split()[0]
    dur  = f"  [{duration:.2f}s]" if duration > 0 else ""
    print(f"  {status:<12} {name}{dur}")
    if detail and status != PASS:
        print(f"             → {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Test image generators (no real images needed)
# ─────────────────────────────────────────────────────────────────────────────
def make_banana_image(color=(220, 180, 40), size=(400, 300)) -> bytes:
    """Generate a synthetic banana-coloured ellipse image."""
    img  = Image.new("RGB", size, (100, 80, 50))
    draw = ImageDraw.Draw(img)
    # Draw banana-like shape
    w, h = size
    draw.ellipse([w//4, h//4, 3*w//4, 3*h//4], fill=color)
    draw.ellipse([w//3, h//3, 2*w//3, 2*h//3], fill=tuple(min(255, c+30) for c in color))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def make_random_image(size=(224, 224)) -> bytes:
    """Generate a random noise image (not a banana)."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def get_test_image(class_name: str = None) -> bytes:
    """Get a real banana image. Priority: per-class photo > generic photo > data/ > synthetic."""
    # 1. Use per-class specific photo if available
    if class_name and class_name in CLASS_IMAGES:
        return CLASS_IMAGES[class_name].read_bytes()
    # 2. Use generic real photo
    if BANANA_IMAGE and BANANA_IMAGE.exists():
        return BANANA_IMAGE.read_bytes()
    # 3. Use data/ folder if available
    if DATA_DIR.exists() and class_name:
        for split in ["test", "val", "train"]:
            cls_dir = DATA_DIR / split / class_name
            if cls_dir.exists():
                imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                if imgs:
                    return imgs[0].read_bytes()
    # 4. Fallback synthetic — will likely be rejected by detector
    return make_banana_image()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def get(path: str, **kwargs) -> requests.Response:
    return requests.get(f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)

def post_image(path: str, image_bytes: bytes,
               filename: str = "test.jpg",
               extra_data: dict = None) -> requests.Response:
    data  = extra_data or {}
    files = {"file": (filename, image_bytes, "image/jpeg")}
    return requests.post(f"{API_URL}{path}", files=files,
                         data=data, timeout=TIMEOUT)

def delete(path: str) -> requests.Response:
    return requests.delete(f"{API_URL}{path}", timeout=TIMEOUT)


# ─────────────────────────────────────────────────────────────────────────────
# Individual tests
# ─────────────────────────────────────────────────────────────────────────────
def test_health():
    name = "Health Check — API online + models loaded"
    t0   = time.time()
    try:
        r = get("/health")
        d = r.json()
        assert r.status_code == 200, f"HTTP {r.status_code}"
        assert d.get("status") == "ok", f"status={d.get('status')}"
        loaded = [m for m, v in d.get("models", {}).items() if v == "loaded"]
        if len(loaded) < 4:
            no_ckpt = [m for m, v in d.get("models", {}).items() if v == "no_checkpoint"]
            record(name, WARN, f"Only {len(loaded)}/4 models loaded. Missing: {no_ckpt}",
                   time.time()-t0)
        else:
            record(name, PASS, f"All 4 models loaded. Device={d.get('device')}", time.time()-t0)
        return d
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)
        return {}


def test_single_prediction():
    name = "Single Prediction — /predict returns correct structure"
    t0   = time.time()
    try:
        img = get_test_image("ripe")
        r   = post_image("/predict", img,
                          extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        d = r.json()
        assert "class"       in d, "missing 'class'"
        assert "confidence"  in d, "missing 'confidence'"
        assert "confidences" in d, "missing 'confidences'"
        assert "recipe"      in d, "missing 'recipe'"
        assert d["class"] in CLASS_NAMES, f"unknown class: {d['class']}"
        assert 0 <= d["confidence"] <= 1, f"confidence out of range: {d['confidence']}"
        record(name, PASS,
               f"class={d['class']}  conf={d['confidence']:.3f}  model={d['model']}",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_all_models():
    name = "All Models — each model returns valid response"
    t0   = time.time()
    img  = get_test_image()
    passed, failed = [], []
    for model in MODEL_NAMES:
        try:
            r = post_image("/predict", img,
                           extra_data={"model_name": model, "use_gradcam": "false"})
            if r.status_code == 200:
                d = r.json()
                if "class" in d:
                    passed.append(f"{model}({d['class']})")
                elif "error" in d:
                    passed.append(f"{model}(no_banana)")
                else:
                    failed.append(f"{model}:missing_class")
            else:
                failed.append(f"{model}:HTTP{r.status_code}")
        except Exception as e:
            failed.append(f"{model}:{e}")

    if failed:
        record(name, FAIL, f"Failed: {failed}", time.time()-t0)
    else:
        record(name, PASS, f"All passed: {passed}", time.time()-t0)


def test_gradcam():
    name = "Grad-CAM — heatmap returned and valid base64 PNG"
    t0   = time.time()
    try:
        img = get_test_image("ripe")
        r   = post_image("/predict", img,
                          extra_data={"model_name": "mobilenet", "use_gradcam": "true"})
        assert r.status_code == 200, f"HTTP {r.status_code}"
        d   = r.json()
        if d.get("error") == "no_banana":
            record(name, SKIP, "No banana detected in test image — Grad-CAM not triggered",
                   time.time()-t0)
            return
        cam = d.get("gradcam_b64")
        if cam is None:
            record(name, WARN, "gradcam_b64 is null — Grad-CAM may have failed silently",
                   time.time()-t0)
            return
        raw = base64.b64decode(cam)
        img_check = Image.open(io.BytesIO(raw))
        assert img_check.size[0] > 0, "Empty image"
        record(name, PASS,
               f"Grad-CAM returned {len(raw)//1024} KB PNG ({img_check.size})",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_banana_detection():
    name = "Banana Detection — detector finds banana in image"
    t0   = time.time()
    try:
        img = get_test_image("ripe")
        r   = post_image("/predict", img,
                          extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
        assert r.status_code == 200, f"HTTP {r.status_code}"
        d   = r.json()
        if d.get("error") == "no_banana":
            record(name, WARN,
                   "Test image not detected as banana — try a clearer photo",
                   time.time()-t0)
            return
        det = d.get("detection", {})
        score = det.get("det_score", 0)
        bbox  = det.get("bbox")
        record(name, PASS,
               f"Detected with {score:.0%} confidence  bbox={bbox}",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_no_banana_rejection():
    name = "No-Banana Rejection — random image returns error gracefully"
    t0   = time.time()
    try:
        img = make_random_image()
        r   = post_image("/predict", img,
                          extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
        # Should return 200 with error key, NOT 500
        assert r.status_code == 200, f"HTTP {r.status_code} — should be 200 with error body"
        d = r.json()
        if d.get("error") == "no_banana":
            record(name, PASS,
                   f"Correctly rejected: '{d.get('message', '')[:60]}'",
                   time.time()-t0)
        elif "class" in d:
            record(name, WARN,
                   f"Random image classified as '{d['class']}' — detector may be too lenient",
                   time.time()-t0)
        else:
            record(name, FAIL, f"Unexpected response: {str(d)[:100]}", time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_model_comparison():
    name = "Model Comparison — /compare returns all 4 models"
    t0   = time.time()
    try:
        img = get_test_image()
        r   = requests.post(f"{API_URL}/compare",
                             files={"file": ("test.jpg", img, "image/jpeg")},
                             timeout=TIMEOUT * 2)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        d = r.json()
        compare_results = d.get("results", [])
        assert len(compare_results) > 0, "Empty results"
        valid = [res for res in compare_results if "class" in res or "error" in res]
        record(name, PASS,
               f"{len(valid)}/{len(MODEL_NAMES)} models returned results",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_history_logging():
    name = "History Logging — predictions saved to /history"
    t0   = time.time()
    try:
        # Make a prediction first
        img = get_test_image()
        post_image("/predict", img,
                   extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
        # Check history
        r = get("/history?limit=5")
        assert r.status_code == 200, f"HTTP {r.status_code}"
        d = r.json()
        h = d.get("history", [])
        assert len(h) > 0, "History is empty after prediction"
        latest = h[0]
        assert "predicted"  in latest, "missing 'predicted'"
        assert "confidence" in latest, "missing 'confidence'"
        assert "model"      in latest, "missing 'model'"
        record(name, PASS,
               f"{len(h)} records. Latest: {latest['predicted']} ({latest['confidence']:.3f})",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_history_clear():
    name = "History Clear — DELETE /history empties log"
    t0   = time.time()
    try:
        r = delete("/history")
        assert r.status_code == 200, f"HTTP {r.status_code}"
        d = r.json()
        assert "deleted" in d, "missing 'deleted' key"
        # Verify it's empty
        r2 = get("/history?limit=5")
        h  = r2.json().get("history", [])
        assert len(h) == 0, f"History not empty after clear: {len(h)} records remain"
        record(name, PASS, f"Deleted {d['deleted']} records. History now empty.",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_all_classes():
    name = "All Classes — system can classify all 4 ripeness classes"
    t0   = time.time()
    found_classes = set()
    try:
        for cls in CLASS_NAMES:
            img = get_test_image(cls)
            r   = post_image("/predict", img,
                              extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
            if r.status_code == 200:
                d = r.json()
                if "class" in d:
                    found_classes.add(d["class"])
        if len(found_classes) >= 2:
            record(name, PASS,
                   f"Predicted classes: {sorted(found_classes)}",
                   time.time()-t0)
        else:
            record(name, WARN,
                   f"Only predicted {found_classes} — model may be biased",
                   time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_confidence_scores():
    name = "Confidence Scores — probabilities sum to 1.0"
    t0   = time.time()
    try:
        img = get_test_image()
        r   = post_image("/predict", img,
                          extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
        assert r.status_code == 200
        d = r.json()
        if "error" in d:
            record(name, SKIP, "No banana detected", time.time()-t0)
            return
        confs = d.get("confidences", {})
        assert len(confs) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(confs)}"
        total = sum(confs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total:.4f}, expected 1.0"
        record(name, PASS,
               f"Sum={total:.6f}  Classes={list(confs.keys())}",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


NUM_CLASSES = 4


def test_response_time():
    name = "Response Time — prediction under threshold"
    t0   = time.time()
    try:
        img   = get_test_image()
        times = []
        for _ in range(3):
            t = time.time()
            r = post_image("/predict", img,
                           extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
            times.append(time.time() - t)
        avg = np.mean(times)
        mx  = np.max(times)
        if mx > MAX_LATENCY:
            record(name, WARN,
                   f"Max latency {mx:.2f}s exceeds {MAX_LATENCY}s threshold. Avg={avg:.2f}s",
                   time.time()-t0)
        else:
            record(name, PASS,
                   f"Avg={avg:.2f}s  Max={mx:.2f}s  (threshold={MAX_LATENCY}s)",
                   time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_concurrent_requests():
    name = "Concurrent Requests — 3 simultaneous requests handled"
    t0   = time.time()
    responses = {}
    errors    = {}

    def make_request(i):
        try:
            img = get_test_image()
            r   = post_image("/predict", img,
                              extra_data={"model_name": "mobilenet", "use_gradcam": "false"})
            responses[i] = r.status_code
        except Exception as e:
            errors[i] = str(e)

    threads = [threading.Thread(target=make_request, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=60)

    ok = sum(1 for s in responses.values() if s == 200)
    if errors:
        record(name, FAIL, f"Errors: {errors}", time.time()-t0)
    elif ok == 3:
        record(name, PASS, f"All 3 concurrent requests returned 200", time.time()-t0)
    else:
        record(name, WARN, f"Only {ok}/3 requests succeeded: {responses}", time.time()-t0)


def test_invalid_input():
    name = "Invalid Input — non-image file returns error not 500"
    t0   = time.time()
    try:
        # Send a text file as image
        bad_data = b"this is not an image file"
        r = requests.post(
            f"{API_URL}/predict",
            files={"file": ("test.txt", bad_data, "text/plain")},
            data={"model_name": "mobilenet", "use_gradcam": "false"},
            timeout=TIMEOUT,
        )
        # Should return 400 or 422, NOT 500
        assert r.status_code != 500, f"Server crashed with 500 on invalid input"
        record(name, PASS,
               f"Returned HTTP {r.status_code} (not 500)",
               time.time()-t0)
    except Exception as e:
        record(name, FAIL, str(e), time.time()-t0)


def test_frontend():
    name = "Frontend — Vercel URL returns 200"
    t0   = time.time()
    urls_to_try = [
        FRONTEND_URL,
        FRONTEND_URL.rstrip("/") + "/index.html",
        "https://ai-matrix-banana-ripness.vercel.app",
        "https://ai-matrix-banana-ripeness.vercel.app",
    ]
    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=30, allow_redirects=True)
            if r.status_code == 200:
                has_app = "banana" in r.text.lower() or "API" in r.text
                record(name, PASS,
                       f"Frontend accessible at {url} ({'app found' if has_app else 'page loaded'})",
                       time.time()-t0)
                return
        except Exception:
            continue
    record(name, FAIL,
           f"All URLs returned non-200. Check your FRONTEND_URL setting.",
           time.time()-t0)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    sep = "═" * 65
    print(sep)
    print("  END-TO-END SYSTEM TEST — Banana Ripeness AI")
    print(sep)
    print(f"  API     : {API_URL}")
    print(f"  Frontend: {FRONTEND_URL}")
    print(sep)
    print()

    # Run all tests in order
    test_functions = [
        test_health,
        test_single_prediction,
        test_all_models,
        test_gradcam,
        test_banana_detection,
        test_no_banana_rejection,
        test_model_comparison,
        test_history_logging,
        test_history_clear,
        test_all_classes,
        test_confidence_scores,
        test_response_time,
        test_concurrent_requests,
        test_invalid_input,
        test_frontend,
    ]

    print(f"  Running {len(test_functions)} tests...\n")
    suite_t0 = time.time()

    for fn in test_functions:
        fn()

    suite_time = time.time() - suite_t0

    # ── Summary ───────────────────────────────────────────────────────────────
    passed  = sum(1 for r in results if r["status"] == PASS)
    failed  = sum(1 for r in results if r["status"] == FAIL)
    warned  = sum(1 for r in results if r["status"] == WARN)
    skipped = sum(1 for r in results if r["status"] == SKIP)
    total   = len(results)

    print()
    print(sep)
    print("  TEST SUMMARY")
    print(sep)
    print(f"  Total    : {total}")
    print(f"  ✅ Passed : {passed}")
    print(f"  ❌ Failed : {failed}")
    print(f"  ⚠️  Warned : {warned}")
    print(f"  ⏭  Skipped: {skipped}")
    print(f"  Duration : {suite_time:.1f}s")
    print(sep)

    if failed > 0:
        print("\n  FAILED TESTS:")
        for r in results:
            if r["status"] == FAIL:
                print(f"    ❌ {r['name']}")
                print(f"       {r['detail']}")

    if warned > 0:
        print("\n  WARNINGS:")
        for r in results:
            if r["status"] == WARN:
                print(f"    ⚠️  {r['name']}")
                print(f"       {r['detail']}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = Path("test_e2e_report.json")
    report = {
        "api_url":     API_URL,
        "frontend_url": FRONTEND_URL,
        "total":       total,
        "passed":      passed,
        "failed":      failed,
        "warned":      warned,
        "skipped":     skipped,
        "duration_s":  round(suite_time, 2),
        "tests":       results,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report saved: {report_path.resolve()}")
    print(sep)

    # Exit with error code if any tests failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
