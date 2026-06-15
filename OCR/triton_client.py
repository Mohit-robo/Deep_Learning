import os
import re
import sys
import time
import cv2
import numpy as np
import tritonclient.http as httpclient
from dataclasses import dataclass, field
from typing import Optional

MODEL_NAME  = "paddle_ocr"
SERVER_URL  = "localhost:8000"
NET_TIMEOUT = 180.0  # seconds – covers PaddleOCR cold-start inside Triton

# ANSI colour codes (degrade gracefully on non-TTY terminals)
_TTY = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

GREEN  = lambda t: _c("92", t)
RED    = lambda t: _c("91", t)
YELLOW = lambda t: _c("93", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)

# Indian number plate format: e.g. MH02MA5324, DL3CAB1234, KA01AB1234
# State(2) + District(2 digits) + Series(1-3 alpha) + Number(4 digits)
_PLATE_RE = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')

def _clean_plate(text: str) -> str:
    """Strip spaces, punctuation and any special chars, keep only A-Z 0-9."""
    return re.sub(r'[^A-Z0-9]', '', text.strip().upper())

def _is_valid_plate(text: str) -> bool:
    return bool(_PLATE_RE.match(_clean_plate(text)))


def _assemble_plate(texts: list[str]) -> tuple[str, bool]:
    """
    Merge multi-line / multi-region OCR outputs into a single plate string.

    The OCR model often returns one text-region per line of the plate, e.g.:
      ['MH04', 'DW 9020'] → 'MH04DW9020'
      ['R', 'MH02AQ2299'] → 'MH02AQ2299'  (single-char noise filtered)
      ['MH47A', 'V6753']  → 'MH47AV6753'

    Algorithm (in priority order):
      1. Clean every token (strip non-alphanumeric).
      2. Drop noise tokens: len < 2 after cleaning.
      3. If any single token is a valid full plate → return it.
      4. Concatenate all filtered tokens in reading order → check validity.
      5. Try every ordered subsequence of 2+ tokens → return first valid hit.
      6. Fallback: return the full concatenation (marked not-valid).
    """
    from itertools import combinations

    tokens = [_clean_plate(t) for t in texts]
    # Keep only tokens with ≥2 alphanumeric characters (filter stray chars/digits)
    filtered = [t for t in tokens if len(t) >= 2]

    if not filtered:
        return "", False

    # Step 3 – any single token already a valid plate?
    for t in filtered:
        if _is_valid_plate(t):
            return t, True

    # Step 4 – concatenate all in order
    concat_all = "".join(filtered)
    if _is_valid_plate(concat_all):
        return concat_all, True

    # Step 5 – try ordered subsets (preserving reading order)
    for r in range(2, len(filtered) + 1):
        for idxs in combinations(range(len(filtered)), r):
            candidate = "".join(filtered[i] for i in idxs)
            if _is_valid_plate(candidate):
                return candidate, True

    # Step 6 – fallback: return full concat, flag as not validated
    return concat_all, False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InferResult:
    path: str
    texts: list[str]           = field(default_factory=list)
    latency_ms: Optional[float] = None   # wall-clock from dispatch → collect
    error: Optional[str]        = None
    img_shape: Optional[tuple]  = None   # (H, W, C)

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def text(self) -> str:
        """Assembled plate string (multi-line fragments merged, noise dropped)."""
        plate, _ = _assemble_plate(self.texts)
        return plate

    @property
    def plate_valid(self) -> bool:
        _, valid = _assemble_plate(self.texts)
        return valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(server_url: str) -> Optional[httpclient.InferenceServerClient]:
    """Create a Triton HTTP client and wait until the server is live."""
    try:
        client = httpclient.InferenceServerClient(
            url=server_url,
            network_timeout=NET_TIMEOUT,
            connection_timeout=10.0,
        )
    except Exception as e:
        print(RED(f"Failed to create Triton client: {e}"))
        return None

    print(DIM("Waiting for Triton server to be live..."))
    for _ in range(30):
        try:
            if client.is_server_live():
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print(RED(f"Error: Triton server at {server_url} did not become live in time."))
        return None

    return client


def _wait_for_model(client: httpclient.InferenceServerClient, model_name: str) -> bool:
    """Poll until the model is READY (initialize() may be slow on first load)."""
    print(DIM(f"Waiting for model '{model_name}' to be ready (up to 2 min on first load)..."))
    for i in range(60):
        try:
            if client.is_model_ready(model_name):
                print(GREEN(f"  ✓ Model ready after ~{i * 2}s"))
                return True
        except Exception:
            pass
        time.sleep(2)
    print(RED(f"Error: Model '{model_name}' did not become ready. Check Triton logs."))
    return False


def _build_infer_io(img: np.ndarray):
    inputs = [httpclient.InferInput("IMAGE", img.shape, "UINT8")]
    inputs[0].set_data_from_numpy(img)
    outputs = [httpclient.InferRequestedOutput("RECOGNIZED_TEXT")]
    return inputs, outputs


def _decode_result(response) -> list[str]:
    text_out = response.as_numpy("RECOGNIZED_TEXT")
    return [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in text_out]


# ---------------------------------------------------------------------------
# Analytics report
# ---------------------------------------------------------------------------

def _print_analytics(results: list[InferResult], total_wall_ms: float) -> None:
    W = 62

    def hr(char="─"):
        print(char * W)

    def row(label: str, value: str, indent: int = 2):
        # Strip ANSI codes for length calculation so padding stays aligned
        raw_value = re.sub(r'\033\[[0-9;]+m', '', value)
        pad = W - indent - len(label) - len(raw_value) - 2
        print(" " * indent + label + " " + "·" * max(pad, 1) + " " + value)

    successes    = [r for r in results if r.success]
    failures     = [r for r in results if not r.success]
    valid_plates = [r for r in successes if r.plate_valid]

    print("\n")
    hr("═")
    print(BOLD(f"{'  TRITON OCR — BATCH SUMMARY':<{W}}"))
    hr("═")

    row("Total images submitted", str(len(results)))
    row("Successful inferences",  GREEN(str(len(successes))))
    row("Failed inferences",      RED(str(len(failures))) if failures else "0")
    row("Success rate",           f"{len(successes)/len(results)*100:.1f}%")
    row("Valid Indian plates",
        GREEN(f"{len(valid_plates)} / {len(successes)}")
        if valid_plates else f"0 / {len(successes)}")
    row("Total batch time",       CYAN(f"{total_wall_ms/1000:.3f} s"))
    row("Throughput",
        CYAN(f"{len(successes)/(total_wall_ms/1000):.2f} img/s")
        if total_wall_ms > 0 else "n/a")

    if failures:
        print()
        print(BOLD(RED("  Failed Requests")))
        hr()
        for r in failures:
            print(f"    {os.path.basename(r.path):<42} {RED(r.error or 'unknown error')}")

    hr("═")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_single(image_path: str, server_url: str = SERVER_URL):
    """Send a single image to Triton and print the OCR result with timing."""
    if not os.path.exists(image_path):
        print(RED(f"Error: Image not found at {image_path}"))
        return

    img = cv2.imread(image_path)
    if img is None:
        print(RED(f"Error: Failed to load image {image_path}"))
        return

    print(f"Loaded: {image_path}  shape={img.shape}  dtype={img.dtype}")

    client = _make_client(server_url)
    if client is None:
        return
    if not _wait_for_model(client, MODEL_NAME):
        return

    inputs, outputs = _build_infer_io(img)
    print("Sending single inference request...")
    t0 = time.perf_counter()
    try:
        response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    except Exception as e:
        print(RED(f"Inference failed: {e}"))
        return
    latency_ms = (time.perf_counter() - t0) * 1000

    texts   = _decode_result(response)
    plate, plate_ok = _assemble_plate(texts)
    fragments = [_clean_plate(t) for t in texts if _clean_plate(t)]  # for display

    print("\n" + "═" * 50)
    print(BOLD("  TRITON OCR RESULT"))
    print("═" * 50)
    flag = GREEN(" ✓ valid") if plate_ok else YELLOW(" ? unverified")
    print(f"  Plate   : {BOLD(plate)}{flag}")
    if len(fragments) > 1 or (fragments and fragments[0] != plate):
        print(f"  Fragments: {DIM(' | '.join(fragments))}")
    print("─" * 50)
    print(f"  Latency : {CYAN(f'{latency_ms:.1f} ms')}")
    print("═" * 50)


def infer_batch(image_paths: list[str], server_url: str = SERVER_URL):
    """
    Send multiple images to Triton concurrently via async_infer.

    Timing strategy:
      - t_dispatch recorded immediately before async_infer()
      - t_collect  recorded immediately after get_result()
      - per-image latency = t_collect - t_dispatch
      - total wall time   = last collect - first dispatch
    """
    # ── Load images ──────────────────────────────────────────────────────────
    images, valid_paths = [], []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(DIM(f"  [skip] Cannot load: {path}"))
            continue
        images.append(img)
        valid_paths.append(path)

    if not images:
        print(RED("Error: No valid images to process."))
        return

    print(f"Loaded {BOLD(str(len(images)))} image(s).")

    client = _make_client(server_url)
    if client is None:
        return
    if not _wait_for_model(client, MODEL_NAME):
        return

    # ── Dispatch all requests ────────────────────────────────────────────────
    print(f"Dispatching {BOLD(str(len(images)))} async inference request(s)...")
    async_handles = []   # async_result | None
    t_wall_start = time.perf_counter()

    for img in images:
        inputs, outputs = _build_infer_io(img)
        try:
            handle = client.async_infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
            async_handles.append(handle)
        except Exception as e:
            async_handles.append(None)
            print(RED(f"  Failed to dispatch: {e}"))

    # ── Collect results ──────────────────────────────────────────────────────
    results: list[InferResult] = []
    print("\n" + "─" * 62)
    print(BOLD(f"  {'#':>3}  {'File':<38}  {'Text':<14}  {'ms':>6}"))
    print("─" * 62)

    for idx, (path, img, handle) in enumerate(
            zip(valid_paths, images, async_handles)):
        basename = os.path.basename(path)
        res = InferResult(path=path, img_shape=img.shape)

        if handle is None:
            res.error = "not dispatched"
            results.append(res)
            print(f"  {idx:>3}  {basename:<38}  {RED('ERROR'):<14}  {'—':>6}")
            continue

        try:
            t_get          = time.perf_counter()      # start blocking wait
            response       = handle.get_result()      # blocks until THIS result is ready
            res.latency_ms = (time.perf_counter() - t_get) * 1000  # per-image, not cumulative
            res.texts      = _decode_result(response)
            plate, valid   = _assemble_plate(res.texts)
            plate_flag     = GREEN("✓") if valid else YELLOW("?")
            text_disp      = plate[:14] if plate else "<empty>"
            lat_disp       = f"{res.latency_ms:.1f}"
            print(f"  {idx:>3}  {basename:<38}  {plate_flag} {text_disp:<12}  {CYAN(lat_disp):>6}")
        except Exception as e:
            res.error = str(e)
            print(f"  {idx:>3}  {basename:<38}  {RED('ERROR'):<14}  {'—':>6}")

        results.append(res)

    t_wall_ms = (time.perf_counter() - t_wall_start) * 1000
    print("─" * 62)

    # ── Analytics report ─────────────────────────────────────────────────────
    _print_analytics(results, t_wall_ms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    default_crop_dir = "images/Indian_number_plate/test/crops"
    args = sys.argv[1:]

    # No args → batch over the entire crop directory
    if not args:
        if not os.path.exists(default_crop_dir):
            print(RED(f"Error: Default crop directory not found: {default_crop_dir}"))
            sys.exit(1)
        paths = sorted(
            os.path.join(default_crop_dir, f)
            for f in os.listdir(default_crop_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not paths:
            print(RED("Error: No images found. Specify paths as arguments."))
            sys.exit(1)
        infer_batch(paths)

    # Single file → single inference with timing
    elif len(args) == 1 and os.path.isfile(args[0]):
        infer_single(args[0])

    # Multiple files or a directory → batch inference
    else:
        paths = []
        for arg in args:
            if os.path.isdir(arg):
                paths.extend(sorted(
                    os.path.join(arg, f)
                    for f in os.listdir(arg)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ))
            elif os.path.isfile(arg):
                paths.append(arg)
            else:
                print(DIM(f"  [skip] Not found: {arg}"))
        if paths:
            infer_batch(paths)
        else:
            print(RED("Error: No valid paths provided."))
            sys.exit(1)
