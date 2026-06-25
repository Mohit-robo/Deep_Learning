## Root Cause
Jitter + artifacts occur when pipeline backpressure causes packet drops at the RTSP/encoded layer (before decoding), breaking H.264/H.265 GOP structure. P-frames reference previous frames—drop a packet, the decoder can't predict correctly, you get visual corruption on frames that do get decoded.

---

## Quick Diagnostics First
1. **Check bottleneck**: `nvidia-smi dmon` while running → GPU/decoder/encoder load? Watch if GPU hits 95%+ (you're throttled) or stays <70% (upstream buffer exhaustion).
2. **Enable perf logs**: Add `GST_DEBUG=3` to your launch command to see buffer underruns and queue fills.

---

## Experiments (in Priority Order)

### **1. Streammux Batching Mismatch** (Most Common Fix)
Set batch size of streammux and primary detector to equal the number of input sources.

**Your config:**
- `num-source-bins=2` but `batch-size=4` → mismatch
- `[primary-gie]` has no explicit batch-size (defaults may differ)

**Experiment:**
```ini
[streammux]
batch-size=2        # Match num-source-bins
[primary-gie]
batch-size=2        # Match explicitly
```

Also set `live-source=1` (you may already have this; ensure it's present):
```ini
[streammux]
live-source=1
```

---

### **2. Increase RTSP Jitter Buffer Tolerance**
For RTSP high jitter, increase latency property of rtspsrc in [source*] group. This allows rtpjitterbuffer to wait longer for late packets before dropping them.

**Current:** `latency=200` (200 ms)

**Experiment:** Raise in increments—test 500, then 1000:
```ini
[source-attr-all]
latency=500         # or 1000 for very jittery networks
```

**Trade-off:** Higher latency = more cumulative delay, but fewer corrupted frames.

---

### **3. Increase Decoder Buffer Surfaces**
If pipeline elements starved for buffers (low CPU/GPU with jitter), increase num-extra-surfaces to allocate more decoder output buffers.

**Current:** `num-extra-surfaces=24`

**Experiment:**
```ini
[source-attr-all]
num-extra-surfaces=48      # Double it; helps during transient stalls
```

---

### **4. Streammux Timeout Tuning**
If frames are queuing behind slower inference, batched-push-timeout controls how long muxer waits for a full batch before flushing partial batches.

**Current:** `batched-push-timeout=40000` (40 ms)

**Experiment:**
```ini
[streammux]
batched-push-timeout=20000  # Flush faster; avoid late arrivals blocking
```

---

### **5. Disable QoS (Quality of Service) Throttling**
If running in docker/console with low FPS, set qos=0 in sink0. QoS can cause early frame drops under initial load.

**Check your [sink0]:** If you have `type=2` (EglSink) or display output, add:
```ini
[sink0]
qos=0
```

---

### **6. Inference Interval (Skip Frames at GIE, Not RTSP Layer)**
Rather than dropping at the RTSP packet layer (which corrupts), drop **after decoding** at the GIE:

**Current:** `[primary-gie] interval=0` (infer every frame)

**Experiment:**
```ini
[primary-gie]
interval=2          # Infer every 2nd frame; decoder still gets all frames
                    # Reduces inference load, fewer backpressure packets
```

This is cleaner than drop-frame-interval on decoder.

---

### **7. Hardware Memory Type Alignment**
Ensure consistent memory allocation across pipeline:

**Your config:** `cudadec-memtype=0` + `nvbuf-memory-type=0`

**For RTX5060 (dGPU):** Consider explicit device memory:
```ini
[source-attr-all]
cudadec-memtype=0           # Keep as-is (default device memory)
[streammux]
nvbuf-memory-type=0         # Ensure consistency
```

---

### **8. Drop-Frame-Interval (If Scaling to Many Streams)**
Use drop-frame-interval on decoder to skip frame decoding itself—reduces decoder load at cost of fewer frames available.

**Only if GPU is at 95%+:**
```ini
[source-attr-all]
drop-frame-interval=1       # Decode every frame (0 = no drops)
                            # Set to 2+ only if decoder saturated
```

**Caution:** This is a last resort; it's less elegant than interval-based inference skipping.

---

### **9. For Growing Stream Count**
As you add streams, profile first:

```bash
# Monitor GPU/decode/encode load
watch -n 1 'nvidia-smi dmon | head -3'
```

When GPU approaches saturation:
- Reduce `[primary-gie] batch-size` (trade latency for stability)
- Increase `[primary-gie] interval` further (2 → 3 or 4)
- Lower `[streammux] batched-push-timeout` more aggressively

Count the maximum number of streams supported on your GPU by profiling. Keep runtime streams below that max for stable performance.

---