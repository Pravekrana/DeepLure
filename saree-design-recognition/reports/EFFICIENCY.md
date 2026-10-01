# Measured efficiency
Selected 256-D model: **1,074,720 parameters**, 1024 bytes per float32 embedding. Checkpoint is 13,139,219 bytes and includes training optimizer state; it is the exact artifact used by inference. Frozen baselines output 576 float32 values (2,304 bytes/image) with 927,008 backbone parameters.

Device: cpu, 13th Gen Intel(R) Core(TM) i5-1334U; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39. Input [1, 3, 224, 224], batch 1, float32, 2 Torch threads. 10 warm-up calls and 50 timed repeats; final benchmark run after training/evaluation stopped. CUDA/MPS code synchronizes when selected; this CPU measurement needs no accelerator synchronization.

| Operation | Median ms | p95 ms |
|---|---:|---:|
| Encoder only | 23.216 | 41.047 |
| Preprocessing, decoded 640×480 PIL image | 6.486 | 9.122 |
| Exact CPU search + sorting, 1,000 generated vectors | 0.080 | 0.130 |

The actual test gallery contains 45 images; the 1,000-vector search benchmark is explicitly a generated scaling probe. Preprocessing excludes disk read/decode, and timings exclude checkpoint loading and app overhead. No local-patch/multiview cost is hidden. Latency varies with CPU scheduling and WSL; these are local measurements, not device-independent promises. FLOPs/MACs were not measured and are unavailable.
