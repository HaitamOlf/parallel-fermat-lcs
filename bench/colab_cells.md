# Colab CUDA benchmark cells

Run these in a Colab notebook with **Runtime → Change runtime type → T4 GPU**.
Upload `rsa_common.h`, `lcs_common.h`, `X_2000.txt`, `Y_2000.txt`, `X_5000.txt`,
`Y_5000.txt`, `X_10000.txt`, `Y_10000.txt` into `/content` first (side panel →
Files → Upload).

The two `.cu` sources go in as `%%writefile` cells (paste the file contents
from `Project RSA/fermat_cuda.cu` and `Project LCS/LCS_CUDA.cu`).

Emits two CSVs — `fermat_cuda_results.csv` and `lcs_cuda_results.csv` —
downloadable from the Files panel afterwards for merging with the Lenovo CSVs.

---

## Cell 1: environment capture

```python
!nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv > /content/env_colab.txt
!echo "" >> /content/env_colab.txt
!nvcc --version >> /content/env_colab.txt
!cat /proc/cpuinfo | grep 'model name' | head -1 >> /content/env_colab.txt
!cat /content/env_colab.txt
```

## Cell 2: write fermat_cuda.cu

```python
%%writefile fermat_cuda.cu
# ... paste the entire contents of Project RSA/fermat_cuda.cu here ...
```

## Cell 3: write LCS_CUDA.cu

```python
%%writefile LCS_CUDA.cu
# ... paste the entire contents of Project LCS/LCS_CUDA.cu here ...
```

## Cell 4: Fermat CUDA benchmark

Three modulus tiers × three launch configs × 5 repeats.

```python
import subprocess, csv, re

tiers = [("SMALL", "-DTIER_SMALL"), ("MEDIUM", ""), ("LARGE", "-DTIER_LARGE")]
configs = [("1xN", 1, 256), ("Nx1", 256, 1), ("NxN", 256, 256)]
REPEATS = 5

rows = []
for tier, flag in tiers:
    print(f"=== tier: {tier}")
    build = f"nvcc -O2 -arch=sm_75 {flag} fermat_cuda.cu -o fermat_cuda"
    r = subprocess.run(build, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr); raise SystemExit(1)

    for name, blocks, threads in configs:
        for run in range(1, REPEATS + 1):
            out = subprocess.run(f"./fermat_cuda {blocks} {threads}",
                                 shell=True, capture_output=True, text=True).stdout
            m = re.search(r"Kernel time:\s+([\d.]+)\s+ms", out)
            ms = m.group(1) if m else ""
            print(f"  {name} ({blocks}x{threads}) run {run}: {ms} ms")
            rows.append({"tier": tier, "config": name, "blocks": blocks,
                         "threads": threads, "run": run, "time_ms": ms})

with open("/content/fermat_cuda_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote /content/fermat_cuda_results.csv")
```

## Cell 5: LCS CUDA benchmark

Three input sizes × three launch configs × 5 repeats. Kernel time is the
comparable metric; H2D and D2H recorded separately.

```python
import subprocess, csv, re

sizes = [2000, 5000, 10000]
configs = [("1xN", 1, 256), ("Nx1", 256, 1), ("NxN", 256, 256)]
REPEATS = 5

r = subprocess.run("nvcc -O2 -arch=sm_75 LCS_CUDA.cu -o lcs_cuda",
                   shell=True, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr); raise SystemExit(1)

rows = []
for size in sizes:
    X = f"@X_{size}.txt"; Y = f"@Y_{size}.txt"
    print(f"=== size: {size}")
    for name, blocks, threads in configs:
        for run in range(1, REPEATS + 1):
            out = subprocess.run(f"./lcs_cuda {X} {Y} {blocks} {threads}",
                                 shell=True, capture_output=True, text=True).stdout
            def grab(label):
                m = re.search(rf"{label}\s*:\s+([\d.]+)\s+ms", out)
                return m.group(1) if m else ""
            h2d  = grab("H2D transfer time")
            kern = grab(r"Kernel \(DP fill\) time")
            d2h  = grab("D2H transfer time")
            total = grab("Total GPU time")
            print(f"  {name} run {run}: kern={kern} h2d={h2d} d2h={d2h}")
            rows.append({"size": size, "config": name, "blocks": blocks,
                         "threads": threads, "run": run,
                         "h2d_ms": h2d, "kernel_ms": kern,
                         "d2h_ms": d2h, "total_ms": total})

with open("/content/lcs_cuda_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote /content/lcs_cuda_results.csv")
```

## Cell 6: download the CSVs and env file

```python
from google.colab import files
files.download("/content/fermat_cuda_results.csv")
files.download("/content/lcs_cuda_results.csv")
files.download("/content/env_colab.txt")
```
