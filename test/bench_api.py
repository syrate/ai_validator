# save as bench_api.py and run:  python bench_api.py --url https://... --iters 30 --warmup 2
import argparse, json, math, statistics, time, warnings, sys
import requests
from urllib3.exceptions import InsecureRequestWarning
from pathlib import Path
import os 
import base64
warnings.simplefilter("ignore", InsecureRequestWarning)

def percentile(values, p):
    data = sorted(values)
    if not data: return float("nan")
    k = (len(data)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return data[int(k)]
    return data[f] + (k - f) * (data[c] - data[f])

def bench(url, payload, iters=20, warmup=2, sleep=0.0, timeout=60.0, verify=False):
    sess = requests.Session()
    headers = {"Content-Type": "application/json"}
    # Warmup (not measured)
    for _ in range(max(0, warmup)):
        try:
            sess.post(url, headers=headers, data=json.dumps(payload), verify=verify, timeout=timeout)
        except Exception:
            pass

    times_ms = []
    codes = []
    failures = 0

    for i in range(iters):
        t0 = time.perf_counter()
        try:
            r = sess.post(url, headers=headers, data=json.dumps(payload), verify=verify, timeout=timeout)
            dt = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt)
            codes.append(r.status_code)
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt)  # include failed latency to see impact
            failures += 1
        if sleep > 0:
            time.sleep(sleep)

    ok_times = [t for t, code in zip(times_ms, codes + [None]*(len(times_ms)-len(codes))) if code and 200 <= code < 400]

    def stats(arr):
        if not arr: return {}
        return {
            "count": len(arr),
            "avg_ms": statistics.mean(arr)/1000.0,
            "stdev_s": statistics.pstdev(arr)/1000.0 if len(arr) > 1 else 0.0,
            "median_s": statistics.median(arr)/1000.0,
            "p90_s": percentile(arr, 90)/1000.0,
            "p95_s": percentile(arr, 95)/1000.0,
            "p99_s": percentile(arr, 99)/1000.0,
            "min_s": min(arr)/1000.0,
            "max_s": max(arr)/1000.0,
            "sum_s": sum(arr)/1000.0,
        }

    all_stats = stats(times_ms)
    ok_stats = stats(ok_times)

    # Status code distribution
    from collections import Counter
    code_counts = Counter(codes)

    return {
        "iters": iters,
        "warmup": warmup,
        "failures": failures,
        "status_counts": dict(sorted(code_counts.items())),
        "all_requests": all_stats,
        "successful_requests": ok_stats,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://ai-validator-654942414948.europe-west9.run.app/validate" , help="API endpoint")
    ap.add_argument("--payload", help="JSON string or @path/to/file.json", default="{}")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--verify", action="store_true", help="Verify TLS cert (omit to match verify=False)")
    args = ap.parse_args()

    # Load payload
    # if args.payload.startswith("@"):
    #     with open(args.payload[1:], "r", encoding="utf-8") as f:
    #         payload = json.load(f)
    # else:
    #     try:
    #         payload = json.loads(args.payload)
    #     except json.JSONDecodeError:
    #         print("Invalid JSON payload; pass a JSON string or @file.json", file=sys.stderr)
    #         sys.exit(1)
    
    project_root = Path(__file__).resolve().parent.parent
    print("Project root:", project_root)
    IMAGE_PATH = os.path.join(project_root, "data","lpr2", "N24M10S727_2460004_5_220850_c0.00_r0.00_p0.00_y0.00_b0.00_w0_0.00.jpg")

    print("Image path:", IMAGE_PATH)
    with open(IMAGE_PATH, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_base64": image_base64,
        "task": "lpr",
        "validator": "gemini"
    }
    t = time.perf_counter()
    res = bench(args.url, payload, args.iters, args.warmup, args.sleep, args.timeout, verify=args.verify)
    dt = (time.perf_counter() - t) 
    print(f"Total time: {dt:.2f}s")
    import pprint; pprint.pprint(res)
