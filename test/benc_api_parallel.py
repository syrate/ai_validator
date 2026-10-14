# save as bench_api_parallel.py
# run:  python bench_api_parallel.py --url https://... --iters 100 --concurrency 20 --warmup 5
import argparse, base64, json, math, statistics, sys, time, warnings, os
from pathlib import Path
from collections import Counter
import concurrent.futures as cf
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", InsecureRequestWarning)

# ---------- helpers ----------
def percentile(values, p):
    if not values: return float("nan")
    data = sorted(values)
    k = (len(data)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return data[int(k)]
    return data[f] + (k - f) * (data[c] - data[f])

def stats_ms(arr):
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

# Thread-local sessions (one per worker)
_thread_local = threading.local()
def get_session(verify: bool):
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        # One connection per thread; avoids cross-thread contention
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.verify = verify
        _thread_local.session = s
    return s

def do_post(url, payload, headers, verify, timeout):
    t0 = time.perf_counter()
    code = None
    err = None
    try:
        s = get_session(verify)
        r = s.post(url, headers=headers, data=json.dumps(payload), verify=verify, timeout=timeout)
        code = r.status_code
    except Exception as e:
        err = str(e)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {"dt_ms": dt_ms, "code": code, "error": err}

def bench_parallel(url, payload, iters=50, concurrency=10, warmup=2, timeout=60.0, verify=False, sleep=0.0):
    headers = {"Content-Type": "application/json"}

    # Warmup (sequential, not measured)
    if warmup > 0:
        s = requests.Session()
        for _ in range(warmup):
            try:
                s.post(url, headers=headers, data=json.dumps(payload), verify=verify, timeout=timeout)
            except Exception:
                pass

    # Parallel run
    iters = int(iters)
    concurrency = max(1, min(int(concurrency), iters))
    results = []
    start_wall = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(do_post, url, payload, headers, verify, timeout) for _ in range(iters)]
        for f in cf.as_completed(futs):
            results.append(f.result())
            if sleep > 0:
                time.sleep(sleep)
    wall_s = time.perf_counter() - start_wall

    # Aggregate
    times_ms = [r["dt_ms"] for r in results]
    codes = [r["code"] for r in results if r["code"] is not None]
    ok_times = [r["dt_ms"] for r in results if r["code"] and 200 <= r["code"] < 400]
    failures = sum(1 for r in results if (r["code"] is None) or not (200 <= (r["code"] or 0) < 400))
    code_counts = dict(sorted(Counter(codes).items()))

    all_stats = stats_ms(times_ms)
    ok_stats = stats_ms(ok_times)

    # Effective average in-flight concurrency ≈ sum(latencies)/wall_time
    eff_conc = (sum(times_ms) / 1000.0) / wall_s if wall_s > 0 else float("nan")
    throughput_rps = iters / wall_s if wall_s > 0 else float("nan")

    return {
        "iters": iters,
        "concurrency": concurrency,
        "warmup": warmup,
        "failures": failures,
        "status_counts": code_counts,
        "batch_wall_time_s": wall_s,
        "throughput_rps": throughput_rps,
        "effective_concurrency": eff_conc,
        "all_requests": all_stats,
        "successful_requests": ok_stats,
    }

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://ai-validator-654942414948.europe-west9.run.app/validate", help="API endpoint")
    ap.add_argument("--iters", type=int, default=50, help="Total number of requests to send")
    ap.add_argument("--concurrency", type=int, default=10, help="Number of parallel workers")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between completed futures (not typical)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--verify", action="store_true", help="Verify TLS cert (omit to match verify=False)")
    ap.add_argument("--image", default=None, help="Path to image to base64-encode (overrides default)")
    args = ap.parse_args()

    # Build payload (same pattern you used)
    project_root = Path(__file__).resolve().parent.parent
    img_path = args.image or os.path.join(project_root, "data", "lpr2", "N24M10S727_2460004_5_220850_c0.00_r0.00_p0.00_y0.00_b0.00_w0_0.00.jpg")
    with open(img_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_base64": image_base64,
        "task": "lpr",
        "validator": "gemini",
    }

    res = bench_parallel(
        args.url, payload,
        iters=args.iters,
        concurrency=args.concurrency,
        warmup=args.warmup,
        timeout=args.timeout,
        verify=args.verify,
        sleep=args.sleep,
    )

    import pprint; pprint.pprint(res)
