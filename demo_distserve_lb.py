import os
import time
import random
import ray
import heapq
from collections import defaultdict

############################################################
# 1. 启动 Ray（不用 ray start，避开 sentinel / acl 问题）
############################################################

RAY_TMPDIR = os.path.expanduser("~/ray_tmp")
os.makedirs(RAY_TMPDIR, exist_ok=True)

ray.init(
    num_cpus=os.cpu_count(),
    num_gpus=0,
    ignore_reinit_error=True,
    _temp_dir=RAY_TMPDIR,
)

print("Ray initialized")
print("Ray nodes:", ray.nodes())

############################################################
# 2. Mock Worker（模拟 distserve worker）
############################################################

@ray.remote
class MockWorker:
    def __init__(self, wid: int, speed: float):
        """
        speed: 越小越快（模拟算力差异）
        """
        self.wid = wid
        self.speed = speed
        self.processed = 0

    def handle_request(self, req_id: int):
        # 模拟真实执行时间
        base = random.uniform(0.05, 0.15)
        latency = base * self.speed
        time.sleep(latency)

        self.processed += 1
        return {
            "worker_id": self.wid,
            "latency": latency,
        }

    def stats(self):
        return self.processed


############################################################
# 3. DistServe-style Scheduler（修正版，核心）
############################################################

class DistServeScheduler:
    """
    正确的 distserve-style time-aware scheduler

    - 维护每个 worker 的 Estimated Completion Time (ECT)
    - 每次选择 ECT 最小的 worker
    - 新 ECT = 旧 ECT + 预计执行时间
    """

    def __init__(self, num_workers: int):
        # min-heap: (ECT, worker_id)
        self.heap = [(0.0, wid) for wid in range(num_workers)]
        heapq.heapify(self.heap)

    def select_worker(self):
        ect, wid = heapq.heappop(self.heap)
        return wid, ect

    def on_dispatch(self, wid, ect, estimated_cost):
        new_ect = ect + estimated_cost
        heapq.heappush(self.heap, (new_ect, wid))


############################################################
# 4. 启动 workers（不同性能）
############################################################

# 模拟异构 worker（越小越快）
worker_speeds = [0.7, 1.0, 1.4, 2.0]
NUM_WORKERS = len(worker_speeds)

workers = [
    MockWorker.remote(i, worker_speeds[i])
    for i in range(NUM_WORKERS)
]

scheduler = DistServeScheduler(NUM_WORKERS)

print(f"Started {NUM_WORKERS} workers")
for i, s in enumerate(worker_speeds):
    print(f"  Worker {i}: speed={s}")

############################################################
# 5. 发请求（真正的负载均衡测试）
############################################################

NUM_REQUESTS = 200
futures = []

print(f"\nDispatching {NUM_REQUESTS} requests...")

t0 = time.time()

for req_id in range(NUM_REQUESTS):
    # 1) scheduler 选择 worker
    wid, ect = scheduler.select_worker()

    # 2) distserve-style cost model（简化版）
    estimated_cost = random.uniform(0.05, 0.15) * worker_speeds[wid]

    # 3) 派发请求
    futures.append(
        workers[wid].handle_request.remote(req_id)
    )

    # 4) 更新 scheduler 中的 ECT
    scheduler.on_dispatch(wid, ect, estimated_cost)

results = ray.get(futures)

t1 = time.time()

############################################################
# 6. 统计与分析
############################################################

dist = defaultdict(int)
latencies = []

for r in results:
    dist[r["worker_id"]] += 1
    latencies.append(r["latency"])

latencies.sort()

print("\n=== Request Distribution (should favor fast workers) ===")
for wid in sorted(dist):
    print(
        f"Worker {wid} (speed={worker_speeds[wid]}): "
        f"{dist[wid]} requests"
    )

print("\n=== Latency Statistics ===")
print(f"Total wall time : {t1 - t0:.2f}s")
print(f"P50 latency     : {latencies[len(latencies)//2]:.4f}s")
print(f"P95 latency     : {latencies[int(len(latencies)*0.95)]:.4f}s")

print("\n=== Worker Internal Counters ===")
counts = ray.get([w.stats.remote() for w in workers])
for i, c in enumerate(counts):
    print(f"Worker {i}: {c} requests")

print("\nDistServe-style load balancing demo finished successfully.")
