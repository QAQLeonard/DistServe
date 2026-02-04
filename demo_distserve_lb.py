import os
import time
import random
import ray
import heapq
from collections import defaultdict

###########################################################
# 1. 启动 Ray（Python API，避开 ray start）
###########################################################

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

###########################################################
# 2. Mock Worker（模拟 distserve worker）
###########################################################

@ray.remote
class MockWorker:
    def __init__(self, wid: int, speed: float):
        """
        speed: 越小越快，模拟不同 worker 性能
        """
        self.wid = wid
        self.speed = speed
        self.processed = 0

    def handle_request(self, req_id: int):
        base = random.uniform(0.05, 0.15)
        t = base * self.speed
        time.sleep(t)
        self.processed += 1
        return {
            "worker_id": self.wid,
            "latency": t,
        }

    def stats(self):
        return self.processed


###########################################################
# 3. DistServe-style Scheduler（核心）
###########################################################

class DistServeScheduler:
    """
    这是 distserve 调度思想的最小实现：
    - 每个 worker 维护一个“预计完成时间”
    - 每次 dispatch 选 ECT (estimated completion time) 最小的 worker
    """

    def __init__(self, num_workers):
        self.clock = 0.0
        # min-heap: (estimated_finish_time, worker_id)
        self.heap = [(0.0, wid) for wid in range(num_workers)]
        heapq.heapify(self.heap)

    def select_worker(self):
        ect, wid = heapq.heappop(self.heap)
        return wid, ect

    def on_dispatch(self, wid, estimated_cost):
        new_ect = self.clock + estimated_cost
        heapq.heappush(self.heap, (new_ect, wid))

    def advance_time(self, delta):
        self.clock += delta


###########################################################
# 4. 启动 workers（性能不一样）
###########################################################

# 模拟不同算力
worker_speeds = [0.7, 1.0, 1.4, 2.0]  # 越小越快
NUM_WORKERS = len(worker_speeds)

workers = [
    MockWorker.remote(i, worker_speeds[i])
    for i in range(NUM_WORKERS)
]

scheduler = DistServeScheduler(NUM_WORKERS)

print(f"Started {NUM_WORKERS} workers")

###########################################################
# 5. 发请求（真正的负载均衡测试）
###########################################################

NUM_REQUESTS = 200
futures = []

print(f"Dispatching {NUM_REQUESTS} requests...")

t0 = time.time()

for req_id in range(NUM_REQUESTS):
    wid, ect = scheduler.select_worker()

    # 估计执行时间（distserve 论文里的 cost model）
    estimated_cost = random.uniform(0.05, 0.15) * worker_speeds[wid]

    futures.append(
        workers[wid].handle_request.remote(req_id)
    )

    scheduler.on_dispatch(wid, estimated_cost)

results = ray.get(futures)

t1 = time.time()

###########################################################
# 6. 统计结果
###########################################################

dist = defaultdict(int)
latencies = []

for r in results:
    dist[r["worker_id"]] += 1
    latencies.append(r["latency"])

print("\n=== Request Distribution ===")
for wid in sorted(dist):
    print(f"Worker {wid}: {dist[wid]} requests (speed={worker_speeds[wid]})")

print("\n=== Latency Stats ===")
latencies.sort()
print(f"Total time: {t1 - t0:.2f}s")
print(f"P50 latency: {latencies[len(latencies)//2]:.4f}s")
print(f"P95 latency: {latencies[int(len(latencies)*0.95)]:.4f}s")

print("\n=== Worker Internal Counters ===")
counts = ray.get([w.stats.remote() for w in workers])
for i, c in enumerate(counts):
    print(f"Worker {i}: {c}")

print("\nDistServe-style load balancing demo finished.")
