import os
import time
import random
import ray
from collections import defaultdict

########################################
# 1. 启动 Ray（不用 ray start）
########################################

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

########################################
# 2. 定义一个“假 Worker”（模拟模型耗时）
########################################

@ray.remote
class MockWorker:
    def __init__(self, wid: int):
        self.wid = wid
        self.count = 0

    def handle_request(self, request_id: int):
        self.count += 1
        # 模拟不同请求耗时
        t = random.uniform(0.05, 0.2)
        time.sleep(t)
        return {
            "worker_id": self.wid,
            "request_id": request_id,
            "latency": t,
        }

    def get_count(self):
        return self.count


########################################
# 3. 启动多个 worker（模拟 distserve worker pool）
########################################

NUM_WORKERS = 4
workers = [MockWorker.remote(i) for i in range(NUM_WORKERS)]

print(f"Started {NUM_WORKERS} workers")

########################################
# 4. 一个“简单调度器”（轮询）
#    👉 distserve 的 scheduler 就在这里换成更聪明的策略
########################################

def dispatch_requests(num_requests: int):
    futures = []
    for i in range(num_requests):
        worker = workers[i % NUM_WORKERS]  # 轮询
        futures.append(worker.handle_request.remote(i))
    return futures


########################################
# 5. 发请求（这一步就是“负载”）
########################################

NUM_REQUESTS = 200

print(f"Dispatching {NUM_REQUESTS} requests...")
t0 = time.time()

results = ray.get(dispatch_requests(NUM_REQUESTS))

t1 = time.time()
print(f"All requests finished in {t1 - t0:.2f}s")

########################################
# 6. 统计负载是否均衡
########################################

stats = defaultdict(int)
for r in results:
    stats[r["worker_id"]] += 1

print("\nRequest distribution per worker:")
for wid in sorted(stats):
    print(f"  Worker {wid}: {stats[wid]} requests")

########################################
# 7. 和 worker 自己的计数做一次交叉验证
########################################

print("\nWorker internal counters:")
counts = ray.get([w.get_count.remote() for w in workers])
for i, c in enumerate(counts):
    print(f"  Worker {i}: {c} requests")

print("\nDemo finished successfully.")
