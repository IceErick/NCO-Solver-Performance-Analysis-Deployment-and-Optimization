import json, time, torch
d = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "available": torch.cuda.is_available()}
if torch.cuda.is_available():
    d["gpu"] = torch.cuda.get_device_name(0)
    d["capability"] = torch.cuda.get_device_capability(0)
    d["arch_list"] = torch.cuda.get_arch_list()        # ← 是否含 'sm_75'
    a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
    for _ in range(3): a @ b                            # 预热
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): a @ b
    torch.cuda.synchronize()
    d["matmul_ms"] = round((time.perf_counter() - t) / 20 * 1000, 2)
print(json.dumps(d, indent=2))
