"""Bounded, controlled work in a separate CUDA context on the measured GPU."""
import argparse
import signal
from time import perf_counter, sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compute", "memory"])
    parser.add_argument("duty", type=float, choices=[0.25, 0.5, 0.75])
    args = parser.parse_args()
    import torch

    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Controlled contention requires one full A100")
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    with torch.inference_mode():
        if args.mode == "compute":
            left = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
            right = torch.randn_like(left)
            out = torch.empty_like(left)

            def work():
                torch.mm(left, right, out=out)
        else:
            left = torch.ones(2**25, device="cuda")
            out = torch.empty_like(left)

            def work():
                out.copy_(left)

        work()
        torch.cuda.synchronize()
        print("READY", flush=True)
        period = .04
        while not stopped:
            period_start = perf_counter()
            while not stopped and perf_counter() - period_start < period * args.duty:
                work()
                torch.cuda.synchronize()
            sleep(max(0, period - (perf_counter() - period_start)))


if __name__ == "__main__":
    main()
