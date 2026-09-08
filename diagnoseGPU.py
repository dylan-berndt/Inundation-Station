"""Find out what this GPU can actually run, without spending a training run on it.

Builds the real model from a real config and pushes a real forward+backward
through it, so the numbers are the ones training will see. Reports the largest
batch size that fits, and whether cuDNN or the allocator configuration is what
is failing.

    python diagnoseGPU.py                       # FloodHubConfig.json
    python diagnoseGPU.py GCLSTMConfig.json     # any config in configs/

Run it with nothing else on the GPU. It takes about a minute.
"""

import argparse
import os
import sys
import traceback

# Same default the training script uses. Override in the shell to test another.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import torch

from utils.config import Config
from utils.models.modules import CMALLoss
from torch_geometric.data import Data, Batch


class FloodData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("basinContinuous", "basinDiscrete", "riverContinuous", "riverDiscrete",
                   "dischargeFuture", "dischargeHistory", "thresholds", "era5"):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def describeDevice():
    if not torch.cuda.is_available():
        print("CUDA is not available; nothing to diagnose.")
        return False
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info()
    print(f"device          {properties.name}")
    print(f"compute         sm_{properties.major}{properties.minor}"
          f"   (bf16 in hardware: {'yes' if properties.major >= 8 else 'no, emulated only'})")
    print(f"memory          {total / 1e9:.2f} GB total, {free / 1e9:.2f} GB free "
          f"({(total - free) / 1e9:.2f} GB already in use by other processes)")
    print(f"torch           {torch.__version__}")
    print(f"cuda / cudnn    {torch.version.cuda} / {torch.backends.cudnn.version()}")
    print(f"alloc conf      {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(default)')}")
    return True


def buildBatch(config, batchSize, timesteps, staticWidth, riverWidth, riverDiscreteWidth, device):
    items = [FloodData(era5=torch.randn(timesteps, len(config.scales)),
                       basinContinuous=torch.randn(staticWidth),
                       riverContinuous=torch.randn(riverWidth),
                       riverDiscrete=torch.randint(0, 5, (riverDiscreteWidth,)),
                       num_nodes=1, nodes=torch.tensor([1]))
             for _ in range(batchSize)]
    return Batch.from_data_list(items).to(device)


def trialStep(model, config, batchSize, widths, device, objective):
    """One real forward + backward at this batch size. Returns peak MB, or raises."""
    onCuda = device.startswith("cuda")
    if onCuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    past = buildBatch(config, batchSize, config.history, *widths, device)
    future = buildBatch(config, batchSize, config.future, *widths, device)
    targetFuture = torch.randn(batchSize, config.future, device=device)
    targetHistory = torch.randn(batchSize, config.history, device=device)

    model.zero_grad(set_to_none=True)
    hindcast, forecast = model((past, future))
    loss = torch.mean(objective(forecast, targetFuture))
    lastStep = tuple(parameter[:, -1:, :] for parameter in hindcast)
    loss = loss + torch.mean(objective(lastStep, targetHistory[:, -1:]))
    loss.backward()

    peak = 0.0
    if onCuda:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e6
    del past, future, targetFuture, targetHistory, hindcast, forecast, loss, lastStep
    model.zero_grad(set_to_none=True)
    if onCuda:
        torch.cuda.empty_cache()
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="FloodHubConfig.json")
    parser.add_argument("--device", default="cuda",
                        help="cpu runs the same shapes without memory numbers, to check a config loads")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda"):
        if not describeDevice():
            return
    else:
        print(f"device          {device} (no memory measurements; shape check only)")
    print()

    config = Config().load(os.path.join("configs", args.config))

    # Static-feature widths, inferred the way the dataset builds them.
    staticWidth = sum(1 for key in config.variables.basin if config.variables.basin[key])
    riverWidth = sum(1 for key in config.variables.river if config.variables.river[key])
    riverDiscreteWidth = sum(1 for key in config.variables.river if not config.variables.river[key])
    widths = (staticWidth, riverWidth, riverDiscreteWidth)

    from utils.models.hub import FloodHub
    model = FloodHub(config).to(device)
    objective = CMALLoss()

    print(f"config          {args.config}   history {config.history}, future {config.future}, "
          f"batchSize {config.batchSize}")
    # The projections are LazyLinear, so parameters do not exist until a forward
    # has run and their input width is known.
    trialStep(model, config, 2, widths, device, objective)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model           FloodHub, {parameters / 1e6:.2f} M parameters "
          f"({parameters * 4 / 1e6:.0f} MB fp32, roughly x4 with gradients and Adam state)")
    print()

    for cudnn in ((True, False) if device.startswith("cuda") else (True,)):
        torch.backends.cudnn.enabled = cudnn
        label = "cuDNN on " if cudnn else "cuDNN off"
        print(f"--- {label} " + "-" * 52)
        largest = 0
        for batchSize in (8, 16, 32, 64, 128, 256, 512):
            try:
                peak = trialStep(model, config, batchSize, widths, device, objective)
                largest = batchSize
                marker = "  <- configured" if batchSize == config.batchSize else ""
                print(f"   batch {batchSize:>4}  ok    peak {peak:>8.0f} MB{marker}")
            except RuntimeError as error:
                message = str(error).split("\n")[0]
                print(f"   batch {batchSize:>4}  FAIL  {message[:88]}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                print(f"   batch {batchSize:>4}  FAIL  out of memory")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                break
        print(f"   largest batch that fits: {largest}")
        if largest and largest < config.batchSize:
            print(f"   -> configured batchSize {config.batchSize} does NOT fit; "
                  f"set it to {largest} or lower")
        print()

    torch.backends.cudnn.enabled = True
    print("--- sustained run at the configured batch size " + "-" * 22)
    try:
        peaks = [trialStep(model, config, config.batchSize, widths, device, objective) for _ in range(30)]
        print(f"   30 steps ok | peak {min(peaks):.0f}-{max(peaks):.0f} MB, "
              f"drift {peaks[-1] - peaks[0]:+.0f} MB")
        if device.startswith("cuda"):
            print(f"   reserved after 30 steps: {torch.cuda.memory_reserved() / 1e6:.0f} MB")
        if peaks[-1] - peaks[0] > 50:
            print("   -> peak is climbing across steps, which points at a retained reference, not fragmentation")
    except Exception:
        traceback.print_exc(limit=3)
        print("   -> failed during the sustained run, not on the first step: "
              "memory pressure or fragmentation rather than a bad configuration")


if __name__ == "__main__":
    sys.exit(main())
