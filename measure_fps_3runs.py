import argparse
import statistics
import time
from pathlib import Path

import torch

from vajra import Vajra


DEFAULT_WEIGHTS = (
    r"F:\material\baseline\VayuAI-main\SAMX compare\VajraV1MerudandaXDCN V2 Conv"
    r"\weights\best-vajra-v1-nano-seg.pt"
)
DEFAULT_SOURCE = r"F:\material\FPS_TEST\STEAL"
DEFAULT_IMG_SIZE = 1600
DEFAULT_DEVICE = "0"
DEFAULT_BATCH = 1
DEFAULT_WARMUP = 1
DEFAULT_RUNS = 3

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Vajra end-to-end inference FPS for 3 runs.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to model weights.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image folder, image, video, stream URL, or webcam id.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Inference image size.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help='Device, e.g. "0" or "cpu".')
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size for folder/image inference.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup runs excluded from FPS.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Measured runs. Default is 3.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    return parser.parse_args()


def is_probably_path(source):
    text = str(source)
    if text.isdigit():
        return False
    return not text.lower().startswith(("http://", "https://", "rtsp://", "rtmp://", "tcp://"))


def count_source_images(source):
    if not is_probably_path(source):
        return None

    path = Path(source)
    if path.is_file():
        return 1 if path.suffix.lower() in IMAGE_SUFFIXES else None
    if path.is_dir():
        return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    return None


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_predict(model, args):
    return model.predict(
        args.source,
        img_size=args.img_size,
        device=args.device,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        stream=True,
        verbose=False,
        save=False,
    )


def measure_one_run(model, args):
    sync_cuda()
    start = time.perf_counter()

    count = 0
    preprocess_ms = 0.0
    inference_ms = 0.0
    postprocess_ms = 0.0

    for result in run_predict(model, args):
        count += 1
        speed = getattr(result, "speed", None) or {}
        preprocess_ms += speed.get("preprocess", 0.0)
        inference_ms += speed.get("inference", 0.0)
        postprocess_ms += speed.get("postprocess", 0.0)

    sync_cuda()
    total_time = time.perf_counter() - start

    if count == 0:
        raise RuntimeError("No images or frames were read from the source.")

    return {
        "count": count,
        "total_time": total_time,
        "fps": count / total_time,
        "latency_ms": total_time / count * 1000,
        "preprocess_ms": preprocess_ms / count,
        "inference_ms": inference_ms / count,
        "postprocess_ms": postprocess_ms / count,
    }


def print_environment(args, weights, source_image_count):
    print("=== FPS Test Config ===")
    print(f"weights: {weights}")
    print(f"source: {args.source}")
    if source_image_count is not None:
        print(f"source_images: {source_image_count}")
    print(f"img_size: {args.img_size}")
    print(f"device: {args.device}")
    print(f"batch: {args.batch}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")
    print(f"half: {args.half}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print()


def print_summary(stats):
    total_images = sum(item["count"] for item in stats)
    total_time = sum(item["total_time"] for item in stats)
    fps_values = [item["fps"] for item in stats]
    latency_values = [item["latency_ms"] for item in stats]
    total_time_values = [item["total_time"] for item in stats]

    print()
    print("=== FPS Summary ===")
    print(f"total_images: {total_images}")
    print(f"total_measured_time: {total_time:.4f} s")
    print(f"avg_total_time/run: {statistics.mean(total_time_values):.4f} s")
    print(f"std_total_time/run: {statistics.stdev(total_time_values) if len(total_time_values) > 1 else 0.0:.4f} s")
    print(f"overall_FPS: {total_images / total_time:.2f}")
    print(f"mean_run_FPS: {statistics.mean(fps_values):.2f}")
    print(f"std_run_FPS: {statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0:.2f}")
    print(f"avg_latency: {total_time / total_images * 1000:.2f} ms/image")
    print(f"mean_run_latency: {statistics.mean(latency_values):.2f} ms/image")
    print(
        "avg_speed: "
        f"preprocess {statistics.mean(item['preprocess_ms'] for item in stats):.2f} ms/image, "
        f"inference {statistics.mean(item['inference_ms'] for item in stats):.2f} ms/image, "
        f"postprocess {statistics.mean(item['postprocess_ms'] for item in stats):.2f} ms/image"
    )


def main():
    args = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    if is_probably_path(args.source) and not Path(args.source).exists():
        raise FileNotFoundError(f"source not found: {args.source}")
    if args.runs != 3:
        print(f"WARNING: --runs is {args.runs}; default is 3.")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    torch.backends.cudnn.benchmark = True
    source_image_count = count_source_images(args.source)
    print_environment(args, weights, source_image_count)

    model = Vajra(str(weights))

    for warmup_idx in range(args.warmup):
        for _ in run_predict(model, args):
            pass
        print(f"warmup {warmup_idx + 1}/{args.warmup}: done")

    stats = []
    for run_idx in range(args.runs):
        result = measure_one_run(model, args)
        stats.append(result)
        print(
            f"run {run_idx + 1}/{args.runs}: "
            f"images={result['count']}, "
            f"time={result['total_time']:.4f}s, "
            f"latency={result['latency_ms']:.2f} ms/image, "
            f"FPS={result['fps']:.2f}"
        )

    print_summary(stats)


if __name__ == "__main__":
    main()
