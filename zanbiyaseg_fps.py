import argparse
import statistics
import time
from pathlib import Path

import torch

from vajra import Vajra


ROOT = Path(__file__).resolve().parent

DEFAULT_WEIGHTS =r"F:\material\baseline\VayuAI-main\zanbiya3\zanbiya_data\weights\best-vajra-v1-nano-seg.pt"
DEFAULT_SOURCE = r"C:\Users\Administrator\Desktop\stealtest"
DEFAULT_IMG_SIZE = 1600
DEFAULT_DEVICE = "0"
DEFAULT_BATCH = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Vajra segmentation inference FPS.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to .pt weights.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image folder, image, video, webcam id, or stream URL.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Inference image size.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help='Device, e.g. "0" or "cpu".')
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size for folder/image inference.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup prediction runs excluded from FPS.")
    parser.add_argument("--runs", type=int, default=30, help="Measured prediction runs used for the average.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    return parser.parse_args()


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def is_probably_path(source):
    text = str(source)
    if text.isdigit():
        return False
    return not text.lower().startswith(("http://", "https://", "rtsp://", "rtmp://", "tcp://"))


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


def main():
    args = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    if is_probably_path(args.source) and not Path(args.source).exists():
        raise FileNotFoundError(f"source not found: {args.source}")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    model = Vajra(str(weights))

    # Warm up first so model loading/CUDA initialization is not included.
    for _ in range(max(args.warmup, 0)):
        for _ in run_predict(model, args):
            pass

    run_stats = []
    for run_idx in range(args.runs):
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
            print("images: 0")
            print("No frames/images were read from the source.")
            return

        latency_ms = total_time / count * 1000
        fps = count / total_time
        run_stats.append(
            {
                "count": count,
                "total_time": total_time,
                "latency_ms": latency_ms,
                "fps": fps,
                "preprocess_ms": preprocess_ms / count,
                "inference_ms": inference_ms / count,
                "postprocess_ms": postprocess_ms / count,
            }
        )

        print(
            f"run {run_idx + 1:02d}/{args.runs}: "
            f"images={count}, total_time={total_time:.4f}s, "
            f"latency={latency_ms:.2f}ms/image, FPS={fps:.2f}"
        )

    total_images = sum(item["count"] for item in run_stats)
    total_measured_time = sum(item["total_time"] for item in run_stats)
    total_times = [item["total_time"] for item in run_stats]
    latencies = [item["latency_ms"] for item in run_stats]
    fps_values = [item["fps"] for item in run_stats]

    avg_total_time = statistics.mean(total_times)
    std_total_time = statistics.stdev(total_times) if len(total_times) > 1 else 0.0
    avg_latency_ms = total_measured_time / total_images * 1000
    avg_fps = total_images / total_measured_time

    print(f"weights: {weights}")
    print(f"source: {args.source}")
    print(f"img_size: {args.img_size}")
    print(f"device: {args.device}")
    print(f"batch: {args.batch}")
    print(f"runs: {args.runs}")
    print(f"total_images: {total_images}")
    print(f"avg_images/run: {total_images / args.runs:.2f}")
    print(f"avg_total_time/run: {avg_total_time:.4f} s")
    print(f"std_total_time/run: {std_total_time:.4f} s")
    print(f"avg_latency: {avg_latency_ms:.2f} ms/image")
    print(f"avg_FPS: {avg_fps:.2f}")
    print(f"mean_run_latency: {statistics.mean(latencies):.2f} ms/image")
    print(f"mean_run_FPS: {statistics.mean(fps_values):.2f}")
    print(
        "avg_speed: "
        f"preprocess {statistics.mean(item['preprocess_ms'] for item in run_stats):.2f} ms/image, "
        f"inference {statistics.mean(item['inference_ms'] for item in run_stats):.2f} ms/image, "
        f"postprocess {statistics.mean(item['postprocess_ms'] for item in run_stats):.2f} ms/image"
    )


if __name__ == "__main__":
    main()
