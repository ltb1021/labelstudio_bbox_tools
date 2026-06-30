from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import threading
import time
import types
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


def _find_repo_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "labelstudio_bbox_tools").exists():
            return candidate
    raise RuntimeError("repo root를 찾지 못했습니다. pyproject.toml이 있는 labelstudio_bbox_tools 루트에서 실행하세요.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src"
PROJECT_ROOT = REPO_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import tqdm as _tqdm_module  # noqa: F401
except ModuleNotFoundError:
    fallback_tqdm = types.ModuleType("tqdm")

    def _identity_tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    fallback_tqdm.tqdm = _identity_tqdm
    sys.modules["tqdm"] = fallback_tqdm

from labelstudio_bbox_tools.pose_inference.common import PoseInstance, load_pose_class_names
from labelstudio_bbox_tools.pose_inference.draw import draw_pose_instances_on_bgr
from labelstudio_bbox_tools.pose_inference.fallback_crop import (
    FALLBACK_SUCCESS_SOURCE,
    apply_pose_nms,
    attach_detection_context,
    detection_to_pose_instance,
    filter_detections_by_class,
    make_crop_regions,
    offset_pose_from_crop,
    split_pose_detection_cases,
)
from labelstudio_bbox_tools.pose_inference.rfdetr import (
    _filter_pose_instances,
    _keypoints_to_pose_instances,
    _load_rfdetr_keypoint_model,
)
from labelstudio_bbox_tools.pose_inference.yolo11 import _result_to_pose_instances
from labelstudio_bbox_tools.pseudo_label.rfdetr import _detections_to_candidates, _load_rfdetr_model
from labelstudio_bbox_tools.pseudo_label.yolo import classwise_nms_indices
from labelstudio_bbox_tools.video_inference.classes import load_class_names, make_class_color_map
from labelstudio_bbox_tools.video_inference.common import Detection, import_cv2, iter_video_files, safe_name, video_rel_path
from labelstudio_bbox_tools.video_inference.yolo11 import _normalise_imgsz, _result_to_detections


# =============================================================================
# User Config
# =============================================================================
# 아래 값들은 사용자가 실험 목적에 맞게 바꿀 수 있는 설정입니다.
# CLI 인자가 들어오면 CLI 값이 이 기본값보다 우선합니다.

DEFAULT_INPUT_PATH = Path("/mnt/workspace/mhj/projects/ml/resource/video/2026_04_09_falldown_sample")
# 추론할 영상 파일 또는 영상 폴더입니다.
# 동일한 입력을 사용해야 YOLO11과 RF-DETR 성능 비교가 객관적입니다.

DEFAULT_TARGETS = ["yolo11_fallback", "rfdetr_fallback"]
# 비교할 모델 목록입니다.
# 사용 가능 값:
# - "yolo11_pose": YOLO11 pose 단일 모델
# - "rfdetr_pose": RF-DETR Keypoint 단일 모델
# - "yolo11_fallback": YOLO11 detector + YOLO11 pose fallback crop
# - "rfdetr_fallback": RF-DETR detector + RF-DETR Keypoint fallback crop

DEFAULT_WARMUP_RUNS = 1
# 성능 집계에 포함하지 않는 예열 실행 횟수입니다.
# 첫 실행의 model load, CUDA 초기화, cache 영향이 measured run에 섞이지 않게 합니다.

DEFAULT_MEASURED_RUNS = 5
# 실제 평균, p50, p95, FPS 계산에 포함할 반복 실행 횟수입니다.
# 값이 클수록 안정적인 평균을 얻지만 전체 실행 시간이 늘어납니다.

DEFAULT_MAX_FRAMES = 500
# 각 run에서 처리할 최대 frame 수입니다.
# 두 모델 모두 같은 frame 수를 사용해야 비교가 유효합니다.

DEFAULT_FRAME_STRIDE = 1
# 1이면 모든 frame, 5이면 5 frame마다 1장 처리합니다.
# stride가 달라지면 처리 난이도와 평균 FPS가 바뀔 수 있으므로 모델 간 동일하게 유지합니다.

DEFAULT_DEVICE = "cuda:0"
# 사용할 device입니다.
# 예: "cuda:0", "cuda", "cpu", None
# 모델 간 device가 다르면 성능 비교가 무효화됩니다.

DEFAULT_PIPELINE_SCOPE = "end_to_end"
# "model_only": drawing/write 없이 순수 추론 성능을 비교합니다.
# "end_to_end": drawing/write까지 포함한 실제 영상 생성 비용을 비교합니다.
# "both": model_only와 end_to_end를 순차 실행합니다.
# 공식 모델 성능 비교는 "model_only"를 기준으로 합니다.

DEFAULT_PROFILE_MODE = "low_overhead"
# "low_overhead": 공식 비교용입니다. CUDA sync를 끄고 resource sampling 빈도를 낮춥니다.
# "detailed_debug": 병목 분석용입니다. CUDA sync 등 상세 계측이 성능을 바꿀 수 있습니다.

DEFAULT_RESOURCE_INTERVAL_SEC = 1.0
# CPU/RAM/GPU resource sampling 간격입니다.
# 너무 짧으면 profiling 자체가 성능에 영향을 줄 수 있습니다.

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "codex-mhj_26_06_29_fallback_profile"
# profiling 결과가 저장될 폴더입니다.
# 기존 추론 결과와 섞이지 않도록 별도 output directory를 사용합니다.

DEFAULT_OVERWRITE = False
# 기존 profiling run 결과를 덮어쓸지 여부입니다.
# 기본값은 안전하게 False로 두어 이전 실험 기록을 보존합니다.


DEFAULT_YOLO_POSE_WEIGHTS = "yolo11x-pose.pt"
DEFAULT_RFDETR_POSE_WEIGHTS = None
DEFAULT_YOLO_DETECTOR_WEIGHTS = PROJECT_ROOT / "weights" / "2026_06_24_28cls_yolo11m_640_ms015" / "weights" / "best.pt"
DEFAULT_YOLO_DETECTOR_CLASS_YAML = (
    PROJECT_ROOT
    / "weights"
    / "2026_06_24_28cls_yolo11m_640_ms015"
    / "weights"
    / "260623_운영서버_도메인맞춤_추가학습_5차_전체merge_28cls_전체통합.yaml"
)
DEFAULT_RFDETR_DETECTOR_WEIGHTS = PROJECT_ROOT / "weights" / "2026_06_24_28cls_rfdetr_medium_640_ms010" / "checkpoint_best_total.pth"
DEFAULT_RFDETR_DETECTOR_CLASS_YAML = PROJECT_ROOT / "weights" / "2026_06_24_28cls_rfdetr_medium_640_ms010" / "data.yaml"
DEFAULT_TARGET_DETECTION_CLASSES = [
    "small_worker",
    "worker",
    "crouched_worker",
    "small_signalman",
    "small_signalman_no_red",
    "signalman",
    "signalman_no_red",
]


SUPPORTED_TARGETS = {"yolo11_pose", "rfdetr_pose", "yolo11_fallback", "rfdetr_fallback"}


@dataclass(frozen=True)
class DevicePolicy:
    requested: str | None
    normalized: str | None
    enable_gpu_metrics: bool
    cuda_sync: bool
    gpu_index: int | None


@dataclass(frozen=True)
class ProfilingMode:
    name: str
    pipeline_scope: str
    resource_interval_sec: float
    buffer_events: bool
    cuda_sync: bool
    write_debug_images: bool


@dataclass(frozen=True)
class MeasurementScenario:
    scenario_name: str
    targets: list[str]
    input_path: Path
    output_dir: Path
    max_frames: int
    frame_stride: int
    max_videos: int | None
    warmup_runs: int
    measured_runs: int
    profile_mode: str
    pipeline_scope: str
    resource_interval_sec: float
    recursive: bool
    overwrite: bool


@dataclass(frozen=True)
class TargetConfig:
    name: str
    yolo_pose_weights: str | Path
    rfdetr_pose_weights: str | Path | None
    yolo_detector_weights: str | Path | None
    yolo_detector_class_yaml: str | Path | None
    rfdetr_detector_weights: str | Path | None
    rfdetr_detector_class_yaml: str | Path | None
    rfdetr_detector_variant: str
    target_detection_classes: list[str]
    device: str | None
    yolo_pose_imgsz: int | tuple[int, int] | None
    yolo_detector_imgsz: int | tuple[int, int] | None
    rfdetr_pose_shape: tuple[int, int] | None
    pose_conf: float
    pose_iou: float
    detector_conf: float
    detector_iou: float
    keypoint_conf: float
    match_iou: float
    crop_padding_ratio: float
    min_crop_size: int
    fallback_batch_size: int
    max_fallback_crops_per_frame: int | None
    max_pose_per_crop: int
    final_nms_iou: float


@dataclass
class ResourceSnapshot:
    timestamp_sec: float
    target_name: str
    pipeline_scope: str
    run_role: str
    run_index: int
    rss_mb: float | None = None
    max_rss_mb: float | None = None
    cpu_process_sec: float | None = None
    gpu_index: int | None = None
    gpu_mem_used_mb: float | None = None
    gpu_mem_total_mb: float | None = None
    gpu_util_percent: float | None = None
    error: str | None = None


@dataclass
class FrameProfileEvent:
    target_name: str
    pipeline_scope: str
    run_role: str
    run_index: int
    video_path: str
    video_rel_path: str
    frame_index: int
    timestamp_seconds: float
    read_sec: float
    predict_sec: float
    draw_sec: float | None
    write_sec: float | None
    instances: int
    keypoints: int
    error: str | None = None


@dataclass
class LoadProfileEvent:
    target_name: str
    pipeline_scope: str
    run_role: str
    run_index: int
    model_load_sec: float
    error: str | None = None


@dataclass
class FallbackProfileEvent:
    target_name: str
    pipeline_scope: str
    run_role: str
    run_index: int
    video_path: str
    video_rel_path: str
    frame_index: int
    detector_sec: float
    full_pose_sec: float
    case_split_sec: float
    crop_make_sec: float
    crop_pose_sec: float
    final_nms_sec: float
    matched_count: int
    pose_only_count: int
    detection_only_count: int
    crop_count: int
    success_count: int
    failed_count: int


@dataclass
class PredictionResult:
    instances: list[PoseInstance]
    fallback_event: FallbackProfileEvent | None = None


@dataclass
class RunPaths:
    run_dir: Path
    frame_events_path: Path
    fallback_events_path: Path
    load_events_path: Path
    resource_events_path: Path
    summary_json_path: Path
    summary_csv_path: Path
    report_path: Path


@dataclass
class FrameContext:
    target_name: str
    pipeline_scope: str
    run_role: str
    run_index: int
    video_path: Path
    video_rel_path: Path
    frame_index: int
    timestamp_seconds: float
    keypoint_conf: float


@dataclass
class DrawConfig:
    color_map: dict[str, tuple[int, int, int]]
    font_path: str | Path | None = None
    font_size: int = 20
    line_width: int = 3
    keypoint_radius: int = 4
    keypoint_conf: float = 0.2
    score_digits: int = 2
    draw_bbox: bool = True
    draw_skeleton: bool = True
    draw_keypoints: bool = True
    color_by_source: bool = False
    dashed_bbox_sources: list[str] = field(default_factory=list)


class EventWriter:
    def __init__(self, path: Path, *, buffer_size: int = 200) -> None:
        self.path = path
        self.buffer_size = max(1, int(buffer_size))
        self.buffer: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: Any) -> None:
        if hasattr(event, "__dataclass_fields__"):
            payload = asdict(event)
        elif isinstance(event, dict):
            payload = dict(event)
        else:
            raise TypeError(f"Unsupported event type: {type(event)!r}")
        self.buffer.append(payload)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        with self.path.open("a", encoding="utf-8") as file:
            for event in self.buffer:
                file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self.buffer.clear()


class ResourceSampler:
    def __init__(
        self,
        *,
        path: Path,
        target_name: str,
        pipeline_scope: str,
        run_role: str,
        run_index: int,
        device_policy: DevicePolicy,
        interval_sec: float,
    ) -> None:
        if interval_sec < 0.1:
            raise ValueError("resource interval must be >= 0.1 seconds")
        self.path = path
        self.target_name = target_name
        self.pipeline_scope = pipeline_scope
        self.run_role = run_role
        self.run_index = int(run_index)
        self.device_policy = device_policy
        self.interval_sec = float(interval_sec)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.writer = EventWriter(path, buffer_size=50)

    def __enter__(self) -> ResourceSampler:
        self.thread = threading.Thread(target=self._loop, name="pose-profile-resource-sampler", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.interval_sec * 2.0))
        self.writer.write(self._sample())
        self.writer.flush()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            self.writer.write(self._sample())
            self.stop_event.wait(self.interval_sec)

    def _sample(self) -> ResourceSnapshot:
        now = time.time()
        rss_mb = _current_rss_mb()
        max_rss_mb = _max_rss_mb()
        cpu_process_sec = time.process_time()
        gpu_error = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_util = None
        if self.device_policy.enable_gpu_metrics:
            try:
                gpu_mem_used, gpu_mem_total, gpu_util = _sample_gpu(self.device_policy.gpu_index)
            except Exception as exc:
                gpu_error = str(exc)
        return ResourceSnapshot(
            timestamp_sec=now,
            target_name=self.target_name,
            pipeline_scope=self.pipeline_scope,
            run_role=self.run_role,
            run_index=self.run_index,
            rss_mb=rss_mb,
            max_rss_mb=max_rss_mb,
            cpu_process_sec=cpu_process_sec,
            gpu_index=self.device_policy.gpu_index,
            gpu_mem_used_mb=gpu_mem_used,
            gpu_mem_total_mb=gpu_mem_total,
            gpu_util_percent=gpu_util,
            error=gpu_error,
        )


class BaseAdapter:
    model_name: str
    class_names: list[str]

    def predict(self, frame_bgr, context: FrameContext) -> PredictionResult:
        raise NotImplementedError


class Yolo11PoseAdapter(BaseAdapter):
    def __init__(self, config: TargetConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required. Activate the YOLO conda env, for example ltb_ultra.") from exc
        self.config = config
        self.model_name = "yolo11_pose"
        self.class_names = load_pose_class_names(class_yaml=None, manual_classes=None)
        self.model = YOLO(str(config.yolo_pose_weights))
        if config.device:
            self.model.to(config.device)

    def predict(self, frame_bgr, context: FrameContext) -> PredictionResult:
        kwargs: dict[str, Any] = {
            "source": frame_bgr,
            "conf": self.config.pose_conf,
            "iou": self.config.pose_iou,
            "verbose": False,
        }
        if self.config.yolo_pose_imgsz is not None:
            kwargs["imgsz"] = self.config.yolo_pose_imgsz
        if self.config.device:
            kwargs["device"] = self.config.device
        results = self.model.predict(**kwargs)
        instances = _result_to_pose_instances(results[0], self.class_names) if results else []
        return PredictionResult(instances=instances)


class RfDetrPoseAdapter(BaseAdapter):
    def __init__(self, config: TargetConfig) -> None:
        self.config = config
        self.model_name = "rfdetr_pose"
        self.class_names = load_pose_class_names(class_yaml=None, manual_classes=None)
        self.model, self.load_info = _load_rfdetr_keypoint_model(model_weights=config.rfdetr_pose_weights, device=config.device)
        self.cv2 = import_cv2()

    def predict(self, frame_bgr, context: FrameContext) -> PredictionResult:
        frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        kwargs: dict[str, Any] = {"threshold": self.config.pose_conf, "include_source_image": False}
        if self.config.rfdetr_pose_shape is not None:
            kwargs["shape"] = self.config.rfdetr_pose_shape
        raw = self.model.predict(frame_rgb, **kwargs)
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        instances = [] if raw is None else _keypoints_to_pose_instances(raw, self.class_names)
        instances = _filter_pose_instances(
            instances,
            self.class_names,
            enable_nms=True,
            iou=self.config.pose_iou,
            keypoint_conf=self.config.keypoint_conf,
        )
        return PredictionResult(instances=instances)


class Yolo11FallbackAdapter(BaseAdapter):
    def __init__(self, config: TargetConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required. Activate the YOLO conda env, for example ltb_ultra.") from exc
        if config.yolo_detector_weights is None:
            raise ValueError("yolo detector weights are required for yolo11_fallback")
        if config.yolo_detector_class_yaml is None:
            raise ValueError("yolo detector class yaml is required for yolo11_fallback")
        self.config = config
        self.model_name = "yolo11_fallback"
        self.detector_class_names = load_class_names(
            class_yaml=config.yolo_detector_class_yaml,
            manual_classes=None,
            expected_count=None,
            strict_count=False,
        )
        self.class_names = load_pose_class_names(class_yaml=None, manual_classes=None)
        self.detector_model = YOLO(str(config.yolo_detector_weights))
        self.pose_model = YOLO(str(config.yolo_pose_weights))
        if config.device:
            self.detector_model.to(config.device)
            self.pose_model.to(config.device)

    def _detector_predict(self, frame_bgr) -> list[Detection]:
        kwargs: dict[str, Any] = {
            "source": frame_bgr,
            "conf": self.config.detector_conf,
            "iou": self.config.detector_iou,
            "verbose": False,
        }
        if self.config.yolo_detector_imgsz is not None:
            kwargs["imgsz"] = self.config.yolo_detector_imgsz
        if self.config.device:
            kwargs["device"] = self.config.device
        results = self.detector_model.predict(**kwargs)
        detections = _result_to_detections(results[0], self.detector_class_names) if results else []
        return filter_detections_by_class(detections, self.config.target_detection_classes)

    def _pose_predict(self, source) -> list[PoseInstance]:
        kwargs: dict[str, Any] = {
            "source": source,
            "conf": self.config.pose_conf,
            "iou": self.config.pose_iou,
            "verbose": False,
        }
        if isinstance(source, list):
            kwargs["batch"] = len(source)
        if self.config.yolo_pose_imgsz is not None:
            kwargs["imgsz"] = self.config.yolo_pose_imgsz
        if self.config.device:
            kwargs["device"] = self.config.device
        results = self.pose_model.predict(**kwargs)
        if not isinstance(results, list):
            results = [results]
        instances: list[PoseInstance] = []
        for result in results:
            instances.extend(_result_to_pose_instances(result, self.class_names))
        return instances

    def _pose_predict_crops(self, crops) -> tuple[list[PoseInstance], set[int]]:
        if not crops:
            return [], set()
        restored: list[PoseInstance] = []
        successful_crop_indices: set[int] = set()
        for batch_start, crop_batch in _chunked(crops, self.config.fallback_batch_size):
            crop_sources = [crop.crop_bgr for crop in crop_batch]
            kwargs: dict[str, Any] = {
                "source": crop_sources,
                "conf": self.config.pose_conf,
                "iou": self.config.pose_iou,
                "verbose": False,
                "batch": len(crop_batch),
            }
            if self.config.yolo_pose_imgsz is not None:
                kwargs["imgsz"] = self.config.yolo_pose_imgsz
            if self.config.device:
                kwargs["device"] = self.config.device
            results = self.pose_model.predict(**kwargs)
            for offset, (crop, result) in enumerate(zip(crop_batch, results, strict=False)):
                crop_index = batch_start + offset
                crop_poses = _result_to_pose_instances(result, self.class_names)
                crop_poses.sort(key=lambda item: item.raw_score if item.raw_score is not None else item.score, reverse=True)
                selected = crop_poses[: self.config.max_pose_per_crop]
                if selected:
                    successful_crop_indices.add(crop_index)
                for pose in selected:
                    restored.append(offset_pose_from_crop(pose, crop))
        return restored, successful_crop_indices

    def predict(self, frame_bgr, context: FrameContext) -> PredictionResult:
        detector_start = time.perf_counter()
        detections = self._detector_predict(frame_bgr)
        detector_sec = time.perf_counter() - detector_start

        full_pose_start = time.perf_counter()
        full_poses = self._pose_predict(frame_bgr)
        full_pose_sec = time.perf_counter() - full_pose_start

        case_start = time.perf_counter()
        cases = split_pose_detection_cases(pose_instances=full_poses, detections=detections, match_iou=self.config.match_iou)
        case_split_sec = time.perf_counter() - case_start

        crop_start = time.perf_counter()
        crops = make_crop_regions(
            frame_bgr,
            cases.detection_only,
            padding_ratio=self.config.crop_padding_ratio,
            min_crop_size=self.config.min_crop_size,
            max_crops=self.config.max_fallback_crops_per_frame,
        )
        crop_make_sec = time.perf_counter() - crop_start

        crop_pose_start = time.perf_counter()
        fallback_poses, successful_crop_indices = self._pose_predict_crops(crops)
        crop_pose_sec = time.perf_counter() - crop_pose_start

        failed_crops = [(idx, crop) for idx, crop in enumerate(crops) if idx not in successful_crop_indices]
        outputs: list[PoseInstance] = []
        outputs.extend(attach_detection_context(pose, None, source="pose_only") for pose in cases.pose_only)
        outputs.extend(
            attach_detection_context(pose, detection, source="matched_full_frame_pose")
            for detection, pose, _iou in cases.matched
        )
        outputs.extend(fallback_poses)
        outputs.extend(detection_to_pose_instance(crop.detection) for _idx, crop in failed_crops)

        nms_start = time.perf_counter()
        outputs = apply_pose_nms(outputs, iou=self.config.final_nms_iou)
        final_nms_sec = time.perf_counter() - nms_start

        fallback_event = FallbackProfileEvent(
            target_name=context.target_name,
            pipeline_scope=context.pipeline_scope,
            run_role=context.run_role,
            run_index=context.run_index,
            video_path=str(context.video_path),
            video_rel_path=str(context.video_rel_path),
            frame_index=context.frame_index,
            detector_sec=detector_sec,
            full_pose_sec=full_pose_sec,
            case_split_sec=case_split_sec,
            crop_make_sec=crop_make_sec,
            crop_pose_sec=crop_pose_sec,
            final_nms_sec=final_nms_sec,
            matched_count=cases.matched_count,
            pose_only_count=cases.pose_only_count,
            detection_only_count=cases.detection_only_count,
            crop_count=len(crops),
            success_count=len(successful_crop_indices),
            failed_count=len(failed_crops),
        )
        return PredictionResult(instances=outputs, fallback_event=fallback_event)


class RfDetrFallbackAdapter(BaseAdapter):
    def __init__(self, config: TargetConfig) -> None:
        if config.rfdetr_detector_weights is None:
            raise ValueError("RF-DETR detector weights are required for rfdetr_fallback")
        if config.rfdetr_detector_class_yaml is None:
            raise ValueError("RF-DETR detector class yaml is required for rfdetr_fallback")
        self.config = config
        self.model_name = "rfdetr_fallback"
        self.detector_class_names = load_class_names(
            class_yaml=config.rfdetr_detector_class_yaml,
            manual_classes=None,
            expected_count=None,
            strict_count=False,
        )
        self.class_names = load_pose_class_names(class_yaml=None, manual_classes=None)
        self.detector_model, self.detector_load_info = _load_rfdetr_model(
            model_weights=config.rfdetr_detector_weights,
            model_variant=config.rfdetr_detector_variant,
            device=config.device,
        )
        self.pose_model, self.pose_load_info = _load_rfdetr_keypoint_model(
            model_weights=config.rfdetr_pose_weights,
            device=config.device,
        )
        self.cv2 = import_cv2()

    def _detector_predict(self, frame_rgb) -> list[Detection]:
        raw = self.detector_model.predict(frame_rgb, threshold=self.config.detector_conf, include_source_image=False)
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        boxes, class_ids, scores = _detections_to_candidates(raw)
        if not boxes:
            return []
        keep = classwise_nms_indices(
            boxes,
            class_ids,
            scores,
            self.detector_class_names,
            None,
            default_iou=self.config.detector_iou,
        )
        detections = []
        for idx in keep:
            class_id = int(class_ids[idx])
            if not 0 <= class_id < len(self.detector_class_names):
                continue
            score = float(scores[idx])
            if score < self.config.detector_conf:
                continue
            box = boxes[idx]
            detections.append(
                Detection(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    class_id=class_id,
                    class_name=self.detector_class_names[class_id],
                    score=score,
                )
            )
        return filter_detections_by_class(detections, self.config.target_detection_classes)

    def _pose_predict_raw(self, source_rgb) -> list[PoseInstance]:
        kwargs: dict[str, Any] = {"threshold": self.config.pose_conf, "include_source_image": False}
        if self.config.rfdetr_pose_shape is not None:
            kwargs["shape"] = self.config.rfdetr_pose_shape
        raw = self.pose_model.predict(source_rgb, **kwargs)
        raw_items = raw if isinstance(raw, list) else [raw]
        instances: list[PoseInstance] = []
        for item in raw_items:
            if item is None:
                continue
            pose_instances = _keypoints_to_pose_instances(item, self.class_names)
            pose_instances = _filter_pose_instances(
                pose_instances,
                self.class_names,
                enable_nms=True,
                iou=self.config.pose_iou,
                keypoint_conf=self.config.keypoint_conf,
            )
            instances.extend(pose_instances)
        return instances

    def _pose_predict_crops(self, crops) -> tuple[list[PoseInstance], set[int]]:
        if not crops:
            return [], set()
        restored: list[PoseInstance] = []
        successful_crop_indices: set[int] = set()
        for batch_start, crop_batch in _chunked(crops, self.config.fallback_batch_size):
            crop_rgbs = [self.cv2.cvtColor(crop.crop_bgr, self.cv2.COLOR_BGR2RGB) for crop in crop_batch]
            kwargs: dict[str, Any] = {"threshold": self.config.pose_conf, "include_source_image": False}
            if self.config.rfdetr_pose_shape is not None:
                kwargs["shape"] = self.config.rfdetr_pose_shape
            raw_predictions = self.pose_model.predict(crop_rgbs, **kwargs)
            if not isinstance(raw_predictions, list):
                raw_predictions = [raw_predictions]
            for offset, (crop, raw) in enumerate(zip(crop_batch, raw_predictions, strict=False)):
                crop_index = batch_start + offset
                crop_poses = [] if raw is None else _keypoints_to_pose_instances(raw, self.class_names)
                crop_poses = _filter_pose_instances(
                    crop_poses,
                    self.class_names,
                    enable_nms=True,
                    iou=self.config.pose_iou,
                    keypoint_conf=self.config.keypoint_conf,
                )
                crop_poses.sort(key=lambda item: item.raw_score if item.raw_score is not None else item.score, reverse=True)
                selected = crop_poses[: self.config.max_pose_per_crop]
                if selected:
                    successful_crop_indices.add(crop_index)
                for pose in selected:
                    restored.append(offset_pose_from_crop(pose, crop))
        return restored, successful_crop_indices

    def predict(self, frame_bgr, context: FrameContext) -> PredictionResult:
        frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)

        detector_start = time.perf_counter()
        detections = self._detector_predict(frame_rgb)
        detector_sec = time.perf_counter() - detector_start

        full_pose_start = time.perf_counter()
        full_poses = self._pose_predict_raw(frame_rgb)
        full_pose_sec = time.perf_counter() - full_pose_start

        case_start = time.perf_counter()
        cases = split_pose_detection_cases(pose_instances=full_poses, detections=detections, match_iou=self.config.match_iou)
        case_split_sec = time.perf_counter() - case_start

        crop_start = time.perf_counter()
        crops = make_crop_regions(
            frame_bgr,
            cases.detection_only,
            padding_ratio=self.config.crop_padding_ratio,
            min_crop_size=self.config.min_crop_size,
            max_crops=self.config.max_fallback_crops_per_frame,
        )
        crop_make_sec = time.perf_counter() - crop_start

        crop_pose_start = time.perf_counter()
        fallback_poses, successful_crop_indices = self._pose_predict_crops(crops)
        crop_pose_sec = time.perf_counter() - crop_pose_start

        failed_crops = [(idx, crop) for idx, crop in enumerate(crops) if idx not in successful_crop_indices]
        outputs: list[PoseInstance] = []
        outputs.extend(attach_detection_context(pose, None, source="pose_only") for pose in cases.pose_only)
        outputs.extend(
            attach_detection_context(pose, detection, source="matched_full_frame_pose")
            for detection, pose, _iou in cases.matched
        )
        outputs.extend(fallback_poses)
        outputs.extend(detection_to_pose_instance(crop.detection) for _idx, crop in failed_crops)

        nms_start = time.perf_counter()
        outputs = apply_pose_nms(outputs, iou=self.config.final_nms_iou)
        final_nms_sec = time.perf_counter() - nms_start

        fallback_event = FallbackProfileEvent(
            target_name=context.target_name,
            pipeline_scope=context.pipeline_scope,
            run_role=context.run_role,
            run_index=context.run_index,
            video_path=str(context.video_path),
            video_rel_path=str(context.video_rel_path),
            frame_index=context.frame_index,
            detector_sec=detector_sec,
            full_pose_sec=full_pose_sec,
            case_split_sec=case_split_sec,
            crop_make_sec=crop_make_sec,
            crop_pose_sec=crop_pose_sec,
            final_nms_sec=final_nms_sec,
            matched_count=cases.matched_count,
            pose_only_count=cases.pose_only_count,
            detection_only_count=cases.detection_only_count,
            crop_count=len(crops),
            success_count=len(successful_crop_indices),
            failed_count=len(failed_crops),
        )
        return PredictionResult(instances=outputs, fallback_event=fallback_event)


def _chunked(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), int(batch_size)):
        yield start, items[start : start + int(batch_size)]


def _current_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    return None


def _max_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _sample_gpu(gpu_index: int | None) -> tuple[float | None, float | None, float | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=2)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if not rows:
        return None, None, None
    selected = rows[0]
    if gpu_index is not None:
        for row in rows:
            parts = [item.strip() for item in row.split(",")]
            if parts and int(parts[0]) == int(gpu_index):
                selected = row
                break
    parts = [item.strip() for item in selected.split(",")]
    if len(parts) < 4:
        return None, None, None
    return float(parts[1]), float(parts[2]), float(parts[3])


def _count_drawable_keypoints(instances: Sequence[PoseInstance], keypoint_conf: float) -> int:
    return sum(1 for instance in instances for keypoint in instance.keypoints if keypoint.is_drawable(keypoint_conf))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * (float(percentile) / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _safe_divide(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_optional_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def _normalise_shape(value: str | int | None) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return int(value), int(value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("shape must be an int or 'height,width'")


def _normalise_device(device: str | None, *, cuda_sync: bool) -> DevicePolicy:
    if device in (None, ""):
        return DevicePolicy(device, None, False, False, None)
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return DevicePolicy(device, "cpu", False, False, None)
    if normalized == "cuda":
        return DevicePolicy(device, "cuda", True, bool(cuda_sync), 0)
    if normalized.startswith("cuda:"):
        index = int(normalized.split(":", 1)[1])
        return DevicePolicy(device, normalized, True, bool(cuda_sync), index)
    raise ValueError(f"unsupported device: {device}")


def _make_profile_mode(name: str, pipeline_scope: str, resource_interval_sec: float) -> ProfilingMode:
    if pipeline_scope not in {"model_only", "end_to_end"}:
        raise ValueError("pipeline_scope must be 'model_only' or 'end_to_end'")
    if name == "low_overhead":
        return ProfilingMode(
            name=name,
            pipeline_scope=pipeline_scope,
            resource_interval_sec=resource_interval_sec,
            buffer_events=True,
            cuda_sync=False,
            write_debug_images=False,
        )
    if name == "detailed_debug":
        return ProfilingMode(
            name=name,
            pipeline_scope=pipeline_scope,
            resource_interval_sec=resource_interval_sec,
            buffer_events=True,
            cuda_sync=True,
            write_debug_images=False,
        )
    raise ValueError("profile_mode must be 'low_overhead' or 'detailed_debug'")


def _make_run_paths(output_dir: Path, scenario_name: str, overwrite: bool) -> RunPaths:
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{safe_name(scenario_name)}"
    run_dir = output_dir.expanduser().resolve() / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"profiling run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        frame_events_path=run_dir / "profile_events.jsonl",
        fallback_events_path=run_dir / "fallback_events.jsonl",
        load_events_path=run_dir / "load_events.jsonl",
        resource_events_path=run_dir / "resource_events.jsonl",
        summary_json_path=run_dir / "profile_summary.json",
        summary_csv_path=run_dir / "profile_summary.csv",
        report_path=run_dir / "codex-mhj_26_06_29_pose_inference_profile_report.md",
    )


def _build_adapter(target_name: str, config: TargetConfig) -> BaseAdapter:
    if target_name == "yolo11_pose":
        return Yolo11PoseAdapter(config)
    if target_name == "rfdetr_pose":
        return RfDetrPoseAdapter(config)
    if target_name == "yolo11_fallback":
        return Yolo11FallbackAdapter(config)
    if target_name == "rfdetr_fallback":
        return RfDetrFallbackAdapter(config)
    raise ValueError(f"unsupported target: {target_name}")


def _sync_cuda_if_needed(device_policy: DevicePolicy) -> None:
    if not device_policy.cuda_sync:
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for cuda synchronization") from exc
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _validate_scenario(scenario: MeasurementScenario) -> None:
    if scenario.max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if scenario.frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if scenario.warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if scenario.measured_runs < 1:
        raise ValueError("measured_runs must be >= 1")
    if scenario.resource_interval_sec < 0.1:
        raise ValueError("resource_interval_sec must be >= 0.1")
    if not scenario.targets:
        raise ValueError("At least one target is required")
    if not scenario.input_path.expanduser().exists():
        raise FileNotFoundError(f"input_path does not exist: {scenario.input_path}")
    unknown = sorted(set(scenario.targets) - SUPPORTED_TARGETS)
    if unknown:
        raise ValueError(f"unsupported targets: {', '.join(unknown)}")


def _validate_target_assets(targets: Sequence[str], config: TargetConfig) -> None:
    if "yolo11_fallback" in targets:
        _require_path(config.yolo_detector_weights, "yolo_detector_weights")
        _require_path(config.yolo_detector_class_yaml, "yolo_detector_class_yaml")
    if "rfdetr_fallback" in targets:
        _require_path(config.rfdetr_detector_weights, "rfdetr_detector_weights")
        _require_path(config.rfdetr_detector_class_yaml, "rfdetr_detector_class_yaml")
    if config.fallback_batch_size < 1:
        raise ValueError("fallback_batch_size must be >= 1")
    if config.max_pose_per_crop < 1:
        raise ValueError("max_pose_per_crop must be >= 1")


def _require_path(value: str | Path | None, name: str) -> None:
    if value in (None, ""):
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _draw_frame(frame_bgr, instances: Sequence[PoseInstance], draw_config: DrawConfig):
    return draw_pose_instances_on_bgr(
        frame_bgr,
        instances,
        color_map=draw_config.color_map,
        font_path=draw_config.font_path,
        font_size=draw_config.font_size,
        line_width=draw_config.line_width,
        keypoint_radius=draw_config.keypoint_radius,
        keypoint_conf=draw_config.keypoint_conf,
        score_digits=draw_config.score_digits,
        draw_bbox=draw_config.draw_bbox,
        draw_skeleton=draw_config.draw_skeleton,
        draw_keypoints=draw_config.draw_keypoints,
        color_by_source=draw_config.color_by_source,
        dashed_bbox_sources=draw_config.dashed_bbox_sources,
    )


def _run_video_loop(
    *,
    adapter: BaseAdapter,
    scenario: MeasurementScenario,
    config: TargetConfig,
    run_paths: RunPaths,
    device_policy: DevicePolicy,
    profile_mode: ProfilingMode,
    run_role: str,
    run_index: int,
    frame_writer: EventWriter,
    fallback_writer: EventWriter,
) -> None:
    cv2 = import_cv2()
    videos = iter_video_files(scenario.input_path, recursive=scenario.recursive)
    if scenario.max_videos is not None:
        videos = videos[: int(scenario.max_videos)]
    if not videos:
        raise FileNotFoundError(f"No videos found under input_path: {scenario.input_path}")

    draw_config = DrawConfig(
        color_map=make_class_color_map(adapter.class_names),
        keypoint_conf=config.keypoint_conf,
        color_by_source=adapter.model_name.endswith("fallback"),
        dashed_bbox_sources=[FALLBACK_SUCCESS_SOURCE] if adapter.model_name.endswith("fallback") else [],
    )

    for video_path in videos:
        rel_path = video_rel_path(video_path, scenario.input_path.expanduser().resolve())
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        writer = None
        if scenario.pipeline_scope == "end_to_end":
            out_path = (
                run_paths.run_dir
                / "videos"
                / safe_name(adapter.model_name)
                / f"{safe_name(rel_path.stem)}__run{run_index:02d}__{safe_name(run_role)}.mp4"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps if fps > 0 else 30.0,
                (width, height),
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not open video writer: {out_path}")
        processed = 0
        frame_index = 0
        try:
            while True:
                read_start = time.perf_counter()
                ok, frame = cap.read()
                read_sec = time.perf_counter() - read_start
                if not ok:
                    break
                if frame_index % scenario.frame_stride != 0:
                    frame_index += 1
                    continue
                if processed >= scenario.max_frames:
                    break
                timestamp_seconds = frame_index / fps if fps > 0 else 0.0
                context = FrameContext(
                    target_name=adapter.model_name,
                    pipeline_scope=scenario.pipeline_scope,
                    run_role=run_role,
                    run_index=run_index,
                    video_path=video_path,
                    video_rel_path=rel_path,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    keypoint_conf=config.keypoint_conf,
                )
                predict_start = time.perf_counter()
                _sync_cuda_if_needed(device_policy)
                prediction = adapter.predict(frame, context)
                _sync_cuda_if_needed(device_policy)
                predict_sec = time.perf_counter() - predict_start
                draw_sec = None
                write_sec = None
                if scenario.pipeline_scope == "end_to_end":
                    if writer is None:
                        raise RuntimeError("writer must be open in end_to_end mode")
                    draw_start = time.perf_counter()
                    drawn = _draw_frame(frame, prediction.instances, draw_config)
                    draw_sec = time.perf_counter() - draw_start
                    write_start = time.perf_counter()
                    writer.write(drawn)
                    write_sec = time.perf_counter() - write_start
                frame_writer.write(
                    FrameProfileEvent(
                        target_name=adapter.model_name,
                        pipeline_scope=scenario.pipeline_scope,
                        run_role=run_role,
                        run_index=run_index,
                        video_path=str(video_path),
                        video_rel_path=str(rel_path),
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                        read_sec=read_sec,
                        predict_sec=predict_sec,
                        draw_sec=draw_sec,
                        write_sec=write_sec,
                        instances=len(prediction.instances),
                        keypoints=_count_drawable_keypoints(prediction.instances, config.keypoint_conf),
                    )
                )
                if prediction.fallback_event is not None:
                    fallback_writer.write(prediction.fallback_event)
                processed += 1
                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()


def _run_target(
    *,
    target_name: str,
    scenario: MeasurementScenario,
    config: TargetConfig,
    run_paths: RunPaths,
    device_policy: DevicePolicy,
    profile_mode: ProfilingMode,
) -> None:
    frame_writer = EventWriter(run_paths.frame_events_path)
    fallback_writer = EventWriter(run_paths.fallback_events_path)
    load_writer = EventWriter(run_paths.load_events_path)
    load_start = time.perf_counter()
    try:
        adapter = _build_adapter(target_name, config)
        load_writer.write(
            LoadProfileEvent(
                target_name=adapter.model_name,
                pipeline_scope=scenario.pipeline_scope,
                run_role="setup",
                run_index=-1,
                model_load_sec=time.perf_counter() - load_start,
            )
        )
        total_runs = [("warmup", idx) for idx in range(scenario.warmup_runs)] + [
            ("measured", idx) for idx in range(scenario.measured_runs)
        ]
        for run_role, run_index in total_runs:
            print(f"[run] target={target_name} scope={scenario.pipeline_scope} role={run_role} index={run_index}")
            with ResourceSampler(
                path=run_paths.resource_events_path,
                target_name=adapter.model_name,
                pipeline_scope=scenario.pipeline_scope,
                run_role=run_role,
                run_index=run_index,
                device_policy=device_policy,
                interval_sec=profile_mode.resource_interval_sec,
            ):
                _run_video_loop(
                    adapter=adapter,
                    scenario=scenario,
                    config=config,
                    run_paths=run_paths,
                    device_policy=device_policy,
                    profile_mode=profile_mode,
                    run_role=run_role,
                    run_index=run_index,
                    frame_writer=frame_writer,
                    fallback_writer=fallback_writer,
                )
    except Exception as exc:
        if "adapter" not in locals():
            load_writer.write(
                LoadProfileEvent(
                    target_name=target_name,
                    pipeline_scope=scenario.pipeline_scope,
                    run_role="setup",
                    run_index=-1,
                    model_load_sec=time.perf_counter() - load_start,
                    error=str(exc),
                )
            )
        raise
    finally:
        frame_writer.flush()
        fallback_writer.flush()
        load_writer.flush()


def _aggregate(run_paths: RunPaths, scenario: MeasurementScenario) -> dict[str, Any]:
    frame_events = [row for row in _read_jsonl(run_paths.frame_events_path) if row.get("run_role") == "measured"]
    load_events = [row for row in _read_jsonl(run_paths.load_events_path) if not row.get("error")]
    resource_events = [row for row in _read_jsonl(run_paths.resource_events_path) if row.get("run_role") == "measured"]
    fallback_events = [row for row in _read_jsonl(run_paths.fallback_events_path) if row.get("run_role") == "measured"]

    target_names = sorted({row["target_name"] for row in frame_events} | {row["target_name"] for row in load_events})
    rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for target_name in target_names:
        target_frames = [row for row in frame_events if row.get("target_name") == target_name]
        target_loads = [row for row in load_events if row.get("target_name") == target_name]
        target_resources = [row for row in resource_events if row.get("target_name") == target_name]
        predict_values = [float(row["predict_sec"]) for row in target_frames]
        read_predict_values = [float(row["read_sec"]) + float(row["predict_sec"]) for row in target_frames]
        end_to_end_values = [
            float(row["read_sec"]) + float(row["predict_sec"]) + float(row.get("draw_sec") or 0.0) + float(row.get("write_sec") or 0.0)
            for row in target_frames
            if row.get("pipeline_scope") == "end_to_end"
        ]
        model_only_values = [value for row, value in zip(target_frames, read_predict_values, strict=False) if row.get("pipeline_scope") == "model_only"]
        instances = [int(row["instances"]) for row in target_frames]
        keypoints = [int(row["keypoints"]) for row in target_frames]
        empty_count = sum(1 for value in instances if value == 0)
        ram_values = [float(row["rss_mb"]) for row in target_resources if row.get("rss_mb") is not None]
        gpu_mem_values = [float(row["gpu_mem_used_mb"]) for row in target_resources if row.get("gpu_mem_used_mb") is not None]
        gpu_util_values = [float(row["gpu_util_percent"]) for row in target_resources if row.get("gpu_util_percent") is not None]
        rows.append(
            {
                "target_name": target_name,
                "pipeline_scope": scenario.pipeline_scope,
                "measured_frames": len(target_frames),
                "model_load_sec_avg": _mean([float(row["model_load_sec"]) for row in target_loads]),
                "predict_sec_avg": _mean(predict_values),
                "predict_sec_p50": _percentile(predict_values, 50),
                "predict_sec_p95": _percentile(predict_values, 95),
                "model_only_fps": _safe_divide(len(model_only_values), sum(model_only_values)) if model_only_values else None,
                "end_to_end_fps": _safe_divide(len(end_to_end_values), sum(end_to_end_values)) if end_to_end_values else None,
                "instances_per_frame_avg": _mean([float(value) for value in instances]),
                "keypoints_per_frame_avg": _mean([float(value) for value in keypoints]),
                "empty_frame_ratio": _safe_divide(float(empty_count), float(len(target_frames))),
                "ram_rss_mb_max": max(ram_values) if ram_values else None,
                "gpu_mem_mb_max": max(gpu_mem_values) if gpu_mem_values else None,
                "gpu_util_avg": _mean(gpu_util_values),
                "gpu_util_max": max(gpu_util_values) if gpu_util_values else None,
            }
        )

        target_fallback = [row for row in fallback_events if row.get("target_name") == target_name]
        if target_fallback:
            crop_counts = [int(row["crop_count"]) for row in target_fallback]
            success_counts = [int(row["success_count"]) for row in target_fallback]
            failed_counts = [int(row["failed_count"]) for row in target_fallback]
            detector_sec = [float(row["detector_sec"]) for row in target_fallback]
            full_pose_sec = [float(row["full_pose_sec"]) for row in target_fallback]
            crop_pose_sec = [float(row["crop_pose_sec"]) for row in target_fallback]
            fallback_rows.append(
                {
                    "target_name": target_name,
                    "frames": len(target_fallback),
                    "crop_count_avg": _mean([float(value) for value in crop_counts]),
                    "detector_sec_avg": _mean(detector_sec),
                    "full_pose_sec_avg": _mean(full_pose_sec),
                    "crop_pose_sec_avg": _mean(crop_pose_sec),
                    "success_count_total": sum(success_counts),
                    "failed_count_total": sum(failed_counts),
                    "fallback_recovery_rate": _safe_divide(float(sum(success_counts)), float(sum(crop_counts))),
                }
            )

    summary = {
        "scenario": {
            **asdict(scenario),
            "input_path": str(scenario.input_path),
            "output_dir": str(scenario.output_dir),
        },
        "rows": rows,
        "fallback_rows": fallback_rows,
        "event_paths": {
            "frame_events": str(run_paths.frame_events_path),
            "load_events": str(run_paths.load_events_path),
            "resource_events": str(run_paths.resource_events_path),
            "fallback_events": str(run_paths.fallback_events_path),
        },
    }
    _write_json(run_paths.summary_json_path, summary)
    _write_csv(run_paths.summary_csv_path, rows + fallback_rows)
    _write_report(run_paths.report_path, summary)
    return summary


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No data._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(":---" for _ in columns) + " |"
    body = ["| " + " | ".join(_format_value(row.get(col)) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    scenario = summary["scenario"]
    speed_columns = [
        "target_name",
        "pipeline_scope",
        "measured_frames",
        "model_load_sec_avg",
        "predict_sec_avg",
        "predict_sec_p50",
        "predict_sec_p95",
        "model_only_fps",
        "end_to_end_fps",
        "gpu_mem_mb_max",
        "ram_rss_mb_max",
        "instances_per_frame_avg",
        "empty_frame_ratio",
    ]
    fallback_columns = [
        "target_name",
        "frames",
        "crop_count_avg",
        "detector_sec_avg",
        "full_pose_sec_avg",
        "crop_pose_sec_avg",
        "success_count_total",
        "failed_count_total",
        "fallback_recovery_rate",
    ]
    lines = [
        "# Pose Inference Profile Report",
        "",
        "## Scenario",
        "",
        f"* input_path: `{scenario['input_path']}`",
        f"* targets: `{', '.join(scenario['targets'])}`",
        f"* pipeline_scope: `{scenario['pipeline_scope']}`",
        f"* profile_mode: `{scenario['profile_mode']}`",
        f"* max_frames: `{scenario['max_frames']}`",
        f"* frame_stride: `{scenario['frame_stride']}`",
        f"* warmup_runs: `{scenario['warmup_runs']}`",
        f"* measured_runs: `{scenario['measured_runs']}`",
        "",
        "## Model Comparison",
        "",
        _markdown_table(summary["rows"], speed_columns),
        "",
        "## Fallback Comparison",
        "",
        _markdown_table(summary["fallback_rows"], fallback_columns),
        "",
        "## Notes",
        "",
        "* `model_only_fps`는 drawing/write 없이 read + predict 기준입니다.",
        "* `end_to_end_fps`는 drawing/write까지 포함한 영상 산출 기준입니다.",
        "* warm-up run은 summary 집계에서 제외했습니다.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _preview(scenario: MeasurementScenario, config: TargetConfig, scopes: list[str]) -> None:
    print("[dry-run] profiling plan")
    print(f"[dry-run] repo_root={REPO_ROOT}")
    print(f"[dry-run] input_path={scenario.input_path.expanduser().resolve()}")
    print(f"[dry-run] output_dir={scenario.output_dir.expanduser().resolve()}")
    print(f"[dry-run] targets={scenario.targets}")
    print(f"[dry-run] scopes={scopes}")
    print(f"[dry-run] warmup_runs={scenario.warmup_runs}, measured_runs={scenario.measured_runs}")
    print(f"[dry-run] max_frames={scenario.max_frames}, frame_stride={scenario.frame_stride}")
    print(f"[dry-run] profile_mode={scenario.profile_mode}")
    print(f"[dry-run] yolo_pose_weights={config.yolo_pose_weights}")
    print(f"[dry-run] rfdetr_pose_weights={config.rfdetr_pose_weights or '<official-default>'}")
    if any(target.endswith("fallback") for target in scenario.targets):
        print(f"[dry-run] yolo_detector_weights={config.yolo_detector_weights}")
        print(f"[dry-run] rfdetr_detector_weights={config.rfdetr_detector_weights}")


def run_profile(scenario: MeasurementScenario, config: TargetConfig) -> dict[str, Any]:
    _validate_scenario(scenario)
    _validate_target_assets(scenario.targets, config)
    scopes = ["model_only", "end_to_end"] if scenario.pipeline_scope == "both" else [scenario.pipeline_scope]
    final_summaries = []
    for scope in scopes:
        scoped_scenario = MeasurementScenario(**{**asdict(scenario), "pipeline_scope": scope})
        profile_mode = _make_profile_mode(scoped_scenario.profile_mode, scope, scoped_scenario.resource_interval_sec)
        device_policy = _normalise_device(config.device, cuda_sync=profile_mode.cuda_sync)
        run_paths = _make_run_paths(scoped_scenario.output_dir, f"{scoped_scenario.scenario_name}_{scope}", scoped_scenario.overwrite)
        _write_json(
            run_paths.run_dir / "run_config.json",
            {
                "scenario": {
                    **asdict(scoped_scenario),
                    "input_path": str(scoped_scenario.input_path),
                    "output_dir": str(scoped_scenario.output_dir),
                },
                "target_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "profile_mode": asdict(profile_mode),
                "device_policy": asdict(device_policy),
            },
        )
        for target_name in scoped_scenario.targets:
            _run_target(
                target_name=target_name,
                scenario=scoped_scenario,
                config=config,
                run_paths=run_paths,
                device_policy=device_policy,
                profile_mode=profile_mode,
            )
        summary = _aggregate(run_paths, scoped_scenario)
        print(f"[ok] run_dir={run_paths.run_dir}")
        print(f"[ok] summary={run_paths.summary_json_path}")
        print(f"[ok] report={run_paths.report_path}")
        final_summaries.append(summary)
    return {"summaries": final_summaries}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile YOLO11/RF-DETR pose inference speed and resource usage.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--pipeline-scope", default=DEFAULT_PIPELINE_SCOPE, choices=["model_only", "end_to_end", "both"])
    parser.add_argument("--profile-mode", default=DEFAULT_PROFILE_MODE, choices=["low_overhead", "detailed_debug"])
    parser.add_argument("--resource-interval-sec", type=float, default=DEFAULT_RESOURCE_INTERVAL_SEC)
    parser.add_argument("--scenario-name", default="pose_profile")
    parser.add_argument("--yolo-pose-weights", default=str(DEFAULT_YOLO_POSE_WEIGHTS))
    parser.add_argument("--rfdetr-pose-weights")
    parser.add_argument("--yolo-detector-weights", default=str(DEFAULT_YOLO_DETECTOR_WEIGHTS))
    parser.add_argument("--yolo-detector-class-yaml", default=str(DEFAULT_YOLO_DETECTOR_CLASS_YAML))
    parser.add_argument("--rfdetr-detector-weights", default=str(DEFAULT_RFDETR_DETECTOR_WEIGHTS))
    parser.add_argument("--rfdetr-detector-class-yaml", default=str(DEFAULT_RFDETR_DETECTOR_CLASS_YAML))
    parser.add_argument("--rfdetr-detector-variant", default="medium", choices=["auto", "nano", "small", "medium", "large"])
    parser.add_argument("--target-detection-classes", default=",".join(DEFAULT_TARGET_DETECTION_CLASSES))
    parser.add_argument("--yolo-pose-imgsz")
    parser.add_argument("--yolo-detector-imgsz", default="640")
    parser.add_argument("--rfdetr-pose-shape")
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--pose-iou", type=float, default=0.45)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--keypoint-conf", type=float, default=0.2)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.15)
    parser.add_argument("--min-crop-size", type=int, default=32)
    parser.add_argument("--fallback-batch-size", type=int, default=8)
    parser.add_argument("--max-fallback-crops-per-frame", type=int, default=4)
    parser.add_argument("--max-pose-per-crop", type=int, default=1)
    parser.add_argument("--final-nms-iou", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE)
    parser.add_argument("--run", action="store_true", help="Actually run inference. Omit for dry-run preview.")
    return parser


def _scenario_from_args(args: argparse.Namespace) -> MeasurementScenario:
    return MeasurementScenario(
        scenario_name=str(args.scenario_name),
        targets=_parse_csv(args.targets),
        input_path=Path(args.input_path).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        max_frames=int(args.max_frames),
        frame_stride=int(args.frame_stride),
        max_videos=args.max_videos,
        warmup_runs=int(args.warmup_runs),
        measured_runs=int(args.measured_runs),
        profile_mode=str(args.profile_mode),
        pipeline_scope=str(args.pipeline_scope),
        resource_interval_sec=float(args.resource_interval_sec),
        recursive=bool(args.recursive),
        overwrite=bool(args.overwrite),
    )


def _config_from_args(args: argparse.Namespace) -> TargetConfig:
    max_crops = args.max_fallback_crops_per_frame
    return TargetConfig(
        name="pose_profile",
        yolo_pose_weights=args.yolo_pose_weights,
        rfdetr_pose_weights=_parse_optional_path(args.rfdetr_pose_weights),
        yolo_detector_weights=_parse_optional_path(args.yolo_detector_weights),
        yolo_detector_class_yaml=_parse_optional_path(args.yolo_detector_class_yaml),
        rfdetr_detector_weights=_parse_optional_path(args.rfdetr_detector_weights),
        rfdetr_detector_class_yaml=_parse_optional_path(args.rfdetr_detector_class_yaml),
        rfdetr_detector_variant=args.rfdetr_detector_variant,
        target_detection_classes=_parse_csv(args.target_detection_classes),
        device=args.device,
        yolo_pose_imgsz=_normalise_imgsz(args.yolo_pose_imgsz),
        yolo_detector_imgsz=_normalise_imgsz(args.yolo_detector_imgsz),
        rfdetr_pose_shape=_normalise_shape(args.rfdetr_pose_shape),
        pose_conf=float(args.pose_conf),
        pose_iou=float(args.pose_iou),
        detector_conf=float(args.detector_conf),
        detector_iou=float(args.detector_iou),
        keypoint_conf=float(args.keypoint_conf),
        match_iou=float(args.match_iou),
        crop_padding_ratio=float(args.crop_padding_ratio),
        min_crop_size=int(args.min_crop_size),
        fallback_batch_size=int(args.fallback_batch_size),
        max_fallback_crops_per_frame=None if max_crops is None or int(max_crops) < 0 else int(max_crops),
        max_pose_per_crop=int(args.max_pose_per_crop),
        final_nms_iou=float(args.final_nms_iou),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    scenario = _scenario_from_args(args)
    config = _config_from_args(args)
    _validate_scenario(scenario)
    _validate_target_assets(scenario.targets, config)
    scopes = ["model_only", "end_to_end"] if scenario.pipeline_scope == "both" else [scenario.pipeline_scope]
    if not args.run:
        _preview(scenario, config, scopes)
        return
    run_profile(scenario, config)


if __name__ == "__main__":
    main()
