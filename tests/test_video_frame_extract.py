from pathlib import Path

from labelstudio_bbox_tools.video.frame_extract import _frame_file_name, _frame_indices, _safe_name, _select_step_frames, _video_output_dir


def test_interval_seconds_uses_video_fps():
    step, mode, value = _select_step_frames(fps=29.97, interval_seconds=2.0, every_n_frames=None, target_fps=None)
    assert step == 60
    assert mode == "interval_seconds"
    assert value == 2.0


def test_frame_indices_respect_start_end_and_limit():
    indices = _frame_indices(
        fps=30.0,
        frame_count=300,
        step_frames=60,
        start_seconds=2.0,
        end_seconds=8.0,
        max_frames_per_video=2,
    )
    assert indices == [60, 120]


def test_output_dir_preserves_relative_video_stem():
    out_dir = _video_output_dir(Path("/tmp/out"), Path("site_a/cam 01.mp4"))
    assert out_dir == Path("/tmp/out/frames/site_a/cam_01")


def test_frame_file_name_keeps_traceable_fields():
    assert _frame_file_name("cam 01", 3, 90, 3.0, "jpg") == "cam_01__idx000003__frame00000090__t000003.000s.jpg"
    assert _safe_name("a/b c") == "a_b_c"
