"""本地网页界面：上传视频 → 自动剪辑 → 预览与下载成片。

启动：
    aicut web
或：
    streamlit run src/aicut/webui.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# 支持直接从源码目录运行（streamlit run src/aicut/webui.py）
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aicut.analysis import highlights  # noqa: E402
from aicut.analysis.asr import extract_audio, transcribe  # noqa: E402
from aicut.analysis.media import probe  # noqa: E402
from aicut.analysis.transcript import Transcript, load_transcript, transcript_from_text  # noqa: E402
from aicut.copywriter.engine import best, generate, generate_candidates  # noqa: E402
from aicut.ir import save_timeline  # noqa: E402
from aicut.plan import PlanOptions, build_timeline  # noqa: E402
from aicut.render import render  # noqa: E402

WORK = Path("out/web")
UPLOADS = WORK / "uploads"
OUTPUTS = WORK / "outputs"

st.set_page_config(page_title="AI 自动剪辑", page_icon="🎬", layout="wide")


def _human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _save_upload(uploaded) -> Path:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    target = UPLOADS / uploaded.name
    with target.open("wb") as fh:
        fh.write(uploaded.getbuffer())
    return target


def _status(label: str):
    """返回 (st.status 上下文, 计时器)，保证每次都能正常收尾。"""
    return st.status(label, expanded=True)


st.title("🎬 AI 自动剪辑")
st.caption("上传视频 → 自动转写、选段、出字幕、出片、生成爆款文案")

with st.sidebar:
    st.header("参数")
    target = st.slider("目标成片时长（秒）", 10, 90, 30, step=5)
    aspect = st.selectbox("画布", ["竖屏 1080×1920", "横屏 1920×1080", "方形 1080×1080"], index=0)
    model_size = st.selectbox(
        "转写模型",
        ["medium", "small", "tiny", "base"],
        index=0,
        help="medium 中文最准但较慢；tiny 最快但错字多",
    )
    top_n = st.number_input("选取片段数", 1, 5, 1, help="多段会按时间顺序拼接")
    presets = st.selectbox("导出质量", ["ultrafast（最快）", "veryfast", "medium（体积小）"], index=0)
    preset = presets.split("（")[0]

    st.divider()
    want_copy = st.checkbox("生成爆款文案", value=True)
    copy_n = st.number_input("文案候选数", 1, 8, 3, disabled=not want_copy)
    topic = st.text_input("视频主题（可选）", placeholder="如：AI自动剪辑")
    platform = st.selectbox("目标平台", ["抖音", "小红书", "视频号", "B站"], index=0)

    st.divider()
    st.caption("密钥配置")
    try:
        from aicut.config import ConfigError, resolve

        resolve()
        st.success("DeepSeek 已配置", icon="✅")
    except ConfigError:
        st.error("未配置 .env", icon="⚠️")
    except Exception:
        st.warning("配置读取异常", icon="⚠️")

uploaded = st.file_uploader("上传视频", type=["mp4", "mov", "mkv", "avi", "flv", "webm", "m4v"])

if uploaded is None:
    st.info("支持常见格式：MP4 / MOV / MKV / AVI / WEBM。视频越大处理越慢。")
    st.stop()

video_path = _save_upload(uploaded)
info = probe(video_path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("时长", f"{info.duration:.1f}s")
col2.metric("分辨率", f"{info.width}×{info.height}")
col3.metric("大小", _human_size(video_path.stat().st_size))
col4.metric("音轨", "有" if info.has_audio else "无")

st.video(str(video_path))

if st.button("开始自动剪辑", type="primary", use_container_width=True):
    OUT = OUTPUTS / f"{video_path.stem}_{int(time.time())}"
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    try:
        with st.status("1/5 提取音频并转写…", expanded=True) as status:
            if info.has_audio:
                wav = extract_audio(video_path, OUT / "audio.wav")
                st.write(f"已抽取音频 {wav.name}")
                segs = transcribe(wav, model_size=model_size)
                transcript = Transcript(segments=segs, source=str(video_path))
            else:
                st.warning("视频没有音轨，已跳转字幕，请在下一次运行时改用文案模式")
                transcript = Transcript()
            status.update(label="1/5 转写完成", state="complete")

        with st.status("2/5 候选片段打分…", expanded=True) as status:
            candidates = highlights.select(
                transcript, target=float(target), max_duration=float(target) * 2, top=int(top_n)
            )
            if not candidates:
                st.error("没有找到合适片段：视频可能过短或没有语音。")
                st.stop()
            for i, c in enumerate(candidates, 1):
                st.write(f"**#{i}** {c.start:.1f}s → {c.end:.1f}s ｜ 得分 {c.score:.1f}")
                st.caption(c.text[:90] + ("…" if len(c.text) > 90 else ""))
            status.update(label="2/5 选段完成", state="complete")

        with st.status("3/5 生成时间线…", expanded=True) as status:
            w, h = (1080, 1920) if "竖屏" in aspect else (1920, 1080) if "横屏" in aspect else (1080, 1080)
            timeline = build_timeline(
                video_path,
                transcript,
                candidates,
                PlanOptions(width=w, height=h, fps=30, subtitles=True),
            )
            tl_path = save_timeline(timeline, OUT / "timeline.json")
            st.write(f"时间线已保存：{tl_path.name}（{len(timeline.text)} 条字幕）")
            status.update(label="3/5 时间线完成", state="complete")

        with st.status("4/5 渲染成片…", expanded=True) as status:
            out_video = OUT / "final.mp4"
            render(timeline, out_video, crf=20, preset=preset)
            st.write(f"成片 {_human_size(out_video.stat().st_size)}")
            status.update(label="4/5 渲染完成", state="complete")

        copy_payload = None
        if want_copy:
            with st.status("5/5 生成文案…", expanded=True) as status:
                text = "\n".join(c.text for c in candidates) or transcript.plain_text()
                try:
                    if int(copy_n) > 1:
                        scored = generate_candidates(
                            text, topic=topic, platform=platform, count=int(copy_n)
                        )
                        copy_payload = {
                            "best": best(scored).to_dict(),
                            "candidates": [s.to_dict() for s in scored],
                        }
                    else:
                        copy_payload = generate(text, topic=topic, platform=platform).to_dict()
                    (OUT / "copy.json").write_text(
                        json.dumps(copy_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    status.update(label="5/5 文案完成", state="complete")
                except Exception as exc:
                    st.warning(f"文案生成失败：{exc}")
                    status.update(label="5/5 文案跳过", state="complete")

        st.session_state["result"] = {
            "dir": str(OUT),
            "video": str(out_video),
            "copy": copy_payload,
            "elapsed": time.time() - started,
        }
    except Exception as exc:
        st.error(f"处理失败：{exc}")
        st.exception(exc)
        st.stop()

result = st.session_state.get("result")
if result:
    st.success(f"完成，用时 {result['elapsed']:.1f} 秒")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("成片预览")
        st.video(result["video"])
        st.download_button(
            "⬇ 下载成片",
            data=Path(result["video"]).read_bytes(),
            file_name=Path(result["video"]).name,
            mime="video/mp4",
            use_container_width=True,
        )

    with right:
        payload = result.get("copy")
        if payload:
            st.subheader("爆款文案")
            top = payload.get("best", payload)
            if "total" in top:
                st.metric("自评分", f"{top['total']:.0f}/100")
            st.markdown(f"**钩子**：{top.get('hook', '')}")
            st.markdown(f"**标题**：{top.get('title', '')}")
            st.markdown(f"**封面字**：{top.get('cover_text', '')}")
            st.markdown(f"**标签**：{' '.join(top.get('hashtags', []))}")
            if top.get("comment_bait"):
                st.markdown(f"**引导评论**：{top['comment_bait']}")
            if top.get("oral_script"):
                with st.expander("口播稿"):
                    st.write(top["oral_script"])

            others = payload.get("candidates", [])
            if len(others) > 1:
                with st.expander(f"其他 {len(others) - 1} 条候选"):
                    for i, cand in enumerate(others[1:], 2):
                        st.markdown(f"**#{i} 总分 {cand.get('total', 0):.0f}**")
                        st.caption(
                            f"{cand.get('hook', '')} ｜ {cand.get('title', '')} ｜ {cand.get('note', '')}"
                        )

    with st.expander("产物文件"):
        for f in sorted(Path(result["dir"]).iterdir()):
            if f.is_file():
                st.write(f"{f.name}  ·  {_human_size(f.stat().st_size)}")
