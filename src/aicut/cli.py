"""命令行入口。

    aicut probe <media>
    aicut transcribe <media> -o transcript.json [--srt out.srt]
    aicut select <transcript.json> --target 30 --top 3
    aicut autoplan <media> -t transcript.json -o timeline.json [--render out.mp4]
    aicut render <timeline.json> -o out.mp4 [--dry-run] [--no-subtitles]
    aicut demo -o out.mp4            # 生成测试素材并跑通全链路
    aicut copy <transcript.txt>      # 生成文案（需 AICUT_LLM_API_KEY）
    aicut backend                    # 查看出片后端可用性
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# HuggingFace 在 Windows 无开发者模式下会打印符号链接缓存告警，与功能无关
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from . import __version__
from .adapters.ffmpeg_backend import FFmpegBackend
from .adapters.jianying_backend import JianyingBackend
from .analysis import highlights
from .analysis.media import FFmpegMissingError, probe
from .analysis.transcript import load_transcript, Transcript, transcript_from_text
from .ir import IRValidationError, Timeline, load_timeline, save_timeline
from .plan import PlanOptions, build_timeline, plan_summary
from .render import describe, render


def _cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.media)
    print(json.dumps(info.__dict__, ensure_ascii=False, indent=2))
    return 0


def _load_or_build_transcript(args: argparse.Namespace, media: str | None = None) -> Transcript:
    """优先用已有的转写文件；没有则调用 ASR；ASR 不可用时给出明确提示。"""
    if args.transcript:
        return load_transcript(args.transcript)
    if args.text:
        text = Path(args.text).read_text(encoding="utf-8", errors="replace") if Path(args.text).exists() else args.text
        return transcript_from_text(text, source=str(media or ""))

    from .analysis.asr import extract_audio, transcribe

    if not media:
        raise ValueError("需要提供 --transcript / --text，或给出视频以便自动转写")
    work = Path(args.workdir)
    wav = extract_audio(media, work / "audio_16k.wav")
    print(f"已抽音频: {wav}，开始转写（模型 {args.model}）…")
    segments = transcribe(wav, model_size=args.model, language=args.lang)
    transcript = Transcript(segments=segments, language=args.lang, source=str(media))
    print(f"转写完成：{len(transcript)} 段，{transcript.duration:.1f}s")
    return transcript


def _cmd_transcribe(args: argparse.Namespace) -> int:
    transcript = _load_or_build_transcript(args, args.media)
    out = Path(args.output)
    transcript.save(out)
    print(f"已保存转写: {out}（{len(transcript)} 段 / {transcript.duration:.1f}s）")
    if args.srt:
        print(f"已保存字幕: {transcript.to_srt(args.srt)}")
    if not args.quiet:
        for seg in transcript.segments[:20]:
            print(f"  [{seg.start:7.2f} → {seg.end:7.2f}] {seg.text}")
        if len(transcript) > 20:
            print(f"  … 其余 {len(transcript) - 20} 段已省略")
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    transcript = load_transcript(args.transcript)
    candidates = highlights.select(
        transcript,
        target=args.target,
        max_duration=args.max_duration,
        top=args.top,
    )
    if not candidates:
        print("未找到合适片段：视频可能过短，或转写内容为空。", file=sys.stderr)
        return 1

    print(f"候选片段 Top{len(candidates)}（目标 {args.target:.0f}s）:")
    for i, cand in enumerate(candidates, start=1):
        reasons = " ".join(f"{k}+{v:.0f}" for k, v in cand.reasons.items())
        print(f"\n#{i}  {cand.start:.2f}s → {cand.end:.2f}s  时长 {cand.duration:.1f}s  得分 {cand.score:.1f}")
        print(f"    {reasons}")
        print(f"    {cand.text[:100]}…" if len(cand.text) > 100 else f"    {cand.text}")

    if args.output:
        payload = [c.to_dict() for c in candidates]
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存: {args.output}")
    return 0


def _cmd_autoplan(args: argparse.Namespace) -> int:
    media = args.media
    info = probe(media)
    if not info.has_audio:
        print("警告: 素材没有音轨，无法自动转写，可改用 --text 提供文案。", file=sys.stderr)

    transcript = _load_or_build_transcript(args, media)
    candidates = highlights.select(
        transcript,
        target=args.target,
        max_duration=args.max_duration,
        top=args.top,
    )
    if not candidates:
        print("未找到合适片段，请调整 --target / --max-duration 或提供更长素材。", file=sys.stderr)
        return 1

    if args.llm_rank:
        candidates = highlights.rerank_with_llm(candidates, top=args.top, model=args.model_llm or None)

    timeline = build_timeline(
        media,
        transcript,
        candidates,
        PlanOptions(
            width=args.width,
            height=args.height,
            fps=args.fps,
            subtitles=not args.no_subtitles,
            muzzle_source=args.mute_source,
            hook=args.hook,
            auto_hook=not args.no_hook,
            bgm=args.bgm,
            bgm_volume=args.bgm_volume,
        ),
    )

    timeline_path = save_timeline(timeline, args.output)
    print(f"已生成时间线: {timeline_path}")
    print(json.dumps(plan_summary(candidates, timeline), ensure_ascii=False, indent=2))
    if args.srt:
        cues = []
        from .subtitle import Cue

        for t in timeline.text:
            cues.append(Cue(t.start, t.end, t.text, t.style))
        from .subtitle import write_srt

        print(f"已保存字幕: {write_srt(cues, args.srt)}")

    if args.render:
        result = render(timeline, args.render, crf=args.crf, preset=args.preset)
        size_mb = Path(result.output).stat().st_size / 1024 / 1024
        print(f"\n成片: {result.output}  ({result.duration:.2f}s, {size_mb:.1f} MB)")

    if args.copy:
        _write_copy(transcript, candidates, args)
    return 0


def _write_copy(transcript, candidates, args: argparse.Namespace) -> None:
    """生成文案并与时间线一起落盘，失败不影响已产出的视频。"""
    from .copywriter.engine import best, generate, generate_candidates

    picked = "\n".join(c.text for c in candidates) or transcript.plain_text()
    out = Path(args.copy)

    try:
        if args.copy_candidates > 1:
            scored = generate_candidates(
                picked,
                topic=args.topic or "",
                platform=args.platform,
                count=args.copy_candidates,
                model=args.model_llm or None,
            )
            top = best(scored)
            payload = {
                "best": top.to_dict(),
                "candidates": [s.to_dict() for s in scored],
            }
        else:
            result = generate(
                picked,
                topic=args.topic or "",
                platform=args.platform,
                model=args.model_llm or None,
            )
            top = None
            payload = result.to_dict()
    except Exception as exc:  # 文案是增强项，不应阻断出片
        print(f"\n文案生成失败（不影响成片）: {exc}", file=sys.stderr)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    chosen = top.result if top else payload
    print(f"\n文案: {out}")
    if top is not None:
        print(f"  共 {len(payload['candidates'])} 条候选，最高分 {top.total:.0f}/100")
    print(f"  开场钩子: {chosen['hook'] if isinstance(chosen, dict) else chosen.hook}")
    print(f"  标题    : {chosen['title'] if isinstance(chosen, dict) else chosen.title}")
    print(f"  封面字  : {chosen['cover_text'] if isinstance(chosen, dict) else chosen.cover_text}")
    tags = chosen["hashtags"] if isinstance(chosen, dict) else chosen.hashtags
    print(f"  标签    : {' '.join(tags)}")


def _cmd_render(args: argparse.Namespace) -> int:
    timeline = load_timeline(args.timeline)
    print(describe(timeline))
    if args.dry_run:
        result = render(
            timeline, args.output,
            burn_subtitles=not args.no_subtitles,
            crf=args.crf, preset=args.preset, dry_run=True,
        )
        print("\nffmpeg " + " ".join(f'"{c}"' if " " in c else c for c in result.command[1:]))
        return 0

    result = render(
        timeline, args.output,
        burn_subtitles=not args.no_subtitles,
        crf=args.crf, preset=args.preset,
    )
    size_mb = Path(result.output).stat().st_size / 1024 / 1024
    print(f"\n完成: {result.output}  ({result.duration:.2f}s, {size_mb:.1f} MB)")
    return 0


def _cmd_copy(args: argparse.Namespace) -> int:
    from .copywriter.engine import best, generate, generate_candidates

    text = Path(args.transcript).read_text(encoding="utf-8", errors="replace") if Path(args.transcript).exists() else args.transcript
    if args.candidates > 1:
        scored = generate_candidates(
            text,
            topic=args.topic,
            platform=args.platform,
            count=args.candidates,
            model=args.model,
        )
        _print_scored(scored)
        top = best(scored)
        payload: dict = {"best": top.to_dict(), "candidates": [s.to_dict() for s in scored]}
        if args.out:
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n已保存 {len(scored)} 条候选: {args.out}")
        return 0

    result = generate(text, topic=args.topic, platform=args.platform, model=args.model)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存: {args.out}")
    return 0


def _print_scored(scored) -> None:
    print(f"候选 {len(scored)} 条（按总分降序）:\n")
    for rank, item in enumerate(scored, start=1):
        dims = " ".join(f"{k[:4]}{v:.0f}" for k, v in item.scores.items())
        print(f"#{rank}  总分 {item.total:.0f}/100   {dims}")
        print(f"    钩子: {item.result.hook}")
        print(f"    标题: {item.result.title}")
        print(f"    封面: {item.result.cover_text}")
        if item.note:
            print(f"    点评: {item.note}")
        print()


def _cmd_config(_args: argparse.Namespace) -> int:
    """只显示掩码后的密钥，避免任何形式的明文回显。"""
    from .config import ConfigError, resolve

    try:
        settings = resolve()
    except ConfigError as exc:
        print(f"未配置: {exc}", file=sys.stderr)
        print("提示：复制 .env.example 为 .env 并填入 AICUT_LLM_API_KEY", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "api_key": settings.masked_key,
                "base_url": settings.base_url,
                "model": settings.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_backend(_args: argparse.Namespace) -> int:
    ff = FFmpegBackend()
    jy = JianyingBackend()
    print(json.dumps({"ffmpeg": {"available": ff.available()}, "jianying": jy.diagnose()}, ensure_ascii=False, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """生成两段测试素材 + 一条 timeline，验证渲染链路。"""
    from .demo import build_demo

    work = Path(args.workdir)
    timeline = build_demo(work)
    timeline_path = save_timeline(timeline, work / "timeline.json")
    print(f"素材与时间线已生成: {timeline_path}")
    print(describe(timeline))
    result = render(timeline, args.output, crf=args.crf, preset=args.preset)
    size_mb = Path(result.output).stat().st_size / 1024 / 1024
    print(f"\n完成: {result.output}  ({result.duration:.2f}s, {size_mb:.1f} MB)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aicut", description=f"AI 辅助剪辑工具 v{__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="查看媒体信息")
    p.add_argument("media")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("transcribe", help="转写音频/视频，产出带时间戳的转写文件")
    p.add_argument("media")
    p.add_argument("-o", "--output", default="out/transcript.json")
    p.add_argument("--srt", default=None, help="同时导出 SRT")
    p.add_argument("--transcript", default=None, help="复用已有转写（.json/.srt/.txt）")
    p.add_argument("--text", default=None, help="直接给文本或文本文件，跳过 ASR")
    p.add_argument("--model", default="medium", help="faster-whisper 模型：tiny/base/small/medium/large-v3")
    p.add_argument("--lang", default="zh")
    p.add_argument("--workdir", default="out/work")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_transcribe)

    p = sub.add_parser("select", help="对转写做候选片段打分")
    p.add_argument("transcript")
    p.add_argument("--target", type=float, default=30.0, help="目标成片时长（秒）")
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("-o", "--output", default=None, help="把候选结果保存为 JSON")
    p.set_defaults(func=_cmd_select)

    p = sub.add_parser("autoplan", help="转写+选段+生成时间线，可一键出片")
    p.add_argument("media")
    p.add_argument("-t", "--transcript", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("-o", "--output", default="out/timeline.json")
    p.add_argument("--render", default=None, help="给出路径则直接渲染成片")
    p.add_argument("--srt", default=None)
    p.add_argument("--target", type=float, default=30.0)
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--top", type=int, default=1)
    p.add_argument("--model", default="medium")
    p.add_argument("--lang", default="zh")
    p.add_argument("--workdir", default="out/work")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-subtitles", action="store_true")
    p.add_argument("--mute-source", action="store_true", help="静音原声，仅保留字幕与 BGM")
    p.add_argument("--hook", default=None, help="指定开场钩子文案")
    p.add_argument("--no-hook", action="store_true")
    p.add_argument("--bgm", default=None)
    p.add_argument("--bgm-volume", type=float, default=0.12)
    p.add_argument("--llm-rank", action="store_true", help="用 LLM 复排候选（需 API Key）")
    p.add_argument("--model-llm", default=None)
    p.add_argument("--copy", nargs="?", const="out/copy.json", default=None,
                   help="生成爆款文案并保存（可指定路径，默认 out/copy.json）")
    p.add_argument("--copy-candidates", type=int, default=1,
                   help="文案候选条数，>1 时生成多条并自评分")
    p.add_argument("--topic", default="")
    p.add_argument("--platform", default="抖音")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="medium")
    p.set_defaults(func=_cmd_autoplan)

    p = sub.add_parser("render", help="按时间线渲染成片")
    p.add_argument("timeline")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--dry-run", action="store_true", help="只打印 ffmpeg 命令")
    p.add_argument("--no-subtitles", action="store_true")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="medium")
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("copy", help="根据转写文本生成文案")
    p.add_argument("transcript", help="转写文本文件路径，或直接给文本")
    p.add_argument("--topic", default="")
    p.add_argument("--platform", default="抖音")
    p.add_argument("--model", default=None)
    p.add_argument("-n", "--candidates", type=int, default=1,
                   help="生成多条候选并自评分（>1 时启用）")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_copy)

    p = sub.add_parser("backend", help="查看出片后端可用性")
    p.set_defaults(func=_cmd_backend)

    p = sub.add_parser("config", help="查看 LLM 配置（密钥以掩码显示）")
    p.set_defaults(func=_cmd_config)

    p = sub.add_parser("demo", help="生成测试素材并跑通全链路")
    p.add_argument("-o", "--output", default="out/demo.mp4")
    p.add_argument("--workdir", default="out/demo_assets")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--preset", default="veryfast")
    p.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认非 UTF-8，中文提示会乱码
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (IRValidationError, FFmpegMissingError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
