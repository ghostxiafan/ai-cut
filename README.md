# AI 自动剪辑

本地运行的 AI 剪辑流水线：**上传视频 → 自动转写 → 挑高光片段 → 出字幕 → 出成片 → 生成各平台发布文案**。

素材全程留在本机，不经过任何第三方服务器（除文案生成需调用你配置的大模型接口）。

---

## 30 秒上手

```powershell
# 1. 安装
cd <项目目录>
python -m pip install -e .
python -m pip install faster-whisper streamlit opencc-python-reimplemented

# 2. 配置文案密钥（可选，不配则只出片不生成文案）
copy .env.example .env
#    编辑 .env，填入 AICUT_LLM_API_KEY

# 3. 打开网页界面
aicut web          # 浏览器访问 http://localhost:8501
```

网页里拖入视频 → 选画布和时长 → 点「开始自动剪辑」→ 等进度条跑完 → 预览并下载。

---

## 三种用法

### 用法一：网页界面（推荐给手动操作）

```powershell
aicut web
```

侧边栏可调：目标时长、画布（默认跟随素材）、转写模型、是否生成发布文案、目标平台（可多选）。

产出：成片 MP4、字幕 ASS、时间线 JSON、文案 JSON，都在 `out/web/outputs/<视频名>_<时间戳>/`。

### 用法二：一条命令全自动

```powershell
aicut autoplan 我的视频.mp4 `
  --target 30 `
  --render out/clip.mp4 `
  --copy --copy-candidates 3 `
  --platforms "抖音,小红书,视频号,B站" `
  --topic "剪辑效率"
```

### 用法三：分步精细控制

```powershell
aicut transcribe 视频.mp4 -o transcript.json --srt sub.srt   # 转写
aicut select transcript.json --target 30 --top 3             # 看候选打分
aicut autoplan 视频.mp4 -t transcript.json -o timeline.json  # 出时间线
# 手改 timeline.json（调顺序、换字幕、切时长）
aicut render timeline.json -o out.mp4                        # 再渲染
```

已有字幕文件可直接用，省掉转写：

```powershell
aicut autoplan 视频.mp4 -t 现成字幕.srt --render out.mp4
```

---

## 命令一览

| 命令 | 作用 |
|---|---|
| `aicut web` | 启动网页界面 |
| `aicut autoplan <视频>` | 全自动：转写 + 选段 + 时间线 +（可选）渲染 +（可选）文案 |
| `aicut transcribe <视频>` | 转写，导出 JSON / SRT |
| `aicut select <转写>` | 只做候选片段打分，看 AI 会怎么选 |
| `aicut render <时间线>` | 按时间线渲染成片 |
| `aicut copy <文本>` | 生成文案，支持 `-n` 多候选自评分 |
| `aicut probe <文件>` | 查看时长、分辨率、是否有音轨 |
| `aicut backend` | 检查出片后端可用性 |
| `aicut config` | 检查密钥配置（密钥以掩码显示） |
| `aicut demo` | 无素材自检，验证环境是否就绪 |

---

## 环境要求

| 依赖 | 已验证版本 | 说明 |
|---|---|---|
| Python | 3.12.10 | 需 3.10 以上 |
| FFmpeg | 9.0.2 | **必须含 libass**（烧字幕用）；`winget install Gyan.FFmpeg` |
| faster-whisper | 1.2.1 | 不依赖 torch，CPU 可跑 |
| streamlit | 1.66.1 | 仅网页界面需要 |
| 大模型接口 | DeepSeek `deepseek-flash` | 任何 OpenAI 兼容接口均可 |

首次转写会自动下载模型：`medium` 约 1.5GB（推荐，中文准确），`small` 约 500MB（快但易错字）。

---

## 它擅长什么

- **口播 / 教程 / 访谈 / vlog**：有人说话，能靠语义选段和生成字幕
- 快速出**粗剪成片**，再拿去精修
- 一条素材**批量产出多平台文案**（标题/正文/标签按各平台规则分别生成）

## 它不擅长什么

- **纯音乐、纯画面、监控录像**：检测不到人声，只能按时间均分截取，没有字幕和文案
- **精细包装**：无转场、花字、贴纸、特效；需要时在剪映人工完成
- **横屏转竖屏**：无主体跟踪，会留黑边（建议画布选「跟随素材」）

## 剪映能接入吗

当前**不能**。本机剪映专业版 7.1 的草稿文件是加密的，无法写入。

代码里已预留 `adapters/jianying_backend.py`：升级到剪映 11.5+ 后可接入 `jy-draftc`（调用 `videoeditor.dll` 加解密），只需实现一个 `render()`，上层代码不用改。

---

## 密钥安全

- 密钥只存在 `.env`，已在 `.gitignore` 中排除，不会进 Git
- 源码无任何硬编码密钥
- `aicut config` 只显示 `sk-...abcd` 这样的掩码

---

## 开发

```powershell
python -m pytest -q                      # 全部测试（当前 68 passed）
python -m pytest tests/test_render.py -v  # 单个文件
```

想改渲染行为或排查 ffmpeg 问题，先用 dry-run 看命令：

```powershell
aicut render timeline.json -o out.mp4 --dry-run
```

---

## 更多文档

完整设计、模块说明、验证记录、已知限制与后续规划见 **[PROJECT.md](./PROJECT.md)**。

---

## 许可

内部项目，未附开源许可。使用第三方素材与生成内容时，请自行确认版权与平台合规要求（含 AI 生成内容标注）。
