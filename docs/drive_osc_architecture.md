# 驱动文件 + OSC 运行时架构（空间换时间）

## 目标

把整个歌曲库**离线用 GPU 跑成"驱动文件"**，运行时只做极轻量的查表 + 定时触发，
经 OSC 分发到效果层。宿主（UE/播放器）只需发送播放状态事件，不跑模型、不依赖外部信号。

## 核心理念：空间换时间

```
【离线阶段】（重，一次性，GPU）
歌曲库全部歌曲 → all-in-one 分析 → 驱动文件（节拍/段落时间线）

【运行时】（轻，常驻，CPU 查表）
宿主发事件（start/stop/pause/seek）→ 驱动引擎查表 + 定时触发 → OSC 分发到效果层
```

- **为什么可行**：all-in-one 是"整首歌全局上下文"的结构分析，天然不是实时逐帧模型。
  把重计算前置到离线，运行时只剩 O(log n) 二分查表。
- **代价**：驱动文件占用磁盘（每首歌几 KB~几十 KB，可忽略）；歌曲库变更需重新跑离线。

## 组件

### 1. 驱动文件（`drive.py`）

- `build_drive_file(result, source_path, audio_path=None)`：从 all-in-one 结果构建驱动文件 dict
- `analyze_and_save_drive(audio_path, drive_path, device='cuda')`：一站式分析 + 落盘

驱动文件格式（`drive-v1`）：

```json
{
  "meta": {"source": "...", "duration": 287.89, "bpm": 82,
           "beats": 380, "downbeats": 95, "format": "drive-v1"},
  "beats": [
    {"t": 0.19, "downbeat": true, "pos": 1, "strength": 0.82},
    {"t": 0.95, "downbeat": false, "pos": 2, "strength": 0.71}
  ],
  "segments": [
    {"label": "intro", "start": 0.0, "end": 11.9}
  ]
}
```

字段说明：
- `beats[i].t`：拍点时间（秒，相对源音频，时间基准 0）
- `beats[i].downbeat`：是否强拍（小节首拍）
- `beats[i].pos`：小节内拍号（1~N，直接采用 all-in-one 原生 beat_positions，弱起拍会标 2/3/4）
- `beats[i].strength`：触发强度 0~1（低频能量包络在拍点采样；无音频时回退 1.0）

### 2. 运行时驱动引擎（`osc_engine.py`）

- `DriveFile.load(path)`：加载驱动文件，`beat_index_at(t)` / `segment_at(t)` 二分查表
- `DriveEngine(song_id, drive, callback)`：单首歌状态机 + 定时触发线程
- `EngineRegistry`：多首歌路由
- `serve_osc(drive_map, ...)`：OSC 服务入口

事件协议（宿主 → 引擎，`/drive/control/*`）：

| 地址 | 参数 | 语义 |
|---|---|---|
| `/drive/control/start` | song_id, [pos] | 开始播放 |
| `/drive/control/stop` | song_id | 停止 |
| `/drive/control/pause` | song_id | 暂停 |
| `/drive/control/resume` | song_id, [pos] | 恢复播放 |
| `/drive/control/seek` | song_id, pos | 重定位到 pos 秒 |

输出协议（引擎 → 效果层，`/drive/out/*`）：

| 地址 | 参数 | 语义 |
|---|---|---|
| `/drive/out/beat` | song_id, t, pos, is_downbeat, strength | 到达一个拍点 |
| `/drive/out/segment` | song_id, label, start, end | 段落切换 |
| `/drive/out/state` | song_id, state | 播放状态（start/stop/pause/resume/seek/end）|

### 3. 批量离线（`examples/build_drive_library.py`）

```bash
python examples/build_drive_library.py music/ drives/ --device cuda
```

遍历音频目录，每个生成同名 `.drive.json`，已存在的跳过（幂等，可断点续跑）。

### 4. OSC 服务入口（`examples/serve_drive_osc.py`）

```bash
python examples/serve_drive_osc.py --map song_a=drives/a.drive.json --map song_b=drives/b.drive.json
```

## 设计决策（可调整项）

| 决策点 | 当前默认 | 备注 |
|---|---|---|
| 驱动文件粒度 | 拍级 + 段落 + 每拍强度 | 甜点：文件小，运行时查表 + 插值即可 |
| 重定位恢复 | 引擎二分查表定位到最近拍 + 同步段落 | 宿主只需发 seek，无需理解节拍 |
| 时间基准 | 相对源音频的 0 秒 | 需与宿主播放时间对齐，注意 mp3 有 20-40ms 编码延迟 |
| 触发强度 | 低频(<120Hz)能量包络 | 供效果层映射亮度/力度，可按需换成其他特征 |
| tick 间隔 | 5ms | 平衡精度与 CPU 开销，可按需调整 |

## 待办

1. [x] 驱动文件生成（drive.py）
2. [x] 运行时引擎 + OSC 服务（osc_engine.py / serve_drive_osc.py）
3. [x] 批量离线脚本（build_drive_library.py）
4. [x] 端到端验证（查表 / seek / 状态机）
5. [ ] 真实 OSC 收发联调（宿主侧发事件 + 效果层收事件）
6. [ ] 与宿主播放时间基准对齐（采样率/偏移校准）
