# 数据管理与流程 — 设计文档 v0.1

本文档记录 ELO 项目的数据下载、存储、结果产出与运行流程。设计公式与模型选择见 `ELO_PROJECT_DESIGN.md`；本文档只包含数据管理与流程相关的决策。

---

## 1. 数据下载与存储规范

### 1.1 下载单位与抓取顺序

下载分三个粒度：

| 粒度 | 端点 | 频率 | 产出 |
|------|------|------|------|
| 赛季级 | `LeagueGameLog`（Regular + Playoffs 各一次） | 每赛季 1 次 | 该赛季全部 game_id 清单（每场主/客各一行） |
| 比赛级 | `BoxScoreTraditionalV3` | 每个 game_id 1 次 | 球员 ↔ 比赛对应关系与全部 box score 字段 |
| 字典级 | `teams`、`CommonAllPlayers`、`DraftHistory` | 一次性 + 每赛季刷新 | 球队/球员静态信息、roster 状态、选秀顺位 |

**最小下载单位是一场比赛的 box score**：先按赛季拿到比赛全集，再逐场抓取两份 box score。这样天然得到"哪些球员打了哪场比赛"，不需要反向按球员遍历。

抓取顺序：

```
for season in 1976-77 .. 2025-26:
    games += LeagueGameLog(season, Regular) + LeagueGameLog(season, Playoffs)
for game_id in sorted(games):
    fetch BoxScoreTraditionalV3(game_id)
    fetch BoxScoreAdvancedV3(game_id)   # 1996-97 前返回空/不可用，也缓存结果
```

- 每次请求先查 `api_cache`，命中直接复用；失败写入 `fetch_log` 并按指数退避重试。
- 原始 JSON 完整保存，normalized 表全部由原始响应生成，Raw 绝不修改。
- 量级估算：约 59,000 场 × 2 端点 ≈ 11.8 万次调用。配合并发、缓存、断点续跑，可在数小时内完成。

### 1.2 球员与比赛的对应关系

对应关系由三个 ID 直接构成：

- `game_id`：比赛主键（LeagueGameLog 与 box score 一致）
- `person_id`：球员主键（nba_api 跨赛季稳定）
- `team_id`：当场所属球队（换队时随比赛变化）

BoxScore V3 每行自带 `gameId / personId / teamId`，因此对应关系直接落表，无需推断。

| 情况 | 处理 |
|------|------|
| DNP / 0 分钟 | 保留行并标记 `played=0`；Elo 不更新，但可用于 roster 追溯 |
| 同赛季换队 | `(game_id, player_id, team_id)` 唯一；Elo 按时间顺序自然衔接 |
| 主客队判定 | `team_game_log` 每场两行，按 `MATCHUP` 中 `@`（客）/ `vs.`（主）解析并交叉校验 |
| 重复抓取 | 幂等 upsert + 唯一约束 |
| 重名球员 | 以 `person_id` 为准，姓名仅作展示 |
| 取消/延期比赛 | 以实际返回的 WL 与 box score 为准；无数据的计划比赛标记异常 |

抓取完整性 gate（全部满足才允许跑 Elo）：

- 每个 game_id 恰好 2 条 `team_game_log` 行（常规赛/季后赛）；
- 每个 game_id 至少 1 条 `played=1` 的 box score 行；
- advanced_boxscores 表暂不填充（Phase 1 不调用 AdvancedV3，未来补）；
- 任一 game_id 缺失则进入异常清单，补齐或显式豁免前阻断运行；2023-24、2024-25 的已知少量比赛缺失按显式豁免处理。

#### processed_data 可用性 gate（Kaggle CSV 基线）

- 比赛双方均至少有一条 `played=1` box score；
- 每队 `sum(played=1 minutes)` 与 `team_game_log.numMinutes` 偏差不超过 ±10 分钟（`numMinutes` 为 240+25k）；
- 当前基线保留 58,927 场，剔除 382 场；
- 剔除场次写入 `KNOWN_EXEMPTIONS`，保留在 `games.csv` / `boxscores.csv` 中，不物理删除。

### 1.3 Raw Layer Schema（增量）

在 `ELO_PROJECT_DESIGN.md` 原有表基础上补充（同一文件 `nba_raw.db`）：

```
api_cache            # (endpoint, params_json) 唯一，保存原始 JSON
fetch_log            # 每次请求状态：game_id, endpoint, status, rows, fetched_at, error
team_game_log        # LeagueGameLog 原样：game_id, season, season_type, team_id, matchup, wl
games                # 去重比赛主表：game_id, game_date, season, season_type, home/away, scores
players              # player_id, name, is_active
player_season_meta   # CommonAllPlayers 每赛季 roster 状态
boxscores            # (game_id, player_id, team_id) 唯一：minutes, 传统统计, plus_minus, played
advanced_boxscores   # 暂不填充，Phase 2（BPM）时补充
```

索引：`games(game_date)`、`boxscores(game_id)`、`boxscores(player_id, game_id)`。

---

## 2. Elo 结果存储

### 2.1 设计原则

- Elo 结果是 Derived Layer 产物，独立存 `nba_elo.db`，可从 `nba_raw.db` + 参数完整重建。
- 所有结果带 `run_id`：burn-in、参数搜索每个 trial、双向初始化三遍，都是独立 run。
- 一次运行写四类表：`elo_runs`（元数据）、`elo_updates`（事件日志）、`elo_states`（状态）、`elo_snapshot_season`（赛季末物化）。
- **不物化每日全量快照**：参数搜索会产生大量 run，每日全量 × run 会爆炸；用状态表做 as-of 查询即可。

### 2.2 表设计

```
elo_runs
  run_id, name, params_json, init_method, pass_no, code_version,
  status, is_final, created_at

elo_init
  run_id, player_id, first_game_id, rating, source, created_at

elo_updates          # 事件日志：每场每人一行
  run_id, game_id, player_id, team_id, track, game_date, season,
  rating_before, rating_after, delta_raw, delta_adj, zero_mean_offset,
  minutes_share, perf_i, s_team, e_team, k_effective
  PK (run_id, game_id, player_id)
  # 审计字段（D-020）：表现归因
  game_score, z_game_score, on_court_rate_shrunk, z_on_court_rate, home

elo_states           # 状态：每场更新后一行
  run_id, player_id, game_id, track, game_date, rating, games_played
  PK (run_id, player_id, game_id)

elo_adjustments      # 非比赛调整（未来赛季锚定等）
  run_id, player_id, season, event_type, delta, rating_before, rating_after

elo_snapshot_season  # 赛季末物化：每 run 每球员每赛季一行
  run_id, season, player_id, rating_at_start, rating_at_end, games_played

param_search_trials  # 调参日志
  trial_id, run_id, params_json, loss, window_start, window_end
```

### 2.3 回答两类查询

- **任意球员生涯**：`elo_updates` 按 `(run_id, player_id) + game_date, game_id` 排序，得到每场后的 `rating_after` 与变化 `delta_adj`；同时保留 `perf_i`、`k_effective` 等审计字段。
- **任意日期全联盟快照**：`elo_states` 做 as-of 查询（每人取 `game_date <= 目标日` 的最后一行；未上场球员状态不变；尚未首秀的球员不出现）。
- **每年快照**：直接读 `elo_snapshot_season`，含赛季初/末分数与出场数。

语义约定：

- 某日快照 = 该日最后一场比赛结束后的状态；无比赛日期状态不变。
- 常规赛与季后赛用 `track` 区分，对外默认展示常规赛 track。
- 新秀初始值写入 `elo_init`，触发时机是**首次出场**（`first_game_id`），不是选秀年。

---

## 3. 其他必要问题

| # | 问题 | 建议 |
|---|------|------|
| 1 | 可复现性 | 每次运行记录代码版本、参数、init_method；公式改动后旧 run 仍可对比 |
| 2 | 比赛处理顺序 | 按 `game_date, game_id` 升序；同日多场顺序固定并写入决策日志 |
| 3 | DNP 与长期缺阵 | 状态保持不更新；`games_played` 区分是否出战 |
| 4 | 新秀初始值 | 首次出场时按选秀顺位初始化；落选秀、推迟入联盟、从未出场要有明确规则 |
| 5 | Play-In（2021 起） | 已确认：NBA Stats API 归类为 Playoffs SeasonType，归入季后赛 track |
| 6 | 全明星赛 | 不在 LeagueGameLog 常规/季后赛中，天然排除 |
| 7 | 早期数据缺失 | +/-、advanced 缺失写入可用性标记，供交叉验证与未来模型使用 |
| 8 | 增量更新 | 新赛季/新比赛日只拉新增 game_id；api_cache + fetch_log 支持断点续跑 |
| 9 | SQLite 性能 | WAL、批量事务、索引；参数搜索多 run 并发写互不干扰 |
| 10 | 下游共享 | Elo 结果进入 Derived/Model Layer，BPM、胜率等模型直接读 `elo_updates` / `elo_states` |

---

## 4. 决策日志

### DM-001：下载单位

**选择**：比赛列表按赛季、box score 按比赛、静态数据按字典下载

**未选**：按球员下载 game log、按赛季下载全量 box score 后自行拼接

**理由**：最小下载单位是一场比赛的 box score，能天然产出球员 ↔ 比赛对应关系；按球员遍历会重复抓取且依赖不可靠的逐球员 game log 历史完整性；按赛季全量 box score 没有对应 API。

### DM-002：结果存储

**选择**：事件日志 `elo_updates` + 状态表 `elo_states` + 赛季末物化 `elo_snapshot_season`，不物化每日全量快照

**未选**：只存最终分数、只存每日全量快照

**理由**：最终分数不可审计，无法回答"每场后变化"；每日全量 × 大量参数 run 会爆炸。事件日志可审计，状态表支持任意日期 as-of 查询，赛季末物化满足"每年"需求。

### DM-003：快照语义

**选择**：某日快照 = 该日最后一场比赛结束后的状态；无比赛日状态不变

**理由**：给"每天"一个无歧义、可复现的定义。

### DM-004：新秀初始化时机

**选择**：首次出场时初始化，而不是选秀年

**理由**：不少球员选秀后隔年或多年才进入联盟，按选秀年初始化会提前注入 Elo；首次出场触发保证初始化即进入可观测状态。

### DM-006：不调用 Advanced Box Score

**选择**：Phase 1 不调用 `BoxScoreAdvancedV3`，仅抓取传统 box score

**理由**：Elo 公式只用传统 box score 的字段（GameScore + plusMinusPoints/minutes），advanced 表中的 netRating 等不参与计算。省下大量 API 调用，并避免 1996-97 前空响应处理。未来 BPM 模型需要时可通过传统 box score 自行计算，也可后补抓取。


### DM-007：赛季起点调整为 1976-77

**选择**：项目从 1976-77 赛季开始，不再从 1954-55 开始

**理由**：Data2 的比赛场次从 1954-55 起已接近完整，但球员级有效出场分钟直到 1976-77 才稳定。1954-55 到 1960 年代的大量比赛虽然存在 box score 行，却缺少足够的有效分钟记录，平均每场只有约 3-14 名球员可计算。1976-77 起平均每场约 20 名球员有分钟记录，且 98% 以上的比赛可用，同时是 ABA-NBA 合并后的第一个赛季，适合作为 Elo 起点。
### DM-005：Raw 与结果分库

**选择**：Raw 存 `nba_raw.db`，Elo 结果存 `nba_elo.db`，后者可由前者重建

**理由**：Raw 绝不修改的原则通过文件边界强制执行；结果层可随时删除重建，不影响数据资产。

### DM-008：比赛可用性 gate 与 +/- 缺失

**选择**：以双方分钟和与 `numMinutes` ±10 分钟作为可用 gate；1996 前 `plusMinusPoints` 全为 0 时标记 on-court 信号不可用，不阻塞 Elo

**未选**：仅按上场人数过滤、严格分钟相等、对早期 +/- 补值

**理由**：1984-85 约 243 场球员分钟加总严重不足，人数过滤无法发现；1996-2023 球员分钟取整导致正常偏差 -9 到 -3，±10 分钟可保留 58,927 场。正负值只进入 on-court 项，缺失时 `perf_i` 退化为 GameScore 项，不影响公式运行。

**实测结果（2026-08-14 基线）**：`scripts/build_clean_data.py` 生成 58,927 场可用、382 场豁免。豁免中 305 场为分钟超差（其中 1984-85 的 243 场双方均有出场行但分钟严重不足），77 场为一方或双方无 `played=1` 行；其余零星分布在 1976-1993、1996、2004、2024。

### DM-009：clean_data 派生层

**选择**：新增 `clean_data/` 目录与 `scripts/build_clean_data.py` 脚本，输出 `games_clean.csv`、`boxscores_clean.csv`、`known_exemptions.csv`、`players_clean.csv`、`validation_report.md`

**未选**：在 Elo 加载时重复清洗、把豁免比赛从 `processed_data` 物理删除

**理由**：清洗规则（D-015 分钟缩放、D-016 可用性 gate、D-017 +/- 标记）只实现一次且可复跑；原始 `processed_data` 保持不动，豁免场次保留原始行便于复核。
