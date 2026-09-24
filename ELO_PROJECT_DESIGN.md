# Dynamic Player Elo — 设计文档 v0.1

---

## 1. 项目概述

### 1.1 目标

建立一个 Dynamic Player Elo 评分系统，以球员**个人表现为主**，为每位 NBA 球员生成一条动态、可解释的生涯能力曲线（D-035 更新：团队胜负辅助修正经调优后取 θ=0；on-court 信号仅保留在 α=0.9 现代对比轨道）。

### 1.2 设计原则

- **个人优先**：Elo 更新主要由球员自身表现驱动，不被队友拖累、不因队友受益
- **全自动**：不依赖人工标注、外部评分或主观判断
- **可追溯**：每个参数和设计选择都有记录和理由
- **可扩展**：Elo 作为数据基础设施的一部分，未来可支撑更多模型

---

## 2. 数据架构

### 2.1 数据源

**唯一数据源：NBA Official Stats API（通过 `nba_api` Python 库）**

### 2.2 数据覆盖

**起始赛季：1976-77（NBA/ABA 合并后的第一个完整赛季）**

选择理由：球员级 box score 从 1976-77 起才稳定达到平均每场约 20 名球员有有效分钟记录，且 98% 以上的常规赛都包含可计算的球员出场数据。1976-77 也是 ABA-NBA 合并后的第一个赛季，联盟进入 22 队、82 场的现代结构。更早赛季虽然比赛清单基本齐全，但球员有效出场分钟覆盖不足，无法可靠支撑逐场 z-score、球队平均 Elo 和零和更新。

**截止赛季：2025-26；2023-24、2024-25 存在少量比赛缺失，允许进入完整性豁免清单，不阻断训练和 holdout。**

### 2.3 使用的数据

#### 核心必需（所有赛季可用）

| 数据 | API 端点 | 关键字段 | 用途 |
|------|---------|---------|------|
| 比赛列表 | `LeagueGameLog` | GAME_ID, GAME_DATE, SEASON_ID, TEAM_ID, MATCHUP, WL | 比赛集合、比赛结果 |
| 球员传统 Box Score | `BoxScoreTraditionalV3` | personId, gameId, teamId, minutes, points, fieldGoalsMade/Attempted, freeThrowsMade/Attempted, reboundsOffensive/Defensive, assists, steals, blocks, turnovers, foulsPersonal, plusMinusPoints | 计算 GameScore、获取在场 +/- |

| 球员静态信息 | `CommonAllPlayers` | personId, displayName | ID → 姓名映射 |
| 球队静态信息 | `teams`（static） | teamId, fullName, abbreviation | ID → 名称映射 |

#### 辅助数据

| 数据 | API 端点 | 用途 |
|------|---------|------|
| 选秀历史 | `DraftHistory` | 新秀初始 Elo |
| 球员进阶 Box Score | `BoxScoreAdvancedV3`（netRating 等，1996-97 起可用） | 交叉验证参考；Elo 核心不依赖 |

### 2.4 数据质量与清洗

| 问题 | 处理方案 |
|------|---------|
| `minutes` 格式为 `"35:56"` | 解析为浮点数分钟 |
| `plusMinusPoints` 早期赛季全为 0（非真实追踪） | 1996 前标记 on-court 不可用，`α=1` 只使用 GameScore（D-017/D-018） |
| `BoxScoreAdvancedV3` 1996-97 前不可用 | Elo 只依赖传统 box score，进阶数据是加分项 |
| 加时赛分钟 | 使用 `player_minutes / team_minutes`（API 直接提供） |
| 球员分钟取整/秒级混合（processed_data 中 1996-2023 多为整数，2024-25 起含秒级小数） | 计算 `minutes_share` 前按队等比缩放到 `team_minutes`，保证每队占比和为 1 |
| 单场 GameScore 标准差极小 | σ 下限设为 1.0 |
| 低分钟球员 on-court rate 噪声 | 贝叶斯收缩 `rate × m/(m+5)` |

### 2.5 数据库

以 **Game** 为中心。Raw Layer 表：

- `games`：game_id, date, season, home_team_id, away_team_id, home_score, away_score
- `players`：player_id, name
- `teams`：team_id, name, abbreviation
- `boxscores`：player_id, game_id, team_id, minutes, pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf, plus_minus
- `advanced_boxscores`：player_id, game_id, team_id, off_rtg, def_rtg, net_rtg（1996-97+）

**原则**：Raw 数据绝不修改；清洗只派生 `clean_data/`，Elo 只消费 `clean_data/`（D-019）。

---

## 3. Dynamic Player Elo 公式

### 3.1 核心理念

**个人表现驱动。**

传统 Elo 只看比赛结果，把团队成就平均分给个人。我们反转这一逻辑：球员的 Elo 更新由他在这场比赛中的**个人表现**驱动，而不是由球队胜负直接分配。

**D-035 更新**：θ 项经 E-022/E-024/E-026 三轮扫描后取 0（团队胜负修正对个人评分无预测增益）；on-court 信号以 α=0.9 现代对比轨道保留，全时代正典轨道为纯 GameScore。

### 3.2 完整公式

#### Step 1：个人表现标准分

```
GameScore = PTS + 0.4×FGM - 0.7×FGA - 0.4×(FTA-FTM)
          + 0.7×OREB + 0.3×DREB + 0.7×AST + STL + 0.7×BLK
          - 0.4×PF - TOV

on_court_rate = plusMinusPoints / minutes
on_court_rate_shrunk = on_court_rate × (minutes / (minutes + 5))

perf_i = α × zscore(GameScore_i) + (1 - α) × zscore(on_court_rate_shrunk_i)
```

| 符号 | 含义 |
|------|------|
| `α` | GameScore 与 on-court +/- 的权重，[0, 1] |
| `zscore(x)` | 在本场所有上场球员中做标准化：(x - μ_game) / max(σ_game, 1.0) |
| 收缩 `m/(m+5)` | 防止低分钟球员的 on-court rate 噪声过大 |

**为什么两个信号都要？**

- **GameScore**：反映球员在 box score 上做了什么。优点是不被队友拖累。缺点是漏掉防守站位、掩护质量等隐形贡献。
- **On-court +/-**：反映球员在场时球队净胜分。优点是能捕捉隐形贡献。缺点是会被队友水平污染。
- 两者互补。α 由交叉验证选择。

#### Step 2：Elo 更新

```
minutes_share = player_minutes / team_minutes

perf_used_i = perf_i - (R_i_old - anchor) / surprise_scale

ΔR_i = K × minutes_share × perf_used_i

R_i_new = R_i_old + ΔR_i
```

`player_minutes` 在进入公式前按队等比缩放到 `team_minutes`。processed_data 中 1996-2023 的球员分钟多为取整整数，加总常为 231-237 而非 240；2024-25 起含秒级小数。缩放后每队 `minutes_share` 之和为 1，加时赛仍按 240+25k 处理。

| 符号 | 含义 |
|------|------|
| `K` | 基础灵敏度系数 |
| `minutes_share` | 该球员占全队总时间比例。加时赛 team_minutes > 240，占比自然缩小 |
| `perf_i` | 个人表现标准分（Step 2），可正可负 |
| `perf_used_i` | 进入更新的表现分 = `perf_i - (R_i_old - anchor) / surprise_scale`，对高分球员形成回归惩罚 |

**评分回归项（surprise）**：`surprise_scale` 控制高分球员维持高分的难度，默认锚点为 `game`（当场所有上场球员赛前评分均值）。评分高于锚点时，相同表现获得更少的加分、更差的比赛扣更多分；评分低于锚点时相反。该回归在比赛日发生，不修改零和性质，也不需要赛季末统一修正。15 年窗口实验见 D-023。

#### 场景举例

| 场景 | perf_i | 总更新 | 直觉 |
|------|--------|--------|------|
| 50 分的顶级表现 | +2.0 | 大涨 | 打出远超同场球员的表现 |
| 4 分的低效表现 | -1.5 | 大跌 | 表现远低于同场球员 |
| 20 分的正常表现 | +0.2 | 小涨 | 略高于同场平均水平 |

### 3.3 特殊规则

#### 新秀初始 Elo：统一起点（原设计为选秀顺位线性递减）

**D-032 更新（R3e 后）**：最初设计意图是按顺位区分起点。R3a-R3e 的实验表明，在当前开局 `rookie_boost=6、rookie_tau=60` 下，最优 draft 先验退化到 floor>pick1 的统一起点（`floor=1450` 时所有新秀初始均为 1450），`pick_slope` 基本无效。因此 draft 先验简化为单一 `R_rookie_start`（R3g 后正式锁定 1425），`pick1_rating/pick_slope/rookie_floor` 不再独立调参。该结论以开局 boost 为前提；若后续 boost 显著下调，需要重新检验顺位先验。

**以下公式与表格为历史设计（D-032 已取代；当前实现为统一起点 `R_rookie_start=1425`）：**


```
R_init = R_pick1 - (pick - 1) × R_slope
下限 = R_rookie_floor
```

| 顺位 | 初始 Elo |
|------|---------|
| Pick 1 | 1580 |
| Pick 10 | 1535 |
| Pick 20 | 1485 |
| Pick 30 | 1435 |
| Pick 40 | 1385 |
| Pick 47+ | 1350 |
| 落选秀 | 1350 |

顺位是进入联盟前最客观的先验信号。好球员会爬升，初始值只是起点。线性递减比聚类保留更多信息。新秀整体中位数远低于联盟平均 1500，符合现实中新秀普遍为负贡献球员的事实。`R_pick1`、`R_slope`、`R_rookie_floor` 三个参数进搜索空间。
（以上为 D-032 前的历史文本，不再作为当前设计。）

#### 新秀 K 值：前 N 场加速收敛

```
K_rookie(games_played) = K_base × (1 + (boost - 1) × e^{-games_played / τ})
```
| 参数 | 含义 | 搜索范围 |
|------|------|---------|
| `boost` | 放大倍数 | 4-6（R3b 外扩；正式锁定 6） |
| `τ` | 衰减速度 | 35-60 场（R3c 外扩；正式锁定 60） |

跨赛季不重置——第一年 82 场后 K 已回到基础值。K-boost 的判定以球员首次出现在 `clean_data` 为准（D-022），不依赖 `draftYear`。

#### 1976-77 赛季初始值：双向 Elo

选定最优参数后，跑三遍：

1. **第一遍（正向预热）**：1976-77→2025-26，所有人使用统一 `rookie_start`（D-033），获得粗糙但自洽的轨迹
2. **第二遍（反向传播）**：2025-26→1976-77，以终值为锚点反向推导，产生高质量 1976-77 初始值
3. **第三遍（最终正向）**：1976-77→2025-26，使用第二遍的初始值，产出最终 Elo

反向跑使用与正向完全相同的公式，不调参。目标是产生好的初始值，而非可解释的反向轨迹。

执行状态（E-031）：最终运行直接采用双向方案（D-007 三遍，`scripts/run_final_elo.py`）；统一起点对照未单独做 A/B——按 D-012 的收敛性判断，初始方案主要影响早期轨迹而非长期预测，不作为 Step 5 的必要前置；若需要可并入 R7 敏感性。

#### 跨联盟球员处理（BAA/NBL/ABA → NBA）

1976 年 ABA-NBA 合并时，已在 ABA 效力过的球员首次在 NBA 出场时，统一按新秀处理，以统一 `R_rookie_start=1425` 初始化（D-032/D-033）。是否加速也按首次出现在数据窗口判定（D-022）。由于项目从 1976-77 开始，更早的 BAA/NBL 历史不进入数据窗口。

理由：(1) 无法跨联盟比较比赛强度，将 ABA 数据混入会引入未知偏差；(2) 新秀 K-boost（当前锁定 6×，τ=60）确保成熟球员快速爬升到真实水平；(3) 双向初始化的反向传播会将球员终值部分回灌到初始值，进一步缩小低估值。典型受益者：Dr. J、Moses Malone（1976 年由 ABA 进入 NBA，均按 1425 初始化，靠 K-boost 快速收敛到真实水平）。

#### 联盟均值漂移修正

**硬约束：每场比赛的更新严格零和。** 同一场比赛全部上场球员的 ΔR 计算完毕后，统一减去本场均值：

```
ΔR_i_adj = ΔR_i - mean(ΔR_all_players_in_game)
```

零和是底线：无论 perf_i 与上场时间如何相关（首发表现系统性优于替补会导致加权 z-score 和为正），每场比赛净注入 Elo 必须为 0。

**不做赛季末修正**：surprise 回归（D-023）已在比赛日提供负反馈并把标度稳定在 SD≈170、均值≈1,585，赛季末统一修正已被否决。早期 D-009 的轻量锚定方案不再采用。

#### 季后赛

- 当前实现：季后赛与常规赛进入同一 Elo 更新，默认 `playoff_k_multiplier=1.0`（同权重，D-025）
- 最后阶段：通过 `playoff_k_multiplier ∈ [0.5, 1.5]` 观察季后赛影响，不采用独立 track / 回灌
- `track == "playoffs"` 的 K 在 `k_effective` 基础上乘该系数；Play-In 归 playoffs track

Play-In（2021 赛季起）：NBA Stats API 将其归类为 Playoffs SeasonType，与首轮/次轮等一起抓取，归入季后赛 track 处理。

### 3.4 调参方案

#### 优化目标

**未来个人表现的预测能力（MSE）**，非比赛胜负 log loss。

选择理由：Elo 代表球员水平，核心测试是能否预测该球员未来的 perf_i。比赛胜负 log loss 会驱使 θ 偏大，与"表现优先"逻辑冲突。

#### 调参流程

```
Step 1: 正向预热（GameScore-only 段）
  1976-77 → 1977-78（2 赛季 burn-in）
  1976-77 → 1995-96：α=1，perf_i = zscore(GameScore)
  历史基线（D-021 回归用）：K=20, θ=0.3
  早期推荐运行参数（D-023/E-002 时代，已由 R2c 后参数取代；最终以 3.5 为准）：K=10, surprise_scale=400, surprise_anchor=game

Step 2: 第二次 burn-in（on-court 过渡）
  1996-97 → 1997-98
  启用 on-court 信号，让评分从 GameScore-only 重新收敛

Step 3: 参数搜索
  评估窗口（被预测赛季）：1998-99 → 2022-23（边界为前一个赛季末：1997-98 → 2021-22，见 D-027）
  方法：时间序列交叉验证（逐年滚动训练 → 预测下一年）
  损失：MSE(perf_i_10game_window, predicted_from_Elo)，定义见 D-027
  搜索空间（D-030，全部进入调优；R3 后 rookie/draft 实际范围已外扩，且 draft 三参数合并为 rookie_start，见 D-032）：K∈[8,60] 与 surprise_scale∈[240,500] 联合搜索（R2c 后实际有效区间约 240-300，两者共同决定标度），θ∈[0,1]、α_modern∈[0,1]、rookie_boost∈[4,6]、rookie_tau∈[35,60]、rookie_start∈[1300,1500]（R3g 后正式锁定 1425）
  anchor 固定 game（D-023 语义选择，不搜索）；playoff_k 按 D-025 留到最后单独实验
  搜索顺序（分阶段粗搜 → 局部复核，保证每个参数都进入搜索且交互项被复查）：
    R1: K × scale 粗搜（E-008，16 组已完成）
    R2: K/scale 外扩细化（K {20,25,30,35} × scale {270,285,300,315,330}，step=15 共 5 点，`grids/refine_k_scale.json`，已完成，见 E-010）；scale=350 已在 R1 测过且超标度约束，不重复
    R2b: K=40 边界检查（K {40} × scale {270,285,300}，已完成，见 E-011）；K=40 仍为最优边界，随后由 R2c 外扩至 60
    R2c: K 45-60 对角网格（8 组，已完成，见 E-012）；最高分 2,200-2,500 维持软约束；暂定 K*=45、scale*=285
    R3: 固定当前 K/scale 最优，rookie/draft 参数联合粗搜（rookie_boost × rookie_tau × pick1_rating，再展开 pick_slope、rookie_floor 局部点）
    R3a: rookie_boost × rookie_tau × pick1_rating 三参数初筛（27 组，已完成，见 E-013）；MSE 最优 boost=4、tau=35、pick1=1550，位于网格边界
    R3b: 外扩 boost {4,5}、tau {35,45}、pick1 {1520,1550,1580} 确认边界后，再展开 pick_slope、rookie_floor 局部点
    R3b 实际执行：先 boost/tau 外扩 9 组（见 E-014），最优 boost=6、tau=55、pick1=1550，仍位于网格边界；pick1 扫描待跑
    R3c 实际计划：用 boost=6、tau=55 扫 pick1 {1490,1520,1550,1580}，再展开 pick_slope、rookie_floor
    R3c 已完成（E-015）：tau {50,55,60} × pick1 {1490,1520,1550,1580}，最优 tau=60、pick1=1490，仍顶低边界
    R3d 实际计划：固定 tau=60、pick1=1490，调 pick_slope {4,5,6} × rookie_floor {1320,1350,1380}（9 组）
    R3d 已完成（E-016）：实际跑 pick_slope {3,4,5,6} × rookie_floor {1300,1325,1350,1375,1400}，最优 slope=6、floor=1400，仍顶边界
    R3e 已完成（E-017）：实际跑 pick_slope {6,7} × rookie_floor {1400,1425,1450} × pick1_rating {1430,1460,1490}，最优 slope=6/7、floor=1450、pick1=1430（MSE 0.19657）；slope 基本收敛，floor/pick1 仍顶边界
    R3f 已完成（E-019）：固定 boost=6、tau=60，扫 rookie_start {1300,1325,1350,1375,1400,1425,1450,1475}；MSE 对起点绝对水平不敏感，按 top 2,200-2,500 标度约束初步建议 rookie_start=1425
    R3g 已完成（E-020）：用当前最终参数计算 1976-2025 每年退出球员的分钟加权分数，确认相对流动稳定；正式锁定 rookie_start=1425（D-029 top 软约束带内最低合理起点）
    R4: θ、α_modern 网格（1996 前 α 固定为 1；R4a/R4b 已完成，见 E-022/E-023）
    R4 目标说明：R4a/R4b 固定 CV target α_ref=1.0（见 D-027a）；E-025 起敏感性实验改为 alpha_ref=alpha_modern（见 D-035 双轨说明）
    R4 网格（D-034）：先单参数粗筛，后局部 2D 细筛；θ 从 0.1 起，不含 0（D-034；实测两种 target 口径下 θ=0 均最优，见 E-022/E-024/E-026，最终保留 0）
      R4a 已完成：实际扩展为 θ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9} 共 10 组（E-022）
      R4b 已完成：α_modern ∈ {0, 0.5, 0.75, 0.9, 1.0} 共 5 组（E-023）
      R4c：原计划 3×3 细筛，实际由 E-024（α×θ 补扫 8 组）、E-025（α_ref 敏感性 8 组）、E-026（alpha_ref=0.9 下 θ 复核 5 组）取代
      R4 评估范围：沿用现有边界 1997-2021（预测 1998-2022），该窗口已全部落在 on-court 时代；引擎仍从 1976 顺序运行，
      不因“只用现代评估”而跳过早期数据（Elo 需要完整历史；早期只固定 α=1，不参与 α_modern 的评估变化）
     R6: 回到 K/scale 与 rookie/draft 最优点做局部复核，确认跨阶段交互（E-027 已完成第一阶段：重建 θ=0 checkpoint，并跑 on-court 口径 {raw, team_relative, on_off} × K {40,45,50} × scale {270,285,300} 共 27 组；三种口径对个体预测无差异，保留 raw；K/scale 取舍：正典轨 45/285、0.9 轨 45/318，见 R6b/c 与 D-036；E-030 在共享 checkpoint 上复核两轨 θ∈{0,0.1,0.3}，θ=0 均最优，锁定不变）
    R6b/c: 0.9 轨 K/scale 标度校准（E-028，已完成）：共享 θ=0 checkpoint 上跑 K {45,50,55,60} × scale {300,315,330} 粗网格 10 组 + K45 × scale {316,318,320} 细网格 3 组；K45/scale318 末季 SD=175.7，对齐 α=1.0 轨 175.5，用户已确认正式锁定 0.9 轨为 K=45、scale=318
     R7: 敏感性检查（R6 后、Step 5 前）：用最终参数集比较赛季末采样与赛季中窗口采样（如第 20/41/60 场后）的 MSE/r/discrimination 与候选排序，确认协议稳健后再进入 Hold-out ✅（E-032/E-032b，已完成）：4 候选 × 4 采样点共 16 行，见 `results/cv_results_r7_v2.csv`（赛季中目标修正为采样点后的接下来 10 场）；修正后赛季中短视窗的 MSE/r 优于赛季末跨季目标，但候选排序仍稳健；正式 Hold-out 仍按 D-027 的赛季末跨季协议执行
  输出：正典轨（K=45、scale=285、θ=0、α=1.0、boost=6、tau=60、rookie_start=1425、on_court_mode=raw、playoff_k=1）；0.9 轨（K=45、scale=318、θ=0、α=0.9，其余同正典轨）

Step 4: 双向初始化 ✅（E-031，已完成）
  使用最优参数跑三遍
  1996 前仍固定 α=1，1996 后用 α_modern
  双轨执行（D-036）：α=1.0 正典轨走完整三遍双向；α=0.9 现代轨不做独立反向传播，从正典轨双向后的 1995 年末状态续跑 1996-2025
  实现：`scripts/run_final_elo.py`（canonical / modern）；输出见 `results/final/` 与 E-031

Step 5: Hold-out 验证 ✅（E-033，已完成）
  2023-24 → 2025-26（内部边界 2022-2024）评估 MSE，L1-L4 baseline 对比：
    L1: 纯随机预测（边界赛季内置换 target）
    L2: 纯团队 Elo（K=20，只读胜负，不读球员数据）
    L3: 朴素统一参数（K=10、scale=400、θ=0.3、rookie_boost=3、rookie_tau=25、rookie_start=1500）
    L4: 最优参数（final 双轨 updates，正典 285/1.0，现代 318/0.9）
  对 L1-L4 均输出 MSE、r、rho、按球员/赛季 cluster bootstrap 的 95% CI 与 p 值（D-028）
  实现：`scripts/run_holdout.py`；输出 `results/holdout_results.csv`、`results/holdout_pairs.csv.gz`
  结果（E-033）：正典轨 L4 MSE=0.1719、r=0.8616；0.9 轨 L4 MSE=0.1419、r=0.8595；两轨 3/3 边界 r>0，player/season bootstrap CI 均 >0 且 p=0；L3 朴素参数已有 r≈0.73 但 L4 的 MSE 相对改善约 34-36%，L2/L1 无个体预测力
```

#### 参数职责

| 参数 | 优化目标 | 搜索方式 |
|------|---------|---------|
| `K`, `α_modern` | perf_i 预测 MSE（1996 前 α 固定为 1） | 网格搜索 |
| `rookie_boost`, `rookie_tau` | perf_i 预测 MSE + 新秀收敛速度 | 联合网格搜索（D-030 R3） |
| `R_rookie_start` | 新秀统一起点（影响末季标度；对 MSE 不敏感） | R3f/R3g 已完成，正式锁定 1425（E-019/E-020） |
| `θ` | perf_i 预测 MSE（主）+ 定性验证（辅） | 几个固定值各跑一遍 |
| `playoff_k` | 季后赛影响定性观察 | 最后阶段单独实验（D-025） |

#### 3.4.1 标度稳定性实验（1976-77 → 1990-91，15 年窗口）

用 `draft` 初始化（D-022）+ surprise 回归在 14,950 场常规赛上跑过 7 组对照。无 surprise 时联盟 SD 从 151 膨胀到 762、最高分 5,923；加入 `perf_used = perf - (R - anchor) / scale` 后标度显著收敛：

| 组合 | SD（首季/末季） | 分钟加权均值（首季/末季） | 最高分（末季） | Moses（末季） | Kareem（末季） |
|------|----------------|------------------------|----------------|--------------|----------------|
| 无 surprise（K=20） | 151 / 762 | 1,509 / 1,836 | 5,923 | - | - |
| K=10, scale=400, fixed | 108 / 174 | 1,495 / 1,585 | 2,340 | 1,889 | 1,761 |
| K=10, scale=400, game | 106 / 179 | 1,493 / 1,589 | 2,360 | 1,902 | 1,773 |
| K=10, scale=400, league | 106 / 166 | 1,493 / 1,576 | 2,314 | 1,871 | 1,747 |
| K=15, scale=400, fixed | 125 / 191 | 1,505 / 1,610 | 2,416 | 1,862 | 1,709 |
| K=10, scale=300, fixed | 102 / 148 | 1,493 / 1,570 | 2,172 | 1,775 | 1,668 |
| K=10, scale=500, fixed | 112 / 196 | 1,496 / 1,596 | 2,476 | 1,995 | 1,855 |
| K=10, scale=1,500, fixed | 125 / 300 | 1,499 / 1,635 | 3,073 | 2,599 | 2,497 |

结论：`anchor` 三种取值数值差异 <20 分，不会改变标度结论；推荐 `game`，因为它把“超常发挥”定义为相对当场球员池水平，更贴合“这场打得如何”的叙事。真正的控制旋钮是 `scale` 和 `K`。scale=400 能把末季最高分压在 2,300-2,400、明星末季评分落在 1,700-1,900，既防止膨胀又没有把头部球员压到无意义。所有组合每场 `|ΣΔR| < 1.2e-14`，零和约束未受影响。完整实验代码见 `scripts/experiments_surprise.py`。

额外跑了 1976-77 → 1995-96 的 20 年验证（K=10, scale=400, fixed 锚点）：SD 在 1985 年后稳定于 165-175，分钟加权均值稳定于 1,582-1,586，最高分稳定于 2,218-2,357，确认标度收敛而非仅仅减速。game 锚点在 15 年窗口与 fixed 差异 <20 分，稳定性结论同样适用。退役球员评分保持退役前最后值，符合“无比赛不掉分”的叙事。

#### 3.4.2 粗搜协议（K × surprise_scale）

第一轮粗搜为 4×4=16 组：

- `K ∈ {8, 10, 15, 20}`
- `surprise_scale ∈ {300, 350, 400, 500}`
- 其余固定：θ=0.3、α_modern=1、init=draft、playoff_k=1

筛选指标按优先级：

1. 主指标：MSE（D-027），预测下赛季前 10 场常规赛 `perf_i` 均值
2. 辅助：Pearson r / Spearman rho（D-028），防止误差下降由少数样本主导
3. 区分度（D-031）：`discrimination = pred_sd / target_sd`，接近 0 表示评分压缩、球员拉不开差距，直接淘汰
4. 标度约束（取末季）：联盟 SD 165-185 为硬约束，超标即淘汰（即使 MSE 更低）；分钟加权均值在 1,500 附近且无持续漂移；最高分 2,200-2,500 为软约束（E-012 起用户确认，略低于 2,200 仍合规，如 K=40/scale=270 的 2,169）
5. 定性验证：从 MSE 最优、区分度与标度合规的 3-5 组看明星轨迹与赛季 top 名单是否合理

细化（R2，已完成，见 E-010；R2b 补测 K=40，见 E-011；R2c 对角网格 K 45-60，见 E-012）：围绕第一轮最优且合规的点，取 K {20,25,30,35} × scale {270,285,300,315,330}（20 组）后补 K=40 × scale {270,285,300}、K 45-60 对角 8 组；正典轨暂定 K*=45、scale*=285（0.9 轨 scale 由 E-028 校准为 318）。

后续轮次按 D-030 的 R2-R6 执行：K/scale 外扩细化后依次搜索 rookie/draft 参数、θ、α_modern，最后回到 K/scale 与 rookie/draft 做局部交互复核；每组都输出并检查区分度（D-031）。

每组输出（`cv_results.csv` 一行）：`mse`、`r`、`baseline_mse`、`pred_sd`、`target_sd`、`discrimination`、`n_players`、`n_boundaries`、`last_season`、`last_sd`、`last_minutes_wmean`、`last_top_rating`、全部参数列、`engine_seconds`。

### 3.5 参数汇总

| 参数 | 含义 | 搜索范围 / 当前取值 |
|------|------|---------|
| `K` | 基础灵敏度 | 45（两轨共用；R2c 锁定、E-028 复核） |
| `K_rookie_boost` | 新秀 K 放大倍数 | 6（R3b 后锁定） |
| `τ` | 新秀 K 衰减速度（场数） | 60（R3c 后锁定） |
| `α` | 1996 前固定为 1（GameScore-only）；1996 后 `α_modern` 为 GameScore vs on-court 权重 | 双轨：1.0（全时代正典）/ 0.9（现代对比），θ=0（D-035） |
| `θ` | 团队信号权重 | 0（R4 三轮扫描最优，E-022/E-024/E-026） |
| `R_rookie_start` | 新秀统一初始 Elo | 1425（R3g 后正式锁定） |
| `surprise_scale` | 评分回归强度：perf 中扣除评分相对锚点的偏离 | 285（α=1.0 正典轨，R2c 后锁定）/ 318（α=0.9 现代轨，E-028 标度校准后锁定） |
| `surprise_anchor` | 回归锚点：fixed=1500 / game=当场球员均值 / league=已追踪球员均值 | game（语义选择，D-023） |
| `on_court_mode` | on-court 信号口径 | raw（E-027 三口径 {raw, team_relative, on_off} 对比后锁定） |
| `alpha_ref` | CV 目标标签混合权重（非引擎更新参数） | 正典轨 1.0；0.9 轨 = alpha_modern=0.9（D-035/E-025） |
| `playoff_k` | 季后赛 K 乘数 | 1.0（同权重；最终单独实验 D-025，范围 0.5-1.5） |
全部可解释参数均进入搜索（D-030）；`on_court_mode` 三口径由 R6/E-027 对比后锁定 raw，`alpha_ref` 作为 CV 目标标签参数随敏感性实验单独设定；`anchor` 固定 `game`，`playoff_k` 留到最终实验（D-025）。

---

## 4. 实施路线图

### Phase 1：数据库搭建

| 步骤 | 内容 |
|------|------|
| 1.1 | 验证所有 API 端点的字段完整性和数据可用性 ✅（已完成） |
| 1.2 | 设计并创建 SQLite Raw Layer |
| 1.3 | 编写数据抓取脚本（请求队列 + 重试 + 缓存） |
| 1.4 | 抓取 1976-77 → 2025-26 全部赛季数据 |
| 1.5 | 数据质量检查 |

### Phase 1.5：数据清洗（已完成）

以 `processed_data/` 为输入，通过 `scripts/build_clean_data.py` 生成 `clean_data/`：

| 步骤 | 内容 |
|------|------|
| 1.5.1 | 按 D-016 运行比赛可用性 gate，生成 `games_clean.csv` 与 `known_exemptions.csv` ✅ |
| 1.5.2 | 按 D-015 为每队等比缩放球员分钟，输出 `boxscores_clean.csv`（仅 `played=1`，含 `minutes_share`、`usable_game`） ✅ |
| 1.5.3 | 按 D-017 标记 `plus_minus_available`（1996-97 起启用，之前不可用） ✅ |
| 1.5.4 | 规范化 `players.csv` 选秀占位值（`-1`/`NaN` 统一为未选中，落选秀与占位值等价处理） ✅ |
| 1.5.5 | 生成 `validation_report.md`：引用完整性、比分一致性、可用场次统计 ✅ |

Elo 实现直接消费 `clean_data/`，不直接读 `processed_data/`。

### Phase 2：Elo 实现

| 步骤 | 内容 |
|------|------|
| 2.1 | 实现 Elo 更新核心函数（含新秀 K 衰减、每场零和更新、surprise 回归） ✅ |
| 2.2 | 正向预热（2 赛季 burn-in，默认参数） |
| 2.2 | 正向预热：首次全量运行已完成（D-023 推荐参数，α=1，见 E-002）；参数搜索后按最优参数重跑 |
| 2.3 | 参数搜索（时间序列交叉验证） |
| 2.4 | 双向初始化 + 最终正向运行 ✅（E-031，`scripts/run_final_elo.py`，输出 `results/final/`） |
| 2.5 | Hold-out 验证 + Baseline 对比 |
| 2.6 | 可视化：球员 Elo 曲线、参数敏感性 |

---

## 5. 决策日志

### D-001：数据源

**选择**：仅 `nba_api`

**未选**：Basketball Reference、Kaggle、手动收集

**理由**：官方数据，免费，Python 生态成熟，格式统一。多源增加清洗成本。

### D-002：数据库中心实体

**选择**：以 Game 为中心

**未选**：以 Player 为中心

**理由**：Game 是最稳定的主键。所有聚合（Player Season、Team Season）均可从 Game + Box Score 派生。

### D-003：起始年份

**选择**：1976-77

**未选**：1954-55、1970-71、1981-82

**理由**：比赛清单从 1954-55 起已经接近完整，但球员级有效出场分钟直到 1976-77 才稳定达到平均每场约 20 人、98% 以上比赛可用。更早赛季会出现大量只有 box score 外壳但缺少有效分钟记录的比赛，无法支撑逐场 Elo 更新。1976-77 是 ABA-NBA 合并后的第一个赛季，联盟结构进入 22 队、82 场的现代形态，适合作为正式起点。

### D-004：Elo 更新方向

**选择**：个人表现为主要驱动，团队胜负为辅助修正

**未选**：团队结果为主，个人表现为修正

**理由**：传统方案中 `(S-E)` 主导更新，预期接近时修正项消失。且团队结果是 5+ 人的产物，不应作为个人评价主信号。反转后 perf_i 是外生的即时信号，更稳定。

### D-005：优化目标

**选择**：未来 perf_i 预测 MSE

**未选**：比赛胜负 log loss

**理由**：Elo 代表球员水平，核心测试是预测该球员未来表现。log loss 会驱使 θ 偏大，与"表现优先"逻辑冲突。log loss 保留为验证报告中的辅助诊断。

### D-006：新秀初始 Elo（已被 D-032 部分取代）

**选择**：选秀顺位线性递减

**未选**：统一 1500、按范围聚类、confidence gate

**理由**：顺位是最客观的赛前先验。好球员会爬升。Confidence gate 制造恶性循环（受伤老将冻结、低顺位锁定）。

**更新（D-032）**：最初设计按顺位区分起点；R3a-R3e 调参后在开局 boost=6、τ=60 下，最优先验退化为统一起点（floor=1450 > pick1=1430，实际所有新秀初始 1450）。R3f 确认起点绝对水平对 MSE 不敏感后，draft 先验改为单一 `R_rookie_start`（R3g 后正式锁定 1425），顺位先验不再独立调参。

### D-007：双向初始化

**选择**：正向预热 → 反向 → 最终正向

**未选**：仅正向

**理由**：1976-77 赛季有大量已经成名的老将，按选秀顺位重新赋值会浪费已知信息。双向法零人工、自洽。

**状态更新（E-031）**：已完成——`src/elo/engine.py` 新增 `reverse=True`（按日期降序处理比赛），`scripts/run_final_elo.py` 跑正典轨三遍双向（1976-2025），输出 `results/final/final_elo_initial_state_1976_canonical.csv` 与最终 updates/diagnostics；0.9 现代轨按 D-036 从双向后的 1995 年末 checkpoint 续跑。

### D-008：on-court rate 贝叶斯收缩

**选择**：`rate × m/(m+5)`

**理由**：低分钟球员 raw rate 噪声极大。收缩因子不调参——功能是保护系统非优化预测。

### D-009：联盟均值漂移修正（已被 D-023 取代）

**选择（历史）**：原方案为每赛季 `R -= (R̄ - 1500) × 0.3`，第一版只观测不修正；实测显示零和更新无法阻止标度膨胀（15 年 SD 151→762），D-023 的 surprise 回归已在比赛日解决该问题，赛季末锚定方案废弃。

**理由**：漂移的方向和幅度本是经验问题，但赛季末一次性修正会让球员在无比赛时评分变化，破坏叙事；比赛日回归是更干净的形式。零和更新仍作为硬约束保留（D-014）。

### D-010：加时赛分钟

**选择**：`player_minutes / team_minutes`

**理由**：API 直接提供 team_minutes。双加时赛占比自然缩小，逻辑干净。

### D-011：单场 z-score 保护

**选择**：`σ_eff = max(σ_game, 1.0)`

**理由**：极端防守大战中 GameScore 方差极小，z-score 会爆炸。正常 σ 在 5-10，下限 1.0。

### D-012：调参流程

**选择**：默认参数 burn-in → 搜索最优参数 → 用最优参数跑双向

**理由**：打破"调参需要好初始值，好初始值需要参数"的死锁。α 几乎不依赖初始值，K 中度依赖但 4 赛季足够收敛。

### D-013：季后赛 K 值

**选择**：初期 K=1.0，稳定后实验

**理由**：季后赛对手单一、样本小。初期不引入额外复杂性。

### D-014：每场零和更新

**选择**：每场比赛所有上场球员的 ΔR 减去本场均值，净变化严格为 0

**理由**：这是系统底线而非可选项。否则 perf_i 与上场时间正相关（首发表现优于替补）会使每场加权 z-score 和为正，导致 Elo 系统性膨胀。零和更新从源头消除该问题，赛季末无需额外修正。

### D-015：球员分钟等比缩放

**选择**：计算 `minutes_share` 前，将每队球员分钟按 `team_minutes / sum(player_minutes)` 等比缩放

**未选**：直接使用原始球员分钟、严格要求每队加总等于 240/265/290/315/340/365

**理由**：processed_data 中 1996-2023 的球员分钟多取整到整数，每队加总常为 231-237，直接使用会导致 `minutes_share` 总和不等于 1；2024-25 起含秒级小数。等比缩放保证分钟口径统一，且不改变球员间相对上场时间。

### D-016：不可用比赛剔除

**选择**：可用比赛需同时满足：双方都有 `played=1` box score；每队 `sum(played=1 minutes)` 与 `team_game_log.numMinutes` 的偏差不超过 ±10 分钟。当前基线保留 58,927 场，剔除 382 场

**未选**：仅按上场人数过滤、严格分钟相等、对不可用比赛补值

**理由**：分钟校验能发现 1984-85 约 243 场球员分钟大量缺失的问题，而人数过滤看不出来；±10 分钟容纳 1996-2023 取整导致的正常偏差（-9 到 -3），又不会放过明显残缺场次。剔除场次保留在主表中并写入豁免清单，不物理删除。

### D-017：正负值缺失处理

**选择**：1996 前 `plusMinusPoints` 全为 0 时标记 on-court 信号不可用，不补值；1996 前 `perf_i = zscore(GameScore)`（等价 `α=1`），不阻塞 Elo 计算

**未选**：对早期 +/- 做估算/补值、因缺失而丢弃早期赛季

**理由**：正负值只进入 `(1-α) × zscore(on_court_rate_shrunk)` 一项，GameScore 仍是主信号。缺失时 on-court 项贡献为 0，Elo 公式仍然可算；为避免全局 `α` 对早期 `perf` 做无意义的缩放，1996 前固定 `α=1`。

### D-018：α 分段与第二次 burn-in

**选择**：1996 前固定 `α=1`、`perf_i = zscore(GameScore)`；1996-97 → 1997-98 作为第二次 burn-in；1998-99 起单独调 `α_modern`

**未选**：1978-79 → 2022-23 全窗口统一调 `α`

**理由**：1996 前没有真实 on-court 信号，统一调 `α` 只会让早期 `perf` 被无意义缩放，不是真正的权重学习；评分加入 on-court 信号后也需要 burn-in 才能进入可比较的状态。分段后，早期曲线由 GameScore 驱动，现代部分由 `α_modern` 决定，调参含义更干净。

### D-019：clean_data 作为 Elo 唯一输入

**选择**：新增 `clean_data/` 派生层，`scripts/build_clean_data.py` 一次性产出可用比赛、缩放分钟、+/- 可用标记、规范化选秀字段与校验报告；Elo 只读 `clean_data/`

**未选**：Elo 直接读 `processed_data/` 并在加载时重复清洗逻辑

**理由**：可用性 gate 与分钟缩放是全局预处理，分散到 Elo 加载步骤会导致同一规则多处实现、结果不可复现。`clean_data/` 使清洗一次完成、可审计，且 Elo 核心与数据质量检查解耦。

### D-020：结果输出模型（三类查询 + 导出格式）

**选择**：`nba_elo.db`（SQLite）作为唯一结果源，`elo_updates` 事件日志 + `elo_states` 状态表 + `elo_snapshot_season` 快照三表回答全部查询；`elo_updates` 增加审计字段 `game_score`、`z_game_score`、`on_court_rate`、`z_on_court_rate`、`perf_used`、`home`，保证“什么表现导致变化”可逐场追溯

- 任意球员生涯曲线：`elo_updates WHERE run_id=? AND player_id=? ORDER BY game_date, game_id`
- 任意日期排名：`elo_states` 取每人 `game_date <= 目标日` 的最后一行，按 `rating` 排序（默认常规赛 track）
- 单场变化归因：同一行同时给出 `rating_before/after`、`delta_adj`、`perf_i`、`perf_used`、`game_score`、`z_game_score`、`on_court_rate`、`z_on_court_rate`、`minutes_share`、`k_effective`

**导出格式**：`nba_elo.db` 为主；对外发布/可视化时按需导出 Parquet 或 CSV 到 `results/`（`elo_updates.parquet`、`elo_snapshot_season.parquet`），导出文件必须带 `run_id` 与生成时间，可随时由 db 重建。

**状态更新（D-036）**：最终 schema、表结构命名与查询 API 以 D-036 为准；本节 `elo_updates/elo_states/elo_snapshot_season` 三表语义保留，表名与派生视图由 D-036 统一。

**未选**：只存最终分数、每日全量快照为主表、以 CSV 为唯一结果源。

**理由**：三类查询天然对应事件日志、状态表、赛季快照三种粒度的联合查询；事件日志保留全部可审计信息，状态表避免每日快照爆炸。CSV 无法高效回答 as-of 排名且易被 Excel 修改，只作为导出格式。

### D-021：Elo 核心可实施性与测试策略

**选择**：Elo 核心现在即可实施；所有输入均来自 `clean_data/`（GameScore 字段、`minutes_share`、`plus_minus_available`、`usable_game`、`track`、选秀顺位），无外部依赖。实施后按三层测试：

1. **单元测试（合成数据）**：固定小样本手算校验 GameScore、z-score、预期胜率、更新公式；验证每场 ΔR 之和严格为 0、`minutes_share` 之和为 1、α=1 时忽略 on-court 项。
2. **不变量测试（全量数据）**：每个 `usable_game=1` 的 `played=1` 行恰好一条更新记录；每场 `delta_adj` 之和为 0；任意球员状态单调随比赛时间推进；无 NaN/无穷值。
3. **端到端回归（固定参数）**：两套基线都要记录首日/首季结果——历史基线 K=20、θ=0.3（无 surprise），推荐基线 K=10、surprise_scale=400、surprise_anchor=game、θ=0.3；用手算样例与 burn-in 收敛曲线做对照，后续改动不得改变基线输出。

**未选**：直接在全量数据上肉眼观察、只测公式不测状态流。

**理由**：核心逻辑的边界条件（零和、K-boost 衰减、z-score 下限）必须由合成样例精确定义；状态流和审计字段必须由全量不变量测试保证；固定参数回归把“公式正确”与“调参正确”解耦。

### D-022：新秀 K-boost 按首次出现在数据窗口判定

**选择**：所有球员在 `clean_data` 中首次出场时获得 K-boost（衰减规则不变），不再按 `draftYear` 判定新秀。

**未选**：按 `draftYear >= 1976` 判定、按选秀状态/年龄判定。

**理由**：`draftYear` 对 ABA 老将（如 Moses Malone）缺失，会被错误当成无加速的老将；而项目从 1976-77 起没有更早的 NBA 数据，所有球员对系统而言都是首次进入，初始先验本来就不确定。首次出场即加速能快速收敛到真实水平，且 5 年对照实验显示不影响长期终局排名。`draftYear` 仍用于初始分先验，但不再决定加速资格。（D-032 后 draft 先验合并为统一 `rookie_start`，`draftYear` 不再用于初始分先验。）

### D-023：评分回归项（surprise）抑制标度膨胀

**选择**：在 Step 3 更新前加入 `perf_used_i = perf_i - (R_i_old - anchor) / surprise_scale`，默认 `anchor = game`（当场所有上场球员赛前评分均值）、`surprise_scale = 400`，并在 `elo_updates` 中保留 `perf_used` 审计列。

**未选**：赛季末统一标准化/减均值修正；仅调低 K 而无回归项。

**理由**：(1) 赛季末一次性修正会让球员在没打比赛时评分下降，破坏叙事，且修正强度每赛季手动决定；(2) 15 年窗口显示仅调低 K（scale=1,500 近似）仍会把 SD 推到 300、最高分推到 3,073，回归项是必需而非可选；(3) fixed / game / league 三种锚点数值差异 <20 分，但 game 的语义最贴近叙事：把“超常发挥”定义为相对当场球员池水平，打弱队时碾压不加太多分、强强对话中 carry 更值钱；(4) 高分球员维持高分的难度来自比赛日本身：R=1,900、当场均值 1,700 时每场 perf 减 0.5，+1.0 的顶级比赛只记 +0.5，普通比赛记 -0.5，低分球员相反；(5) 回归发生在 raw delta 计算前，每场零和约束不变。


### D-024：赛季显示使用结束年份

**选择**：展示文件（`full_elo_top_by_season_*.csv`、`full_elo_diagnostics_*.csv`）以 `season_end = season_start + 1` 为主键，例如 1997-98 显示为 1998；同时保留 `season_start` 便于对照。内部 `clean_data`、引擎与 parquet 的 `season` 仍为起始年份，展示层转换（`scripts/build_results_summary.py`）。

**未选**：使用起始年份、直接使用 "1997-98" 字符串。

**理由**：NBA 口语和常规统计中 "98 赛季" 通常指 1997-98；起始年份标签会把 1998-99（乔丹缺席）显示成 "1998"，造成 "98 乔丹被马龙超越" 一类的展示误解。

### D-025：季后赛处理：同一 Elo 更新 + 最后调 playoff_k_multiplier

**选择**：季后赛与常规赛进入同一 Elo 更新（同一 rating、同一 K 衰减），当前默认 `playoff_k_multiplier = 1.0`，即季后赛与常规赛完全同权重；最后阶段再通过该系数（0.5-1.5）观察季后赛对评分的影响，不实现独立 track / 回灌。

**状态更新（E-034）**：已按三段式协议物化 playoff_k ∈ {0.5, 1.0, 1.5, 2.0} 的双轨变体，输出见 `results/playoff_variants/` 与 E-034；交付层以 track 形式入库，后续用于常规赛/季后赛贡献分析。

**未选**：独立 track + 温和回灌（早期 3.3 草案）、完全剔除季后赛。

**理由**：(1) 当前目标是常规赛个人表现叙事，季后赛样本小且对手池特殊，先不引入额外状态；(2) 单一系数是最小可解释旋钮，最后再实验即可；(3) 保留 track 列，后续若要改回独立 track 不丢数据。

### D-027：调参预测目标定义

**选择**：滚动 CV 的预测目标是球员下一赛季前 10 场常规赛 `perf_i` 均值（至少 3 场）；边界赛季为 1997-98 → 2021-22（内部年份 1997..2021），预测目标覆盖 1998-99 → 2022-23（内部 1998..2022），保证 2023-24 起的 Hold-out 与评估窗口不重叠；`predicted_from_Elo = (R_season_end - league_mean_season_end) / surprise_scale`；损失为逐球员边界（不按分钟加权）MSE。

**未选**：比赛胜负 log loss（D-005 已排除）；额外拟合线性校准参数；按分钟加权 MSE（保留为敏感性检查）。

**理由**：评分回归项本身就是模型对 `perf_i` 的预测假设，直接用同一公式做预测无需引入第二个可调模型；`league_mean` 锚点让预测与单场 z-score 的中心化口径一致；MSE 直接度量“Elo 对未来表现的预测能力”。该目标同时惩罚收敛太慢（K 太小）和预测过钝/评分膨胀（scale 不合适），因此 K 与 scale 必须联合搜索。

**补充（D-027a）**：CV target 中的 `perf_i` 固定使用 `α_ref=1.0`（纯 GameScore z）计算，R4a/R4b 共用同一标签；`alpha_modern` 只进入 Elo 更新，不改变评价目标。R1-R3 的 `alpha_modern=1.0` 与 `α_ref` 一致，现有结果可直接延续；若未来改用 `α_ref≠1`，需要重跑已完成的调参轮次以保持标签可比。

**更新（E-025/E-026）**：为观察 on-court 信号价值，敏感性实验改用 `alpha_ref=alpha_modern`（更新公式与评价标签同步混合，96 前无 on-court 时退回纯 GameScore）。最终双轨见 D-035：`α=1.0` 正典轨保持纯 GameScore 标签，`α=0.9` 现代轨使用同步混合标签。

**补充（D-027b：核心链路）**：`perf_i` 是外生标签，计算公式中不出现任何 Elo 项；Elo 是预测器，`pred = (R_season_end - league_mean_season_end) / scale`。完整链路为：过去 `perf_i` → 更新 Elo → 赛季末 Elo → 预测未来 `perf_i` → MSE。因此 MSE 度量的是“Elo 对未来表现标签的预测能力”，不是 Elo 对 Elo 的拟合。固定 `α_ref` 后，所有参数网格共享同一 target，MSE 差异只来自 Elo 本身。

### D-028：预测力统计检验

**选择**：`scripts/cv_stats.py` 对 CV 预测对（`results/cv_pairs_1997-2021.parquet`）做四类统计检验；Hold-out（Step 5）对 L1-L4 输出相同统计量：

1. Pearson / Spearman 相关（`pred` vs `target`）
2. MSE improvement over `pred=0`（即 baseline_mse - model_mse）
3. 按 `player_id` / `boundary_season` 的 cluster bootstrap（默认 2000 次）95% CI 与 p（H0：统计量 ≤ 0）
4. 按边界赛季正相关个数做 exact binomial 检验

**未选**：只报告单一 MSE；只用 naive t / 正态近似 p 而不处理球员、赛季内的聚类。

**理由**：MSE 绝对量缺少判断尺度，相关性和“每个边界赛季是否都正向”给出更直接的解释；同一球员多次进入样本、同一赛季球员共享环境，样本不独立，naive p 偏乐观，cluster bootstrap 分别按球员和赛季聚类更稳健。基线 CV 结果见 E-005：25/25 个边界 r>0（exact binomial p=2.98e-08），player/season 两种 bootstrap 的 95% CI 均在 0 之上，判定当前 Elo 对未来 `perf_i` 有显著预测力。p=0 表示 2000 次 bootstrap 未出现违反 H0 的样本，需要更精细 p 时可增加迭代次数。

### D-029：粗搜网格与筛选标准

**选择**：第一轮粗搜 4×4=16 组（`K ∈ {8,10,15,20}` × `surprise_scale ∈ {300,350,400,500}`，见 3.4.2）；筛选以 MSE 为主、r/Spearman 为辅，并以区分度（D-031）和末季标度作约束：SD 165-185 与均值稳定为硬约束，最高分 2,200-2,500 为软约束（E-012 起用户确认，略低于带下界仍合规）；围绕最优且合规的点做 R2 外扩细化（K {20,25,30,35} × scale {270,285,300,315,330}，20 组，已完成，见 E-010）。

**未选**：3×3 稀疏网格直接进入细调；只看 MSE 不看标度；第一轮直接跑 5×5 全网格。

**理由**：K 与 scale 共同决定标度，单看 MSE 可能选出膨胀参数；3×3 中 K 从 10 到 20 没有中间点，MSE 对 K 的响应可能非单调（太小收敛慢、太大过拟合单场噪声）；scale 只有 300/400/500 三个点，300 已接近压扁、500 已接近膨胀，无法定位拐点。4×4 每轮约 13 分钟、共约 3.5 小时，成本可控；5×5 全网格约 5.5 小时，信息增益不足以抵消。标度约束直接复用了 3.4.1 与 E-002 的标度结论。

### D-030：全部参数进入调优

**选择**：所有可解释引擎参数都进入搜索：K、surprise_scale、θ、α_modern、rookie_boost、rookie_tau、rookie_start（原 pick1_rating/pick_slope/rookie_floor 在 R3e 后合并为 rookie_start，见 D-032）；`on_court_mode` 三口径由 R6/E-027 对比后锁定 raw；`alpha_ref` 作为 CV 目标标签参数随敏感性实验单独设定（E-025 起）；playoff_k 按 D-025 留到最后单独实验，anchor 固定 game（语义选择）。采用分阶段搜索：R1 K×scale 粗搜（已完成）→ R2/R2b/R2c K/scale 外扩细化（已完成，见 E-010/E-011/E-012）→ R3 rookie/draft 参数联合粗搜（R3a-R3e 已完成，R3f 单参数确认与 R3g entry-exit 校准已完成，正式锁定 rookie_start=1425，见 E-019/E-020）→ R4 θ/α_modern 网格（协议见 D-034）→ R6 回到 K/scale 与 rookie/draft 最优点做局部复核。所有调参网格运行均以 `--end-season 2022`（2022-23，最后一个 CV 目标赛季）为上限，2023-24 起的 Hold-out 在 Step 5 前完全不读取，避免调参触碰测试数据。

**未选**：只调 K/scale 后把 rookie/draft 参数当固定启发式；一次全网格搜索所有参数。

**理由**：K 与新秀 K-boost/τ 相乘作用于同一更新，初始分参数决定新秀起点，surprise_scale 又决定起点到稳态的回归强度，参数间存在真实交互；只调 K/scale 会在错误的新秀动力学上收敛。分阶段比一次全网格便宜（R2 实测单组约 17-23 分钟），R6 局部复核捕获跨阶段交互。

### D-031：区分度诊断指标

**选择**：每组参数输出 `pred_sd`（预测值标准差）、`target_sd`（目标值标准差）、`discrimination = pred_sd / target_sd`；筛选时 discrimination 接近 0（预测分布明显压缩、球员拉不开差距）直接淘汰，并与 MSE、r、标度约束联合判断。

**未选**：只用 MSE/r 选参；要求 discrimination 恰好等于 1。

**理由**：MSE 度量校准/预测力而不是区分度。对完全压缩的预测（pred≡0），MSE = baseline_mse，因此“所有人预测成均值”不会被选为最优；但向均值收缩的预测（pred = c×target，c<1）可以在保持高 r 的同时降低 MSE，让球员之间差距同步缩小，MSE/r 对此不敏感。discrimination 直接度量评分解释了多少目标分布的标准差：接近 0 表示评分钝化；目标是 10 场 perf 均值、自带个体噪声，因此不要求恰好等于 1，而是要求不能过低。该指标配合标度约束（D-029）一起防止“更准但更钝”的候选胜出。

### D-032：draft 先验合并为统一起点

**选择**：R3e 后，将 `pick1_rating/pick_slope/rookie_floor` 合并为单一 `R_rookie_start`（R3g 后正式锁定 1425）；原有顺位线性递减仅保留为历史设计。

**未选**：保留三参数并强制 floor < pick1 维持顺位区分；直接对 rookie_start × boost × tau 做全联合网格。

**理由**：R3a-R3e 最优解为 floor=1450 > pick1=1430，公式 `max(...)` 使所有新秀初始均为 1450，slope 失效；最优点附近 MSE 差异极小，说明在当前开局 boost=6、tau=60 下顺位先验不提供预测增益。R3f 完成 1300-1475 单参数确认：MSE/r/discrimination/SD 几乎完全相同，起点水平只影响末季标度（整体平移不变，SD 不变）。R3g 的全量 entry-exit 检查显示相对流动稳定（1425 下整体退出分钟加权 1476.7、entry-exit 差距约 -52），但该差距在整体平移下不变，不能作为绝对水平选择器；正式锁定 1425 的依据是 D-029 的 top 2,200-2,500 软约束带内取最低合理起点。boost/tau 负责从统一起点快速收敛，因此不需要重新做三参数联合调优。该结论以开局 boost 为前提，后续若 boost 显著下调需重新检验。

---

## 附录：术语表

| 术语 | 定义 |
|------|------|
| Elo | 源自国际象棋的评分系统，通过比赛结果动态更新评分 |
| Dynamic Player Elo | 本项目设计的球员 Elo，以个人表现为主驱动、团队胜负为辅助修正 |
| GameScore | John Hollinger 提出的单场表现综合评分 |
| perf_i | 球员 i 在一场比赛中的个人表现标准分（z-score 标准化后的加权组合） |
| perf_used_i | 扣除评分回归项后的实际更新表现分 = `perf_i - (R - anchor) / surprise_scale` |
| on-court rate | 球员在场时每分钟的净胜分 |
| burn-in | 让系统从不准确初始状态收敛到合理的预热期 |

### D-033：rookie_start 作为固定展示约定

**选择**：`R_rookie_start=1425` 正式固定，不再作为可调参数；绝对水平是展示约定，不由 MSE 或 entry-exit 决定。若未来需要更直观的刻度（例如“联盟均值=1500”），在展示层统一平移即可，不回改引擎参数。

**未选**：为 1425/1450 再做全量对照；让双向初始化重新引入绝对水平选择；用 entry-exit 差距作为绝对水平拟合器。

**理由**：更新与预测只依赖相对水平；R3f 显示起点 1300-1475 对 MSE/r/discrimination/SD 无影响，R3g 显示 entry-exit 差距在整体平移下不变。1425 是 D-029 top 2,200-2,500 软约束带内最低合理起点，且比 1450 的绝对标度更保守。双向初始化（D-007）的第一遍预热也应使用同一 `rookie_start`，保持整体平移不变性，避免“1977 老将固定 1500、后续新秀随起点变化”造成的人为跨时代差异。

### D-034：R4 θ/α_modern 调优协议（只调现代评估窗口）

**选择**：R4 分三阶段：R4a 单调 θ（{0.1, 0.2, 0.3, 0.4, 0.6}，α_modern=1.0）；R4b 单调 α_modern（{0, 0.5, 0.75, 0.9, 1.0}，θ 取 R4a 最优）；R4c 围绕最优点做约 3×3 局部细筛，最优值顶边界则外扩。θ 不含 0：0 的语义是“完全去掉团队修正”，与项目“个人表现为主、团队胜负为辅”的前提相悖，且与现有 R1-R3 基线（θ=0.3）不可比。评估沿用现有边界 1997-2021（预测 1998-2022），该窗口已全部落在 on-court 时代，因此“只调现代”不需要再切更窄边界；引擎仍从 1976 顺序运行，早期数据只固定 α=1，不因 α_modern 的搜索而改变。

**状态更新（E-022/E-024/E-026，取代本节“θ 不含 0”）**：R4a 实际扩展为 θ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9}；θ=0 在固定 target 与 `alpha_ref=alpha_modern` 两种口径下均为 MSE 最优，最终两条轨道都保留 θ=0（D-035）。R4c 原 3×3 细筛由 E-024（α×θ 补扫 8 组）、E-025（α_ref 敏感性 8 组）、E-026（alpha_ref=0.9 下 θ 复核 5 组）取代。本节协议仅保留为历史记录。

**未选**：θ 网格包含 0；把评估边界改成 1996 起（α_modern 生效首季即参与评估）；只跑 1996 后的数据做训练。

**理由**：θ=0 既无业务意义又会让 MSE 与历史轮次失去可比性；1996-97 是 α 切换边界且属于 burn-in，不应直接进入评估；Elo 是顺序引擎，跳过 1976-1995 会让现代球员失去生涯历史，初始状态不成立，计算量也不会明显下降（评估指标只占运行时间很小部分）。α_modern 只影响 1996 后 Elo 更新（D-027a：CV target 固定 α_ref=1.0），早期评分在 R4 各网格间完全一致，因此不需要为早期单独调优。

### D-035：最终输出双轨制（α=1.0 全时代 + α=0.9 现代对比）

**选择**：正式结果同时维护两条轨道：(1) `α=1.0, θ=0` 全时代正典序列（1976-77 → 2025-26，内部 1976..2025，展示 1977..2026；`alpha_ref=1.0`），全程纯 GameScore z、同一更新尺度，用于任意跨时代球员比较；(2) `α=0.9, θ=0` 现代对比序列（1996+，on-court 混合，`alpha_ref=alpha_modern=0.9`），用于 1996 后球员相互比较。两轨共用 K=45、rookie_boost=6、rookie_tau=60、rookie_start=1425、playoff_k=1、on_court_mode=raw；scale 分轨：α=1.0 轨 285、α=0.9 轨 318（E-028/D-036）。调参与 Hold-out 边界不受展示范围影响：CV 仍以 1997-2021 边界为评估窗口，2023-24 起只用于最终 Hold-out 评估，展示序列可覆盖全部可用数据。

**状态更新（D-036）**：正式锁定两轨参数差异仅为 α 与 scale：共用 θ=0、K=45、rookie_boost=6、rookie_tau=60、rookie_start=1425、playoff_k=1、on_court_mode=raw；α=1.0 轨 scale=285，α=0.9 轨 scale=318。

**状态更新（E-028，标度校准）**：为让 1996 后双轨可直接互比，0.9 轨不做事后分数放大，而是通过调 K/scale 校准末季 SD：K=45、scale=318（E-028，last_sd=175.7，对齐 1.0 轨 175.5；top 不作为 0.9 轨约束）。因此两轨参数为：共用 θ=0、K=45、rookie_boost=6、rookie_tau=60、rookie_start=1425、playoff_k=1、on_court_mode=raw；α=1.0 轨 scale=285，α=0.9 轨 scale=318。

**正式锁定（E-028，用户确认）**：0.9 轨锁定为 K=45、scale=318、α=0.9、θ=0、rookie_boost=6、rookie_tau=60、rookie_start=1425、playoff_k=1、on_court_mode=raw；与 1.0 轨唯一差异是 α 与 scale（1.0 轨 scale=285）。

**状态更新（E-027，时点记录）**：R6 第一阶段在 θ=0 checkpoint（`results/elo_checkpoint_1995_theta0.csv`）上重跑了 K/scale 与 on-court 口径的 27 组联合网格；on_court_mode 三种口径（raw/team_relative/on_off）对 MSE/r/discrimination 无实质差异，raw 的 team_win_corr 最高，正式保留 raw。该时点 K/scale 仍待定，随后由 E-028 完成标度校准并锁定（见上），最终取值以 D-036 为准。

**状态更新（E-028）**：0.9 轨按“96 后双轨可比”定位完成 K/scale 标度校准：不重建 checkpoint（早期段为共享初始化），K 保持 45，scale 从 285 上调至 318，使末季 SD=175.7 对齐 α=1.0 轨 175.5；MSE/r 在 13 组网格中几乎不变，确认这是纯标度校准。两轨共用参数除 scale 外一致。

**未选**：只保留 α=0.9 并把现代段重新标定后与早期合并；只保留 α=1.0 完全不展示 on-court 版本。

**理由**：α<1 只在 1996 后生效（on-court 数据边界），必然把历史切成两段更新尺度，现代球员与早期球员的绝对分不可直接比较；α=1.0 是唯一全历史同一信号体系的配置。E-025 对比显示两轨 top-50 峰值名单 49/50 重合、逐赛季第一人一致，差异主要是绝对标度和个别跨时代边缘比较。θ 在固定 target 与 alpha_ref=0.9 两种口径下均以 0 最优（E-022/E-024/E-026），因此两条轨道都不含胜负修正。

### D-036：最终输出 schema、双轨交付与查询 API

**选择**：`results/nba_elo.db`（SQLite）为唯一事实源，统一 D-020 的三表语义并扩展为以下结构：

- `meta`：schema_version、generated_at、params_hash、initial_state_hash、checkpoint_hashes_json，保证全库可复现
- `params`：每条轨道的全部引擎参数（track、k、scale、theta、alpha_modern、alpha_ref、rookie_*、on_court_mode、playoff_k），rating 行按 track 关联
- `run_variants`：8 个 variant 的展开表（variant_id、track、playoff_k、continuation_source），把 shared 段复制到 canonical/modern 两条时间线
- `players` / `games`：静态元数据，来自 clean_data
- `rating_events`：球员 × 比赛 × track 事件表，直接落盘引擎 updates 列（rating_before/after、delta_raw/delta_adj、zero_mean_offset、perf_i、perf_used、game_score、z_game_score、on_court_rate、z_on_court_rate、minutes_share、k_effective、game_date、season、team_id、home）；1996 前两轨数值相同并标记 source='shared'
- `season_ratings`：球员 × 赛季 × track 物化表（season_end 按 D-024 展示年份、rating、rank、minutes、games）
- `snapshots`：球员 × 日期 × track 快照（snapshot_date、rating、rank；全明星约 2 月初、赛季末等）
- `dual_track_season`：视图，join 两条轨的 season_ratings → R_1.0、R_0.9、delta、rank_1.0、rank_0.9、rank_diff

**导出**：`results/export/` 只导出派生层（`season_ratings.csv`、`snapshots/*.csv`、`dual_track_season.csv`）；事件层留在库内查询。导出文件带 run_id 与生成时间。

**查询 API（src/query.py）**：`player_timeline(player_id, track, date_range)` 返回生涯曲线；`rank_as_of(date, track)` 返回任意日期排名；`explain_change(player_id, game_id)` 返回单场归因（perf_i、perf_used、delta_adj、zero_mean_offset 等）；`compare_dual(season, player_id=None)` 返回 dual_track_season 视图。

**双轨最终运行协议**：α=1.0 正典轨走完整 1976-2025 双向初始化（D-007 三遍）；α=0.9 现代轨不做独立反向传播，从正典轨双向后的 1995 年末状态续跑 1996-2025，保持两轨 96 前完全一致。Hold-out 与 R7 敏感性按轨道分别报告，0.9 轨只报告 1996+ 评估。

**状态更新（E-031）**：协议已执行——正典轨三遍双向与 0.9 轨续跑均完成，输出 `results/final/`；双轨末季 SD 差 <0.7、分钟加权均值差 <2.5，标度对齐成功。

**状态更新（E-035）**：交付库已建——`scripts/build_delivery_db.py` 完整重写并跑通，`results/nba_elo.db` 含 8 个 variant（canonical/modern × playoff_k 0.5/1.0/1.5/2.0）；`rating_events` 7,924,960 行、`season_ratings` 171,184 行（8 × 50 个展示赛季 1977-2026）、`snapshots` 329,912 行；`variant_timeline` 8 条完整时间线共 9,616,312 事件；零和、NULL、96 前双轨一致性与末季标度校验全部通过，`PRAGMA integrity_check` ok。

**未选**：以 CSV/Parquet 为唯一结果源；0.9 轨独立双向初始化；只存赛季末分数而丢弃事件表。

**理由**：事件表一次写入、多粒度派生，任意时间排名与逐场归因都不需要重跑引擎；双轨对比只是同一 schema 上的视图 join，不会出现两套口径；0.9 轨独立双向会破坏“96 前共享”的对照前提，也会让现代轨的 96 前初始值失去与正典轨的可比性。
