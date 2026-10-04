# DeepSeek Smart Router · v3.0 Omega

> 为 Agent Harness CLI 打造的**自我进化**多模型智能路由 —— 三层省钱防御（缓存/级联/路由）之上，再加**经验回忆、模型竞技场、质量成本拨盘**，把每一次调用都变成系统的养料。

## v3.0 Omega: 三大新前沿能力

| 能力 | 前沿依据 | 实现 | 实测 |
|------|----------|------|------|
| 🧬 **kNN 经验回忆** | RouteLLM 语义路由器：相似历史任务的成败应指导当下路由 | 每次调用把（问题向量， 模型， 奖励）入库；新请求检索 top-k 相似经验，±15 分回忆加成直接改变派单 | "Redis 缓存穿透怎么解决啊？" → ds-v3 获得 +1.0 回忆加成（54→69 分）,Claude 被压到 30 分；新问题完全不受影响 |
| ⚔️ **MoA 竞技场** | Mixture-of-Agents (ICLR 2025 Spotlight) 超 GPT-4o;Princeton 2025: 提议者质量>多样性 | 困难档并行 fan-out Top-N 模型，内置评审选冠军；**胜负反哺学习引擎（竞赛学习）**，冠军 +0.9 / 败者 +0.3 | 评审拒答型答案 0.1 分、完整答案 1.0 分，异常参赛者自动出局不拖垮全场 |
| 🎛️ **λ 质量-成本拨盘** | 研究共识：路由本质是 quality-cost Pareto 上的 λ 权衡 | `routing.quality_lambda`: 0=极限省钱 / 1=极限质量，默认 0.55 平衡 | λ=1 时 Claude 居首，λ=0 时最贵模型殿后，一键切换人格 |

## v2.0 Frontier: 三层省钱防御

基于 RouteLLM (ICLR 2025)、FrugalGPT (TMLR)、GPTCache/SISO (2025)、GPT-5 混合路由架构的深度分析，本项目实现了学术界与工业界验证有效的完整省钱链路：

```
请求 ──> ① 语义缓存: 相似问题直接返回, 零成本零延迟 (GPTCache 路线)
           │ 未命中
           ▼
         ② 级联升级: 便宜档先答, 内置评审不合格再升级 (FrugalGPT 路线)
           │ 升级事件中差评反哺学习引擎, 级联越用越少
           ▼
         ③ 预测路由: 任务画像 × 能力画像 × 学习反馈 (RouteLLM 路线)
           │
           ▼
         熔断降级 + 预算守卫 + 协议翻译 (基础设施层)
```

| 层 | 前沿依据 | 本项目实现 |
|----|----------|-----------|
| 语义缓存 | GPTCache: 命中即省 100% 成本；FAQ 类负载命中率 30-60% | 内置零依赖哈希向量（可选换 API embedding)，余弦阈值 + 档位硬边界 + TTL + LRU，命中计入成本看板 |
| 级联升级 | FrugalGPT: 匹配最优模型可省 50-98%;GPT-5 用"预测+级联"混合架构 | 模糊任务降一档先答 → 启发式评审（拒答/过短/错误/重复/任务契合）→ 不合格升级；评审结果反哺学习引擎 |
| 预测路由 | RouteLLM: 省 2-3.66 倍成本保 95% 质量 | 难度分类器 × best_fit 能力匹配 × Thompson Sampling 学习闭环 |

**为什么是三层而不是一层**：缓存处理"重复"，级联处理"模糊"，预测路由处理"明显"——三者覆盖的请求类型互不重叠，叠加后省钱效果相乘。

## v1.0 Genesis 遗产 (全部保留)

| 支柱 | 说明 |
|------|------|
| 🧠 **学习闭环** | Thompson Sampling: 每个 (模型×任务标签) 维护 Beta 分布，隐式信号 + `POST /v1/feedback` 显式打分；好评放大 1.5x，差评压到 0.5x,SQLite 持久化 |
| 💰 **预算守卫** | 日花费限额，超限自动降档或 429，看板实时用量条 |
| 🔧 **运行时管理** | `/v1/admin` 免重启启停模型；配置文件保存即热重载 |
| 📊 **可观测** | 看板：学习胜率、平均延迟、预算用量、缓存命中率；响应带 `request_id` 供反馈引用 |

## 解决什么问题

用 Harness 跑模型的用户有四个痛点：**全程挂旗舰模型太贵，全程挂便宜模型怕翻车；手握多家 Key 却无法整体调度；静态规则不知道哪个模型在你的真实任务上表现好；重复的问题一遍遍重复付钱**。

Smart Router 以本地代理形式接入任意 OpenAI 兼容的 Harness(aider / opencode / continue / 自研 harness)，把所有供应商的模型注册进同一个池子，用三层防御 + 学习闭环把每一分钱花在刀刃上。

## 支持的供应商

| 协议类型 | 覆盖厂商 |
|----------|----------|
| `openai` | **DeepSeek** / OpenAI / 通义千问 / 智谱 GLM / Kimi / OpenRouter / Groq / Mistral / Together / SiliconFlow / Ollama / vLLM …（一切 OpenAI 兼容端点） |
| `anthropic` | Claude 全系（原生 Messages API，请求/响应/流式自动协议翻译，含 tool_use) |
| `gemini` | Google Gemini 全系（原生 generateContent，自动协议翻译，含 function call) |
| `azure` | Azure OpenAI(deployment 路径 + api-key 认证） |

新增一家供应商 = 在 YAML 里加 4 行配置，无需改代码。

## 核心特性

- **多供应商模型池**: 每个模型声明 `tier`(trivial/standard/hard 难度档)、`priority`、`price`,同档可多模型互为备份
- **最适配分配 (best_fit)**: 识别任务类型（代码/数学/推理/翻译/写作/长上下文）× 每个模型的能力画像，计算适配度得分，把任务交给池里**最适合**的模型，而不是机械地按档位派单
- **硬过滤保障**: 视觉任务自动排除纯文本模型、超长上下文自动排除小窗口模型、工具调用任务自动排除不支持 tools 的模型——杜绝"派错人"
- **零成本难度分类器**: 纯启发式多信号评分（推理关键词、堆栈信息、编码意图、上下文规模、对话深度、工具调用链），不额外调用模型，零延迟
- **三种寻路方式**:
  - `model: "auto"`（或任意未知名）→ 按难度自动路由
  - 别名 `cheap` / `fast` / `smart` / `reasoning` / `default` → 锁定难度档
  - 直接写池内名称（如 `ds-r1`)→ 锁定该模型，**显式选择永远优先**
- **消息级控制**: 消息里加 `[think]` 强制困难档，`[quick]` 强制简单档
- **单模型熔断 + 候选链降级**: 某模型连续失败 3 次熔断 60 秒，期间自动沿候选链（同档备选 → 跨档降级）切换，任务不中断
- **调度策略**: `best_fit`（任务×能力最适配，默认推荐）/ `priority` / `cheapest` / `round_robin`
- **流式协议翻译**: Anthropic/Gemini 的 SSE 事件流实时转 OpenAI chunk 格式，harness 无感知
- **实时成本看板**: 内置 `/dashboard`，按 `provider/model` 展示每笔路由决策、实际花费、对比"全走最贵档"省下的金额
- **干跑模式**: `/v1/route/preview` 只看决策不发请求，含完整候选链和熔断状态

## 快速开始

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxx          # 有哪些 Key 就导出哪些
# export ANTHROPIC_API_KEY=sk-ant-xxx
# export GEMINI_API_KEY=xxx

uvicorn router.main:app --port 8355

# (可选) 本地 trivial 档: 安装 Ollama 后
ollama pull qwen2.5-coder:7b
```

把 Harness 的 API 地址指向路由即可，无需改其他配置：

```bash
export OPENAI_API_BASE=http://localhost:8355/v1
export OPENAI_API_KEY=sk-xxx
aider --model openai/auto      # auto=按难度路由; 也可写 smart / ds-r1 等
```

看板： <http://localhost:8355/dashboard>

## 使用方式

```bash
# 干跑: 查看路由计划 (任务画像/适配度分解/学习系数/出局原因)
curl localhost:8355/v1/route/preview -H 'Content-Type: application/json' -d '{
  "model":"auto",
  "messages":[{"role":"user","content":"设计一个百万并发的订单系统架构, 分析权衡"}]}'

# 显式反馈: 让调度越用越准 (响应里的 router.request_id 直接引用)
curl localhost:8355/v1/feedback -H 'Content-Type: application/json' -d '{
  "request_id": 42, "score": 0.9}'

# 运行时管理: 免重启禁用/启用模型
curl -X POST localhost:8355/v1/admin/models/ds-r1/disable
curl localhost:8355/v1/admin/models

# 成本统计 + 学习状态 + 延迟 + 预算
curl localhost:8355/v1/stats
```

响应头始终携带 `X-Router-Tier` / `X-Router-Model` / `X-Router-Score`；非流式响应体里的 `router` 字段包含 provider、候选链降级情况等完整决策细节。

## 架构

```
Harness CLI ──OpenAI 兼容──> Smart Router (FastAPI, :8355)
                                 │
              ┌──────────────────┼─────────────────────┐
              ▼                  ▼                     ▼
        难度分类器          候选链调度器            成本统计 (SQLite)
        (0-100 分)    (策略排序+熔断+跨档降级)          │
              │                  │                     │
              ▼                  ▼                     ▼
   ┌──────────────┬──────────────┬──────────────┬──────────────┐
   │ trivial 档    │ standard 档  │ hard 档       │  协议适配层    │
   │ 本地小模型    │ V3/4o-mini/ │ R1/Claude/   │ openai/azure/│
   │ Gemini Flash │ Qwen/GLM... │ o1/Gemini Pro│ anthropic/   │
   │              │              │              │ gemini       │
   └──────────────┴──────────────┴──────────────┴──────────────┘
```

| 文件 | 职责 |
|------|------|
| `router/classifier.py` | 难度评分（0-100) + 任务类型识别，每个信号记录理由，决策可解释 |
| `router/matcher.py` | 能力匹配器：任务画像 × 模型能力画像 → 适配度（0-100)，内置常见模型画像 |
| `router/learner.py` | 学习引擎：Thompson Sampling 反馈闭环，SQLite 持久化 |
| `router/memory.py` | **kNN 经验回忆**: 相似历史任务检索，±15 分回忆加成 |
| `router/moa.py` | **MoA 竞技场**: 并行 fan-out + 评审选冠军 + 竞赛学习 |
| `router/cache.py` | 语义缓存：向量相似命中 + 档位硬边界 + TTL + LRU 淘汰 |
| `router/embedder.py` | 向量化：内置零依赖哈希 n-gram / 可插拔 API embedding |
| `router/judge.py` | 级联评审：拒答/过短/错误/重复/任务契合五维判分 |
| `router/budget.py` | 预算守卫：日限额超限自动降档/拒绝 |
| `router/providers.py` | 协议适配层：4 类协议族的请求/响应/流式翻译 |
| `router/proxy.py` | 候选链调度：best_fit×学习系数×回忆加成×λ 拨盘，熔断降级，SSE 透传 |
| `router/stats.py` | SQLite 持久化：token/成本/基线差额/延迟/问题摘要，request_id 追踪 |
| `router/main.py` | 三层防御+竞技场编排 + 全部端点 + 热重载 + 运行时管理 |

## 最适配分配是如何工作的

```
任务 ──> 任务画像: {类型标签, 输入规模, 硬性需求(视觉/工具)}
              │
              ▼
   1. 硬过滤: 需要视觉但模型不支持? 出局
              输入 50k tokens 但窗口 32k? 出局
              需要工具调用但模型不支持? 出局
              ▼
   2. 适配度 = 0.60×能力匹配 + 0.25×档位契合 + 0.15×成本契合(对数曲线)
              ▼
   3. 候选链按适配度排序, 逐一下发, 失败自动降级
```

内置画像覆盖 DeepSeek / GPT / Claude / Gemini / Qwen / GLM / Kimi / Llama 等常见模型（按模型 ID 自动匹配）;接入新模型时在 YAML 里声明即可：

```yaml
models:
  my-model:
    provider: openrouter
    model: some/new-model
    tier: standard
    price: { input: 0.5, output: 1.5 }
    strengths: { reasoning: 8, math: 7, code: 9, writing: 6, translation: 6, long_context: 5 }
    context_window: 131072
    supports_tools: true
    supports_vision: false
```

实测示例（DeepSeek + Claude + Gemini + Qwen + 本地七模型池）:

| 任务 | 首选模型 | 说明 |
|------|----------|------|
| "证明根号 2 是无理数并推导连分数" | deepseek-reasoner （适配度 82) | 推理能力 10/10 居首 |
| "识别这张截图里的错误" | gemini-2.0-flash | 纯文本模型全部出局 |
| "写一篇 2000 字小红书文案" | gemini-2.0-flash | 简单任务+写作强项+便宜 |

## 配置

复制 `config.example.yaml` 为 `config.yaml`，按需修改后 `export ROUTER_CONFIG=config.yaml`。

接入新供应商示例（以 Claude 顶困难档备份为例）:

```yaml
providers:
  anthropic:
    type: anthropic
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY

models:
  claude-sonnet:
    provider: anthropic
    model: claude-sonnet-4-5
    tier: hard          # 和 ds-r1 同档, R1 熔断后自动顶上
    priority: 2
    price: { input: 3.0, output: 15.0 }
```

`price` 表用于成本核算与 `cheapest` 策略，厂商调价时同步即可。

## 测试

```bash
python -m pytest tests/ -q   # 101 个用例: 分类器 / 能力匹配 / 学习引擎 / 语义缓存 /
                             # 级联评审 / 经验回忆 / MoA 竞技场 / λ 拨盘 / 协议翻译 ...
```

## 路线图

- [ ] SISO 式聚类质心缓存, 提升泛化命中率
- [ ] LLM-as-Judge 可选评审器 (用小模型替代启发式, 级联/MoA 判分更准)
- [ ] LLMLingua 提示词压缩 (>1000 tokens 的长上下文场景, 省 50-80% 输入)
- [ ] MoA 聚合器模式 (冠军答案+其他提案合成最终答案, 而非单纯选拔)
- [ ] A/B 影子模式: 新模型小流量灰度, 学习数据够了再转正
