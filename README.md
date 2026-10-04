# DeepSeek Smart Router

> 为 Agent Harness CLI 打造的多模型智能路由插件 —— 按任务难度在**你所有的 API 供应商**之间自动调度，省钱不降质。

## 解决什么问题

用 Harness 跑模型的用户有两个痛点：**全程挂旗舰模型太贵，全程挂便宜模型怕复杂任务翻车**；而且手里往往同时有 DeepSeek、OpenAI、Claude、Gemini 等多家 Key，却没有工具能把它们当成一个整体调度。

Smart Router 以本地代理形式接入任意 OpenAI 兼容的 Harness(aider / opencode / continue / 自研 harness),把**所有供应商的模型注册进同一个池子**，对每次请求实时评估难度，路由到性价比最高的可用模型。

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
- **零成本难度分类器**: 纯启发式多信号评分（推理关键词、堆栈信息、编码意图、上下文规模、对话深度、工具调用链），不额外调用模型，零延迟
- **三种寻路方式**:
  - `model: "auto"`（或任意未知名）→ 按难度自动路由
  - 别名 `cheap` / `fast` / `smart` / `reasoning` / `default` → 锁定难度档
  - 直接写池内名称（如 `ds-r1`)→ 锁定该模型，**显式选择永远优先**
- **消息级控制**: 消息里加 `[think]` 强制困难档，`[quick]` 强制简单档
- **单模型熔断 + 候选链降级**: 某模型连续失败 3 次熔断 60 秒，期间自动沿候选链（同档备选 → 跨档降级）切换，任务不中断
- **调度策略**: `priority`（按优先级）/ `cheapest`（最便宜优先）/ `round_robin`（轮询）
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
# 干跑: 查看路由计划 (分数/档位/候选链/熔断状态)
curl localhost:8355/v1/route/preview -H 'Content-Type: application/json' -d '{
  "model":"auto",
  "messages":[{"role":"user","content":"设计一个百万并发的订单系统架构, 分析权衡"}]}'
# => {"score":65,"tier":"hard","candidates":[{"name":"ds-r1",...},{"name":"ds-v3",...}],...}

# harness 可选的模型列表 (别名 + 池内名称)
curl localhost:8355/v1/models

# 成本统计
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
| `router/classifier.py` | 难度评分（0-100)，每个信号记录理由，决策可解释 |
| `router/providers.py` | 协议适配层：4 类协议族的请求/响应/流式翻译 |
| `router/proxy.py` | 候选链调度：策略排序、单模型熔断、跨档降级、SSE 翻译透传 |
| `router/stats.py` | SQLite 持久化，逐笔记录 token/成本/基线差额 |
| `router/main.py` | `/v1/chat/completions`、`/v1/models`、`/v1/route/preview`、`/v1/stats`、`/dashboard` |

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
python -m pytest tests/ -q   # 39 个用例: 分类器 / 候选链调度 / 熔断降级 /
                             # Anthropic·Gemini·Azure 协议翻译 / 成本计算
```

## 路线图

- [ ] 基于历史反馈的自学习阈值 (bandit 算法动态调 trivial/hard 分界线)
- [ ] 小型 Embedding 分类器作为启发式的可选增强
- [ ] 按 Harness 会话维度聚合成本报表
- [ ] 延迟感知调度 (实测 TTFT 参与候选排序)
