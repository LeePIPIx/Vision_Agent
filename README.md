# Medical Imaging Agent 技术路线

## 1. 项目简介

当前 Demo 的入口是 `backend_api.py`：前端与 API 共用一个端口，注册账号后可运行官方 nnU-Net v1 脾脏 CT 分割。Laya 负责其他聊天任务的业务路由；官方权重和 CT 样例通过 `scripts/setup_demo.py` 安装。下面是新的运行说明。

本项目旨在构建一个面向医学影像分析的 Agent 系统，用于完成医学影像信息读取、分割模型调用、病灶统计分析、医学知识检索以及结构化报告生成等任务。

项目重点不是简单调用大模型，而是通过从零手写 Agent 到使用 LangGraph 重构的过程，系统掌握 Agent 的核心机制和工程化实现方式。


### CT 脾脏分割 Demo（当前可运行）

使用 Python 3.10+，建议 CUDA GPU。官方 nnU-Net `Task009_Spleen` 是 **v1** 权重，与 nnU-Net v2 不兼容；本项目把 v1 推理安装在 `.venv`，只下载 `3d_fullres` 的 `fold_0` 权重，不运行五折集成，也不应用额外后处理。样例来自公开 MSD `Task09_Spleen` 训练集，只用于流程演示，不可作为独立准确率验证。官方来源：[nnU-Net 预训练权重](https://zenodo.org/records/4003545)、[nnU-Net v1/v2 兼容说明](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/run_inference_with_pretrained_models.md)、[MONAI 样例数据源](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/datasets.py)。

```bash
cd /home/ljr/workspace/DL/Medical_agent
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install 'nnunet==1.7.1' remotezip
.venv/bin/python scripts/setup_demo.py
export OPENAI_API_KEY='你的 DeepSeek Key'   # 仅聊天模型需要，CT Demo 可不设置
export USE_TF=0
.venv/bin/python backend_api.py
```

打开后端地址的根路径，例如服务器的 8000 端口在 VS Code 转发为本机 8001，则访问 `http://localhost:8001/`。首次使用注册账号，点击“运行公开 CT Demo”，也可上传腹部 CT `.nii`/`.nii.gz` 后直接发送分析请求，系统会自动进入脾脏分割流程。保留同一会话可以追问已有结果。任务状态来自 SQLite 和 SSE；断线刷新后可在“任务记录”查看、取消或重试。运行中的任务在服务重启时会标记为中断，可手动重试。

结果包括 `mask.nii.gz`、`report.json`、`report.md` 和叠加 PNG。体积按 `体素数 × |det(affine, mm)| / 1000` 计算为 mL；CT 与 mask 的尺寸、空间坐标和单位必须相符。报告记录输入、mask 与模型权重摘要，初始状态是“待人工复核”。页面可查看 CT 切片、叠加 mask、记录复核意见，并上传修订 mask 重新计算，旧版报告保留。连通区域数不是病灶数。Demo 权重只覆盖脾脏，不代表其他器官或病灶。

用户、病例、任务、会话和事件位于 `runtime/app.sqlite3`，LangGraph 会话检查点位于 `runtime/checkpoints.sqlite3`，输入和结果也在 `runtime/cases/`。这些目录在 `.gitignore` 中。账号间按归属隔离任务和文件；部署到公网前请配置 HTTPS、反向代理和受控注册（`MEDICAL_ALLOW_REGISTRATION=0`），并备份 `runtime/`。单实例 worker 使用文件锁，Uvicorn 必须保持单 worker 运行。上传上限 256 MB，可通过 `MEDICAL_TASK_TIMEOUT` 设置单任务超时时间，默认 900 秒。医学知识检索仍需 `TAVILY_API_KEY`。

验证：

```bash
.venv/bin/python -m pytest -q tests
USE_TF=0 HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_router.py
```

路由脚本只有 14 条手写样例，旨在发现明显误分，不代表上线准确率。使用真实病例前仍需独立标注数据、影像质量检查、模型性能验证以及临床人员复核。

---

## 2. 项目目标

本项目最终希望实现如下能力：

- 支持输入医学影像路径，例如 `.nii.gz`、`.nii`，后续可扩展 DICOM。
- 支持根据用户问题自动判断任务类型。
- 支持调用不同医学影像分析工具。
- 支持调用分割模型生成病灶 mask。
- 支持根据 mask 计算病灶体积、占比、最大层面等统计信息。
- 支持基于医学知识库进行 RAG 检索。
- 支持生成结构化医学影像分析报告。
- 支持记录 Agent 执行状态、工具调用结果和错误信息。
- 支持从手写 Agent 流程重构为 LangGraph 工作流。

---

## 3. 项目最终形态

用户输入医学影像路径和自然语言问题后，Agent 自动完成相关任务。

### 3.1 输入示例

```bash
python main.py \
  --image_path data/samples/case001.nii.gz \
  --query "请帮我生成这个病例的结构化分析报告"
```

### 3.2 输出内容

系统最终输出：

1. 影像基本信息
2. 分割结果路径
3. 病灶统计信息
4. 医学知识检索结果
5. 结构化医学影像分析报告
6. 工具执行日志
7. 错误信息记录

### 3.3 调用链日志

每次 LangGraph 请求都会生成独立的 `request_id`，并将完整调用顺序写入
`logs/app.log`。日志记录 Supervisor 路由、模型请求与响应、工具调用参数、
工具执行结果、重试/熔断和最终回答，同时包含顺序号与累计耗时。

```text
[CALL_TRACE] request=req-... session=sess-... seq=05 +1234.5ms component=tool action=execute_start details={...}
```

按请求查看完整调用链：

```bash
rg 'request=req-具体编号' logs/app.log
```

实时观察所有模型和工具调用：

```bash
tail -f logs/app.log | rg 'CALL_TRACE'
```

API 响应中的 `request_id`、`call_sequence` 和 `logs` 字段也会返回同一条
调用链。敏感字段会自动脱敏，过长的模型内容或工具结果会被截断。

---

## 4. 整体架构

```text
User
 ↓
Agent Controller
 ↓
Task Router
 ├── Image Info Tool
 ├── Segmentation Tool
 ├── Lesion Analysis Tool
 ├── Medical RAG Tool
 ├── Report Generation Tool
 └── Memory Tool
 ↓
Result Synthesizer
 ↓
Final Answer / Report
```

---

## 5. 核心模块说明

| 模块                 | 作用                           | 对应知识点                  |
| -------------------- | ------------------------------ | --------------------------- |
| Agent Controller     | 控制 Agent 主流程              | Agent Loop、任务调度        |
| Task Router          | 判断用户意图，决定调用哪些工具 | Tool Calling、任务分流      |
| Image Info Tool      | 读取 NIfTI / CT 影像基本信息   | 医学影像处理、nibabel       |
| Segmentation Tool    | 调用分割模型生成 mask          | 模型推理、工具封装          |
| Lesion Analysis Tool | 根据 mask 计算病灶体积、占比等 | NumPy、医学影像统计         |
| Medical RAG Tool     | 检索医学知识库，辅助解释       | RAG、Embedding、向量检索    |
| Report Tool          | 生成结构化医学报告             | Prompt 工程、结构化输出     |
| Memory Tool          | 保存对话历史和长期记忆         | 短期 Memory、长期 Memory    |
| LangGraph Workflow   | 将流程重构为状态图             | State、Node、Edge、条件分支 |

---

## 6. 开发策略

本项目采用三阶段开发路线：

```text
第一阶段：手写 Agent MVP
第二阶段：使用 LangGraph 重构
第三阶段：接入真实医学影像模型和 RAG
```

### 6.1 开发原则

- 先跑通最小闭环，再逐步增强能力。
- 先手写理解底层机制，再使用框架提升工程化能力。
- 先使用 dummy 工具模拟流程，再替换为真实模型。
- 先实现命令行 Demo，再考虑前端或可视化界面。
- 先保证流程清晰可解释，再追求功能复杂度。

---

# 第一阶段：手写 Agent MVP

## 1. 阶段目标

不使用 LangGraph，先手写一个最小可运行的 Agent 系统，理解 Agent 的基本执行流程。

核心流程如下：

```text
用户输入
 ↓
任务识别 Router
 ↓
调用对应 Tool
 ↓
Tool 返回结果
 ↓
汇总生成最终回答
```

---

## 2. 阶段一功能范围

第一阶段只实现最小闭环，不加入复杂 RAG、多 Agent 和真实分割模型。

支持工具如下：

| 工具               | 功能                    | 备注                     |
| ------------------ | ----------------------- | ------------------------ |
| ImageInfoTool      | 读取 NIfTI 影像基本信息 | 真实实现                 |
| SegmentationTool   | 模拟分割模型输出 mask   | dummy 实现               |
| LesionAnalysisTool | 根据 mask 计算病灶统计  | 可先 dummy，后续真实实现 |
| ReportTool         | 生成结构化报告          | 模板实现                 |

---

## 3. 阶段一目录结构

```text
medical_agent/
├── main.py
├── agents/
│   └── medical_agent.py
├── tools/
│   ├── base_tool.py
│   ├── registry.py
│   ├── image_info_tool.py
│   ├── segmentation_tool.py
│   ├── lesion_analysis_tool.py
│   └── report_tool.py
├── utils/
│   └── logger.py
├── data/
│   └── samples/
└── outputs/
    ├── masks/
    ├── reports/
    └── logs/
```

---

## 4. 阶段一核心任务

### 4.1 实现 BaseTool

所有工具统一继承 `BaseTool`，保证工具接口一致。

每个 Tool 至少包含：

```text
name
description
run()
```

目标：

- 理解工具抽象。
- 理解类、继承、方法重写。
- 为后续 Tool Calling 打基础。

---

### 4.2 实现 ToolRegistry

`ToolRegistry` 负责管理所有工具。

核心功能：

1. 注册工具。
2. 根据名称获取工具。
3. 查看所有可用工具。

目标：

- 理解 Agent 的“工具箱”机制。
- 统一管理工具调用。
- 避免在 Agent 中硬编码具体工具类。

---

### 4.3 实现 ImageInfoTool

功能：

```text
输入：NIfTI 文件路径

输出：
- shape
- affine
- min value
- max value
- mean value
```

使用库：

```text
nibabel
numpy
```

---

### 4.4 实现 SegmentationTool

第一版先使用 dummy 工具，不直接接真实模型。

功能：

```text
输入：image_path
输出：mask_path
```

示例输出：

```json
{
  "status": "success",
  "model": "dummy_segmentation_model",
  "mask_path": "outputs/masks/demo_mask.nii.gz"
}
```

后续再替换为：

```text
Inf-Net / nnUNet / 自研医学影像分割模型
```

---

### 4.5 实现 LesionAnalysisTool

功能：

```text
输入：
- image_path
- mask_path

输出：
- lesion_volume_mm3
- lesion_ratio
- max_slice_index
```

第一版可以先 dummy，第二版再读取真实 mask 计算。

---

### 4.6 实现 ReportTool

功能：

根据影像信息、分割结果和病灶统计生成结构化报告。

报告结构：

```text
1. 影像基本信息
2. 分割结果
3. 病灶统计
4. 初步分析
5. 注意事项
```

医学安全提示必须保留：

```text
本报告仅作为辅助分析，不替代医生诊断。
```

---

### 4.7 实现 MedicalAgent

Agent 主流程：

```text
1. 接收 user_query 和 image_path
2. 判断 task_type
3. 调用对应工具
4. 保存中间结果到 state
5. 汇总 final_answer
```

任务类型设计：

| task_type     | 触发条件         | 调用工具                                              |
| ------------- | ---------------- | ----------------------------------------------------- |
| image_info    | 用户询问影像信息 | ImageInfoTool                                         |
| segmentation  | 用户要求分割     | ImageInfoTool + SegmentationTool                      |
| analysis      | 用户要求病灶分析 | ImageInfoTool + SegmentationTool + LesionAnalysisTool |
| report        | 用户要求生成报告 | 全部工具                                              |
| full_analysis | 默认完整分析     | 全部工具                                              |

---

## 5. 阶段一完成标准

- [ ] 可以通过命令行输入影像路径和问题。
- [ ] Agent 可以识别任务类型。
- [ ] Agent 可以调用不同工具。
- [ ] 所有工具返回结果保存到 state。
- [ ] 可以生成最终回答。
- [ ] 出错时可以记录 error。
- [ ] 项目结构清晰。
- [ ] 工具接口统一。

---

# 第二阶段：使用 LangGraph 重构

## 1. 阶段目标

将手写 Agent 流程重构为 LangGraph 状态图工作流。

重构前：

```text
if / else 控制流程
```

重构后：

```text
State + Node + Edge + Conditional Edge
```

---

## 2. LangGraph 版本架构

```text
START
 ↓
router_node
 ↓
conditional_edges
 ├── image_info_node
 ├── segmentation_node
 ├── lesion_analysis_node
 ├── rag_node
 └── report_node
 ↓
final_node
 ↓
END
```

---

## 3. 阶段二目录结构

```text
medical_agent/
├── graphs/
│   ├── __init__.py
│   ├── state.py
│   ├── nodes.py
│   ├── edges.py
│   └── build_graph.py
```

---

## 4. State 设计

`State` 是 Agent 执行过程中的共享数据结构。

```text
AgentState:
- user_query
- image_path
- task_type
- image_info
- segmentation_result
- lesion_stats
- rag_context
- report
- messages
- errors
- final_answer
```

理解重点：

```text
State = Agent 的共享记忆板
Node = 读取 State，执行一步操作，然后更新 State
Edge = 根据 State 决定下一步走哪里
```

---

## 5. Node 设计

| Node                 | 功能                 |
| -------------------- | -------------------- |
| router_node          | 判断任务类型         |
| image_info_node      | 调用影像信息工具     |
| segmentation_node    | 调用分割工具         |
| lesion_analysis_node | 调用病灶分析工具     |
| rag_node             | 调用医学知识检索工具 |
| report_node          | 调用报告生成工具     |
| final_node           | 汇总最终回答         |

---

## 6. Conditional Edge 设计

根据 `task_type` 决定执行路径：

```text
image_info
  → image_info_node
  → final_node

segmentation
  → image_info_node
  → segmentation_node
  → final_node

analysis
  → image_info_node
  → segmentation_node
  → lesion_analysis_node
  → final_node

report
  → image_info_node
  → segmentation_node
  → lesion_analysis_node
  → report_node
  → final_node

full_analysis
  → image_info_node
  → segmentation_node
  → lesion_analysis_node
  → rag_node
  → report_node
  → final_node
```

---

## 7. 阶段二完成标准

- [X] 使用 LangGraph 定义 AgentState。
- [ ] 将工具调用封装为 Node。
- [ ] 使用 Edge 控制执行顺序。
- [ ] 使用 Conditional Edge 实现任务分流。
- [ ] `graph.invoke()` 可以正常执行。
- [ ] LangGraph 版本输出结果与手写版本一致。
- [ ] 能解释手写版和 LangGraph 版的对应关系。

---

# 第三阶段：加入 RAG 医学知识检索

## 1. 阶段目标

构建医学知识库，使 Agent 可以基于医学资料回答相关问题，而不是只依赖模型自身知识。

---

## 2. RAG 流程

```text
医学资料
 ↓
文档加载
 ↓
文本切分
 ↓
Embedding
 ↓
向量数据库
 ↓
用户问题检索
 ↓
相关内容拼接 Prompt
 ↓
LLM 生成回答
```

---

## 3. 知识库目录

```text
data/
└── knowledge/
    ├── lung_infection.md
    ├── mucus_plug.md
    ├── bronchiectasis.md
    └── ct_report_rules.md
```

建议第一版使用 Markdown 文件，不要一开始处理复杂 PDF。

---

## 4. RAG 模块目录

```text
rag/
├── __init__.py
├── document_loader.py
├── text_splitter.py
├── embedding.py
├── vector_store.py
└── retriever.py
```

---

## 5. RAG Tool 功能

输入：

```text
query
```

输出：

```text
retrieved_docs
```

作用：

1. 根据用户问题检索医学知识。
2. 为报告生成提供医学解释。
3. 为问答提供资料依据。

---

## 6. 阶段三完成标准

- [ ] 可以加载 Markdown 医学资料。
- [ ] 可以完成文本切分。
- [ ] 可以生成 Embedding。
- [ ] 可以存入 FAISS / Chroma。
- [ ] 可以根据 query 检索 top_k 文档。
- [ ] RAG Tool 可以被 Agent 调用。
- [ ] 报告中可以加入医学知识解释。

---

# 第四阶段：接入真实医学影像分割模型

## 1. 阶段目标

将 dummy segmentation 替换为真实模型推理工具。

候选模型：

```text
1. Inf-Net
2. nnUNet
3. 自己训练的医学影像分割模型
```

---

## 2. 分割工具流程

```text
输入 NIfTI / CT
 ↓
预处理
 ↓
模型推理
 ↓
生成 mask
 ↓
保存为 NIfTI
 ↓
返回 mask_path
```

---

## 3. SegmentationTool 真实版本输出

```json
{
  "status": "success",
  "model": "nnUNet / Inf-Net",
  "input_path": "data/samples/case001.nii.gz",
  "mask_path": "outputs/masks/case001_mask.nii.gz",
  "message": "segmentation completed"
}
```

---

## 4. 阶段四完成标准

- [ ] 可以读取真实医学影像。
- [ ] 可以调用真实分割模型。
- [ ] 可以保存 mask 结果。
- [ ] mask 与原图空间对齐。
- [ ] 输出结果可用于病灶统计。
- [ ] 工具失败时可以返回错误信息。

---

# 第五阶段：病灶统计分析

## 1. 阶段目标

根据分割结果计算结构化病灶指标。

---

## 2. 统计指标

| 指标                 | 说明                     |
| -------------------- | ------------------------ |
| lesion_voxel_count   | 病灶体素数量             |
| lesion_volume_mm3    | 病灶体积                 |
| lesion_ratio         | 病灶占整个图像或肺区比例 |
| max_slice_index      | 病灶最大层面             |
| max_slice_area       | 最大层面病灶面积         |
| bbox                 | 病灶外接框               |
| connected_components | 连通区域数量             |

---

## 3. 计算逻辑

```text
1. 读取 CT 图像
2. 读取 mask 图像
3. 提取 mask > 0 的区域
4. 根据 spacing 计算单个 voxel 体积
5. 计算病灶总体积
6. 统计每层病灶面积
7. 找出最大病灶层面
8. 输出结构化结果
```

---

## 4. 阶段五完成标准

- [ ] 可以读取 mask。
- [ ] 可以计算 voxel 数量。
- [ ] 可以根据 spacing 计算体积。
- [ ] 可以计算病灶占比。
- [ ] 可以定位最大病灶层面。
- [ ] 可以输出结构化 JSON 结果。

---

# 第六阶段：报告生成

## 1. 阶段目标

根据影像信息、分割结果、病灶统计和医学知识生成结构化报告。

---

## 2. 报告模板

```text
# 医学影像分析报告草稿

## 1. 影像基本信息

## 2. 模型分割结果

## 3. 病灶定量分析

## 4. 医学知识解释

## 5. 初步总结

## 6. 注意事项

本报告仅作为辅助分析，不替代医生诊断。
```

---

## 3. 报告生成方式

第一版：

```text
使用固定模板生成
```

第二版：

```text
使用 LLM 对结构化结果进行自然语言总结
```

第三版：

```text
结合 RAG 检索内容生成带依据的报告
```

---

## 4. 阶段六完成标准

- [ ] 可以根据 state 生成报告。
- [ ] 报告包含影像信息。
- [ ] 报告包含病灶统计。
- [ ] 报告包含医学解释。
- [ ] 报告包含医学安全提示。
- [ ] 报告可以保存到 `outputs/reports/`。

---

# 第七阶段：工程化增强

## 1. 阶段目标

提升项目稳定性、可维护性和展示价值。

---

## 2. 需要补充的工程模块

| 模块             | 作用         |
| ---------------- | ------------ |
| logger.py        | 记录运行日志 |
| config_loader.py | 读取配置文件 |
| .env             | 管理 API Key |
| requirements.txt | 管理依赖     |
| tests/           | 单元测试     |
| README.md        | 项目说明     |
| outputs/         | 保存结果     |

---

## 3. 配置文件设计

```text
configs/
├── model_config.yaml
├── rag_config.yaml
└── path_config.yaml
```

---

## 4. 日志内容

需要记录：

1. 用户输入。
2. 当前任务类型。
3. 调用的工具。
4. 工具输入参数。
5. 工具输出结果。
6. 错误信息。
7. 总运行时间。

---

## 5. 阶段七完成标准

- [ ] 支持配置文件。
- [ ] 支持日志记录。
- [ ] 支持错误捕获。
- [ ] 支持结果保存。
- [ ] 支持命令行运行。
- [ ] 支持基础测试。

---

# 第八阶段：项目展示与面试包装

## 1. 阶段目标

将项目整理成可以展示、可以讲解、可以写进简历的工程项目。

---

## 2. README 需要包含

1. 项目背景
2. 项目目标
3. 整体架构图
4. 技术路线
5. 目录结构
6. 安装方式
7. 运行示例
8. 核心功能
9. 实验结果
10. 后续优化方向

---

## 3. Demo 展示内容

建议展示 3 个案例：

1. 单独读取影像信息。
2. 运行分割并输出 mask。
3. 完整生成结构化报告。

---

## 4. 面试表达重点

可以这样描述项目：

```text
本项目实现了一个面向医学影像分析的多工具 Agent 系统。

系统支持影像信息读取、分割模型调用、病灶统计、医学知识库检索和结构化报告生成。

在架构设计上，我先手写了一个 Agent Loop，理解任务路由、工具调用和状态管理的底层逻辑。随后使用 LangGraph 将流程重构为 State、Node、Edge 形式，提高了复杂流程的可维护性和可扩展性。

每个工具都采用统一接口封装，支持后续替换真实模型、扩展新工具和增加错误恢复机制。
```

---

# 推荐开发进度

## Day 1：工具系统

任务：

- [ ] 创建项目目录。
- [ ] 实现 BaseTool。
- [ ] 实现 ToolRegistry。
- [ ] 实现 ImageInfoTool。
- [ ] 实现 SegmentationTool dummy。

目标：

```text
掌握 Tool 抽象和工具注册机制。
```

---

## Day 2：手写 Agent 主流程

任务：

- [ ] 实现 MedicalAgent。
- [ ] 实现 route_task。
- [ ] 实现 state 保存。
- [ ] 实现 final_answer 汇总。
- [ ] 打通命令行运行。

目标：

```text
掌握 Agent Loop、任务分流和工具调用流程。
```

---

## Day 3：病灶分析工具

任务：

- [ ] 实现 LesionAnalysisTool。
- [ ] 读取 mask。
- [ ] 计算病灶体素数。
- [ ] 计算病灶体积。
- [ ] 输出结构化统计结果。

目标：

```text
掌握医学影像 mask 统计分析。
```

---

## Day 4：报告生成

任务：

- [ ] 实现 ReportTool。
- [ ] 设计报告模板。
- [ ] 保存报告文件。
- [ ] 添加医学安全提示。

目标：

```text
掌握结构化结果到自然语言报告的转换。
```

---

## Day 5：LangGraph 重构

任务：

- [ ] 定义 AgentState。
- [ ] 编写 nodes.py。
- [ ] 编写 edges.py。
- [ ] 编写 build_graph.py。
- [ ] 使用 `graph.invoke()` 执行。

目标：

```text
掌握 State、Node、Edge、Conditional Edge。
```

---

## Day 6：加入 RAG

任务：

- [ ] 准备 Markdown 医学知识库。
- [ ] 实现文档加载。
- [ ] 实现文本切分。
- [ ] 实现向量数据库。
- [ ] 实现 Retriever。
- [ ] 实现 MedicalRAGTool。

目标：

```text
掌握医学知识库检索增强生成。
```

---

## Day 7：包装项目

任务：

- [ ] 整理 README。
- [ ] 添加运行示例。
- [ ] 添加架构图。
- [ ] 添加日志。
- [ ] 添加测试样例。
- [ ] 准备 Demo 说明。

目标：

```text
形成一个可展示的 Agent 项目。
```

---

# 项目完整目录结构

```text
medical_agent/
├── README.md
├── requirements.txt
├── .env.example
├── main.py
│
├── configs/
│   ├── model_config.yaml
│   ├── rag_config.yaml
│   └── path_config.yaml
│
├── agents/
│   ├── __init__.py
│   ├── medical_agent.py
│   └── prompts.py
│
├── graphs/
│   ├── __init__.py
│   ├── state.py
│   ├── nodes.py
│   ├── edges.py
│   └── build_graph.py
│
├── tools/
│   ├── __init__.py
│   ├── base_tool.py
│   ├── registry.py
│   ├── image_info_tool.py
│   ├── segmentation_tool.py
│   ├── lesion_analysis_tool.py
│   ├── rag_tool.py
│   └── report_tool.py
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── retriever.py
│
├── memory/
│   ├── __init__.py
│   ├── short_term_memory.py
│   └── long_term_memory.py
│
├── models/
│   ├── __init__.py
│   ├── llm_client.py
│   └── segmentation_model.py
│
├── schemas/
│   ├── __init__.py
│   ├── agent_state.py
│   ├── tool_schema.py
│   └── report_schema.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── file_utils.py
│   ├── image_utils.py
│   └── config_loader.py
│
├── data/
│   ├── samples/
│   ├── knowledge/
│   └── vector_db/
│
├── outputs/
│   ├── masks/
│   ├── reports/
│   └── logs/
│
└── tests/
    ├── test_tools.py
    ├── test_rag.py
    └── test_agent.py
```

---

# 当前优先级

## 最高优先级

1. 先手写 Agent MVP。
2. 打通工具调用流程。
3. 让 state 能完整记录中间结果。
4. 让 final_answer 能输出结构化结果。

---

## 暂时不要优先做

1. 不要一开始就接复杂真实模型。
2. 不要一开始就做复杂多 Agent。
3. 不要一开始就处理完整 DICOM 流程。
4. 不要一开始就追求前端界面。
5. 不要一开始就过度封装。

---

# 总结

本项目的核心路线是：

```text
先手写 Agent，理解底层机制；
再用 LangGraph 重构，提升工程化能力；
然后加入 RAG 和医学影像模型，形成完整医疗 Agent 项目。
```

最终目标不是简单调用大模型，而是构建一个具备以下能力的医学影像 Agent：

1. 能理解用户任务。
2. 能选择合适工具。
3. 能执行医学影像分析。
4. 能检索医学知识。
5. 能生成结构化报告。
6. 能记录状态和错误。
7. 能扩展到真实工程项目。
