# M8 回归与交付检查报告

- 检查时间：2026-11-08
- 分支：`feat/m8.1-retrieval-foundation`
- 检查范围：M8.1–M8.5 全量代码、测试、Demo、评估与文档

## 1. 全量回归

| 项 | 命令 | 结果 |
|---|---|---|
| 单元测试 | `py -3.14 -m pytest -q` | **228 passed / 2 skipped**，14 warnings，8.36s |
| M8 Demo | `py -3.14 -m examples.demo_travel_memory_rag --out-dir demo_output` | 退出码 0 |
| RAG 评估 | `py -3.14 scripts/eval_rag.py --backend hashing --out ...` | 退出码 0，两张表生成 |
| 完整评估报告 | `py -3.14 scripts/eval_rag.py --out docs/m8-memory-rag-eval.md` | bm25 + hashing 双后端，报告 714 行 |

2 个 skip 为既有的真实 API 集成测试（无密钥时跳过），与 M8 无关。

Demo 产物（`demo_output/`，已在 `.gitignore` 中）：

- `knowledge/` 4 篇知识文档，共 18 个 chunk
- `uploads/` PDF / DOCX / HTML(URL 落盘) 各 1 份
- `memory.json` / `memory.sqlite3` 长期记忆（512 维 hashing）
- `rag.sqlite3` 向量库、`rag_eval.md` 带/不带 RAG 对照
- `trace.jsonl` 三范式轨迹（`memory.build` / `memory.write` / `rag.answer` 事件齐全）

## 2. 禁用项检查

对 `agentflow examples scripts tests` 检索：

```text
v1/embeddings | OpenAICompatibleEmbedder | neo4j | torch | langchain | llama_index |
sentence_transformers
```

结果：**0 命中**。本期未引入中转站 embedding、Neo4j 或重型框架依赖。

## 3. 依赖与 extras

`pyproject.toml` 主依赖仅 `pydantic / python-dotenv / openai`；可选能力全部在 extras：

| extra | 依赖 | 缺省行为 |
|---|---|---|
| `embed` | `fastembed` | 未安装时保持 `HashingEmbedder`，`FastEmbedEmbedder` 抛出带安装提示的 `RAGError` |
| `rag` | `pypdf` / `python-docx` | 未安装时仅 PDF/DOCX 不可用，txt/md/URL 正常 |
| `qdrant` | `qdrant-client` | 未安装时默认 SQLite 向量库 |

可选依赖未混入主依赖。

## 4. 文档与链接

| 文档 | 检查结果 |
|---|---|
| `README.md` | 快速开始命令均为当前可执行命令；D1–D8 路径全部存在 |
| `docs/architecture.md` | Mermaid 含 `MEM` / `RAG` 节点与 `MEM --> RAG`；PNG 已重新生成（1560×1000） |
| `docs/api-design.md` | 记忆与检索 API 章节完整，异常层级与实际代码一致 |
| `docs/m8-memory-rag-eval.md` | 召回表 + 回答对照表 + 逐题 + 原始 JSON，可重复生成 |
| `docs/m8-memory-rag-requirements.md` | 需求依据，随本期提交 |
| `docs/m8-task-prompts.md` | 任务清单与 DoD，随本期提交 |

## 5. 仓库卫生

- 无密钥、模型文件、SQLite 数据库、大文件被跟踪（`git ls-files` 检查通过）。
- `demo_output/`、`.env`、虚拟环境、缓存均在 `.gitignore`。
- M8 代码无任务范围外的重构。
- `docs/trajectory/*.jsonl` 的既有改动属于工作区遗留，未纳入 M8 提交。

## 6. 风险与限制

| 风险 | 说明 | 建议 |
|---|---|---|
| 召回上限 | `HashingEmbedder` 为词面特征，Recall@1 0.78；qa_05「外滩夜景」排第 2 且回答事实未命中 | 需要语义召回时安装 `.[embed]`，用 FastEmbed 复跑评估 |
| 未验证路径 | 本机未安装 `fastembed` / `qdrant-client`，其真实调用仅由单元测试的守卫逻辑覆盖 | 交付环境安装后跑一次 `--backend fastembed` 与 Qdrant 集成测试 |
| 扫描版 PDF | 无 OCR，提取为空时给出明确错误 | 后续按需引入 OCR，不在本期 |
| `.doc` / 旧版 Word | 明确拒绝并提示转 `.docx` | 用户侧转换即可 |
| Python 3.14 | FastEmbed 对 3.14 的 wheel 支持未验证 | 如遇安装失败，保持 hashing 后端 |

## 7. 结论

- 通过项：全量测试、Demo、评估可复现；禁用项 0 命中；extras 正确；文档链接可用；无敏感/大文件入库。
- 失败项：无。
- 建议下一步：如需更高召回，安装 `.[embed]` 后重跑评估并对比；在具备 Qdrant 凭据的环境补一次远程向量库集成验证。
