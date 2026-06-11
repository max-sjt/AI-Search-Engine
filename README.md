# AI Search Engine

一个基于 C++ 实现的轻量级语义搜索与 RAG 系统。项目从零实现了向量检索核心，包括 Brute Force、KD-Tree、HNSW 三种搜索方式，并在此基础上接入可切换的 AI Provider，支持本地 Ollama 和云端 Qwen/DashScope API。

这个项目适合作为 AI + C++ 工程实践项目：代码规模不大，但覆盖了向量数据库、RAG、MCP-style 工具接口、评测 harness、前端可视化和多模型后端抽象。

## 核心功能

| 功能 | 说明 |
|---|---|
| 向量检索 | 支持 Brute Force、KD-Tree、HNSW 三种算法 |
| 距离度量 | 支持 Cosine、Euclidean、Manhattan |
| HNSW 索引 | C++ 实现分层小世界图，用于近似最近邻搜索 |
| 文档语义搜索 | 文档自动切分为 chunk，并生成 embedding 后写入索引 |
| RAG 问答 | 先检索相关文档片段，再调用生成模型回答 |
| 多 AI 后端 | 默认使用 Ollama，也可切换到 Qwen/DashScope API |
| MCP-style 工具 | 提供 `search_docs`、`list_sources`、`ask_doc` 工具接口 |
| 评测接口 | 提供 retrieval recall@k 与 latency 评测接口 |
| Web UI | 支持搜索、算法对比、文档插入、RAG 问答、二维可视化 |

## 项目结构

```text
AI-Search-Engine/
├── src/
│   ├── ai/          # AI Provider 抽象、Ollama、Qwen/DashScope
│   ├── rag/         # 文档存储、文本切分、检索评测
│   ├── server/      # HTTP API、RAG 编排、MCP-style 工具
│   ├── util/        # JSON 工具函数
│   ├── vector/      # HNSW、KD-Tree、Brute Force 向量检索
│   └── main.cpp     # 服务入口
├── web/             # 前端页面
├── tests/           # smoke test
├── third_party/     # 第三方单头文件依赖
└── README.md
```

## 架构说明

```text
用户文档 / 用户问题
        |
        v
AI Provider embedding 模型
        |
        v
C++ HNSW 文档索引
        |
        v
Top-K 语义检索结果
        |
        v
AI Provider generation 模型
        |
        v
RAG 答案
```

项目中 C++ 负责：

- HTTP 服务与 REST API
- 文档切分、存储和重新加载
- Brute Force / KD-Tree / HNSW 向量检索
- RAG 流程编排
- MCP-style 工具接口
- retrieval eval harness

AI Provider 负责：

- 将文本转换为 embedding
- 根据检索上下文生成回答

## AI Provider 配置

默认后端是 Ollama：

```powershell
$env:AI_PROVIDER="ollama"
$env:OLLAMA_EMBED_MODEL="nomic-embed-text"
$env:OLLAMA_GEN_MODEL="llama3.2:1b"
ollama serve
```

切换到 Qwen/DashScope：

```powershell
$env:AI_PROVIDER="qwen"
$env:QWEN_API_KEY="your-api-key"
$env:QWEN_EMBED_MODEL="text-embedding-v4"
$env:QWEN_GEN_MODEL="qwen-plus"
```

Qwen 使用 HTTPS API，需要启用 `CPPHTTPLIB_OPENSSL_SUPPORT` 并链接 OpenSSL：

```powershell
g++ -std=c++17 -O2 -DCPPHTTPLIB_OPENSSL_SUPPORT src/main.cpp -o db_qwen.exe -static -static-libgcc -static-libstdc++ -lssl -lcrypto -lcrypt32 -lws2_32
.\db_qwen.exe
```

注意：不同 embedding 模型的向量维度可能不同。切换 embedding 模型后，建议删除或重建 `data/documents.tsv`。

## Windows 构建

推荐使用 MSYS2 UCRT64 的 g++。

```powershell
g++ -std=c++17 -O2 src/main.cpp -o db_refactored.exe -static -static-libgcc -static-libstdc++ -lws2_32
g++ -std=c++17 -O2 tests/smoke_test.cpp -o smoke_test.exe -static -static-libgcc -static-libstdc++ -lws2_32
```

运行测试：

```powershell
.\smoke_test.exe
```

启动服务：

```powershell
.\db_refactored.exe
```

浏览器打开：

```text
http://localhost:8080
```

## 快速开始

1. 安装 Git。
2. 安装 MSYS2，并将 `C:\msys64\ucrt64\bin` 加入 PATH。
3. 安装 Ollama。
4. 拉取模型：

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

5. 克隆仓库：

```powershell
git clone https://github.com/max-sjt/AI-Search-Engine.git
cd AI-Search-Engine
```

6. 编译并运行：

```powershell
g++ -std=c++17 -O2 src/main.cpp -o db_refactored.exe -static -static-libgcc -static-libstdc++ -lws2_32
.\db_refactored.exe
```

启动成功后会看到类似输出：

```text
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
0 persisted document chunks | MCP+Eval endpoints enabled
AI provider: ollama ONLINE
  embed model: nomic-embed-text  gen model: llama3.2:1b
```

## 使用方式

### 1. Demo Vector Search

该页面使用内置的 16 维 demo 向量，方便观察不同搜索算法的行为。

可以选择：

- HNSW
- KD-Tree
- Brute Force
- Cosine / Euclidean / Manhattan 距离

同时可以运行 benchmark，对比三种算法的检索耗时。

### 2. Documents

将文本粘贴到页面后，系统会：

1. 按 250 词左右切分为重叠 chunk。
2. 调用当前 AI Provider 生成 embedding。
3. 将 chunk 和向量写入本地文档索引。
4. 持久化到 `data/documents.tsv`。

### 3. Ask AI

RAG 问答流程：

1. 用户问题生成 embedding。
2. HNSW 在文档索引中检索 Top-K chunk。
3. 将检索结果拼入 prompt。
4. 调用生成模型得到回答。
5. 返回答案和引用上下文。

## REST API

### 向量检索接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | KNN 搜索 |
| `POST` | `/insert` | 插入 demo 向量 |
| `DELETE` | `/delete/:id` | 删除 demo 向量 |
| `GET` | `/items` | 查看所有 demo 向量 |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | 对比三种算法耗时 |
| `GET` | `/hnsw-info` | 查看 HNSW 图结构信息 |
| `GET` | `/stats` | 查看基础统计信息 |

### 文档与 RAG 接口

| 方法 | 路径 | Body | 说明 |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | 插入文档并生成 embedding |
| `GET` | `/doc/list` | 无 | 查看已插入文档 |
| `DELETE` | `/doc/delete/:id` | 无 | 删除文档 chunk |
| `POST` | `/doc/search` | `{"question":"...","k":3}` | 只检索，不生成回答 |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG 问答 |
| `GET` | `/status` | 无 | 查看 AI Provider 和模型状态 |

### MCP-style 工具接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/mcp/tools` | 查看可用工具 |
| `POST` | `/mcp/call` | 调用 `search_docs`、`list_sources`、`ask_doc` |

### 评测接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/eval/retrieval` | 单条检索评测，返回 recall@k 和 latency |

## 示例请求

向量搜索：

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

RAG 问答：

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

查看 Provider 状态：

```powershell
curl http://localhost:8080/status
```

## 算法说明

### Brute Force

遍历所有向量并计算距离，结果准确，但复杂度为 `O(N*d)`。适合作为正确性基准。

### KD-Tree

按维度递归划分空间，低维场景下可以减少搜索范围。但在高维向量中，剪枝效果会明显下降。

### HNSW

HNSW 是分层小世界图索引。搜索从稀疏高层开始，快速接近目标区域，再下降到底层做更精细的候选扩展。相比 Brute Force，HNSW 更适合高维近似最近邻检索。

## 工程亮点

- 使用 C++ 实现核心向量检索算法，不依赖现成向量数据库。
- 将 AI 能力抽象为 `AiProvider`，支持本地 Ollama 和云端 Qwen/DashScope。
- 具备完整 RAG 流程：chunk、embedding、index、retrieve、generate、context 返回。
- 提供 MCP-style 工具接口，方便包装为 Agent 工具能力。
- 提供 retrieval eval harness，可评估 recall@k 和延迟。
- 前端可视化展示向量分布、检索结果和算法耗时。
- 目录按功能拆分，便于继续扩展模型后端、索引算法和评测集。

## 常见问题

| 问题 | 解决方式 |
|---|---|
| 页面显示 AI Provider offline | 访问 `/status` 查看具体提示，确认 Ollama 或 Qwen 配置是否正确 |
| `g++: command not found` | 检查 `C:\msys64\ucrt64\bin` 是否加入 PATH |
| `undefined reference to WSA...` | 编译时需要加 `-lws2_32` |
| 8080 端口被占用 | 使用 `netstat -ano \| findstr 8080` 找到进程并结束 |
| Ollama 首次 embedding 很慢 | 首次加载或下载模型需要等待 |
| 切换 embedding 模型后检索异常 | 删除或重建 `data/documents.tsv` |
| Qwen provider 无法请求 | 确认 `QWEN_API_KEY`、网络代理和 OpenSSL 构建参数 |

## 简历描述参考

可以写成：

> 基于 C++ 实现轻量级 RAG 语义搜索系统，从零实现 Brute Force、KD-Tree、HNSW 向量检索算法，支持文档 chunk、embedding、HNSW 索引、Top-K 检索、RAG 生成回答，并抽象 AI Provider 以兼容本地 Ollama 与云端 Qwen/DashScope API；同时提供 REST API、MCP-style 工具接口、retrieval eval harness 和 Web 可视化界面。

## License

MIT
