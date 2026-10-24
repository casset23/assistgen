# 测试说明

## 文档处理

```bash
python -m pip install loguru==0.7.3
python -m unittest discover -s tests -v
python scripts/chunk_mineru.py examples/mineru-demo --output data/demo-chunks.json
```

4 个测试方法覆盖：

- Markdown 处理与 JSON 保存，隔离 OCR 和 torch 依赖。
- 分块参数合法性，包含 8 组非法配置子用例。
- 完整段落保留，检查“并发限制”及等待队列说明进入最终 chunk。

CLI 示例输出见 [demo-chunks.json](examples/demo-chunks.json)。

## 前端

在 `frontend/DsAgentChat_web` 执行：

```bash
npm run type-check
npm run build
```

类型与构建检查通过。`/preview` 使用内置示例数据，禁用业务写入，支持独立查看知识库界面。
