# 验证记录

本轮验证日期：2026-10-24。

- Python unittest：4 个方法通过，非法配置测试内包含 8 组子用例。
- 合成 Markdown CLI：读取公开示例并保存 JSON，实际输出随 docs/examples 提供。
- 前端：`npm run type-check` 和 `npm run build` 通过。
- 浏览器：直接访问 `/preview`，可见合成资料说明，新建与用于对话按钮禁用，上传入口隐藏；数据初始化分支不调用 ApiService。
- 回归修复先复现失败：非法切分参数未被拒绝；“并发限制”完整段落在 chunk 中消失。

未验证：完整 MySQL/Redis/模型集成、PDF/OCR、真实检索/问答效果、完整安全边界。
示例页码只有 Markdown 估计含义。没有用截图中的合成状态作为真实索引或评测证据。
