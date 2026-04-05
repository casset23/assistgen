# AssistGen

一个基于 `Vue 3 + FastAPI` 的全栈 AI 助手项目，目前包含：
- 通用聊天
- 会话管理
- LangGraph 对话流程
- 文件上传
- 知识库问答探索
- 电商客服场景原型

## 技术栈

### 前端
- Vue 3
- TypeScript
- Vite
- Pinia
- Axios

### 后端
- FastAPI
- SQLAlchemy
- LangGraph
- DeepSeek / Ollama
- GraphRAG
- MySQL
- Redis
- Neo4j

## 项目结构

- `frontend/DsAgentChat_web`：前端项目
- `backend/deepseek_agent/llm_backend`：后端服务

## 本地启动

### 1. 前端

```bash
cd frontend/DsAgentChat_web
npm install
npm run dev
```

### 2. 后端

```bash
cd backend/deepseek_agent
pip install -r requirements.txt
cd llm_backend
python main.py
```

## 环境变量

请参考根目录下的 `.env.example` 配置本地环境变量。

## 当前阶段

当前仓库主要用于：
- AI 助手能力验证
- PDF / 知识库问答方向迭代
- 后续扩展为教材/书籍问答演示系统

