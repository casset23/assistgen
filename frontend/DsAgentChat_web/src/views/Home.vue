<template>
  <div class="page-shell">
    <aside class="left-rail">
      <div class="brand-card">
        <div class="brand-mark">A</div>
        <div>
          <p class="eyebrow">智能资料助手</p>
          <h1>AssistGen</h1>
        </div>
      </div>

      <nav class="nav-panel">
        <RouterLink to="/chat" class="nav-item" active-class="active">
          <span>对话工作台</span>
          <small>问答 / 检索 / 推理</small>
        </RouterLink>
        <RouterLink to="/knowledge" class="nav-item" active-class="active">
          <span>知识空间</span>
          <small>教材 / 讲义 / 资料上传</small>
        </RouterLink>
      </nav>

      <section class="conversation-panel">
        <div class="section-title-row">
          <span>最近会话</span>
          <button class="ghost-btn" @click="startNewChat">新建</button>
        </div>
        <div class="conversation-list">
          <button
            v-for="conversation in conversationStore.conversations"
            :key="conversation.id"
            class="conversation-item"
            :class="{ active: conversation.id === conversationStore.currentConversationId }"
            @click="handleConversationClick(conversation.id)"
          >
            <span class="conversation-title">{{ conversation.title }}</span>
            <span class="conversation-time">{{ formatTime(conversation.created_at) }}</span>
          </button>
        </div>
      </section>

      <section class="user-card">
        <div class="user-meta">
          <div class="avatar">{{ userInitial }}</div>
          <div>
            <div class="user-name">{{ userStore.username || '未登录用户' }}</div>
            <div class="user-email">{{ userStore.email || '暂无邮箱信息' }}</div>
          </div>
        </div>
        <button class="logout-btn" @click="handleLogout">退出登录</button>
      </section>
    </aside>

    <main class="workspace">
      <header class="workspace-header">
        <div>
          <p class="eyebrow">当前模式</p>
          <h2>{{ currentModeLabel }}</h2>
        </div>
        <div class="header-badges">
          <span class="header-badge active">{{ activeKnowledgeBase?.name || '未选择知识库' }}</span>
          <span class="header-badge">{{ isDeepThinking ? '深度思考' : '标准回答' }}</span>
          <span class="header-badge">{{ isSearching ? '联网搜索已启用' : '本地模式' }}</span>
        </div>
      </header>

      <section class="hero-card" v-if="messages.length === 0">
        <div class="hero-copy">
          <p class="eyebrow">知识驱动问答</p>
          <h3>把教材、论文、讲义和课题资料接进同一个工作台。</h3>
          <p>
            当前已经接通后端会话接口与教材知识库接口。你可以直接开始问答，也可以先去知识空间上传和管理资料。
          </p>
        </div>
        <div class="hero-metrics">
          <div class="metric-card">
            <span>当前知识库</span>
            <strong>{{ activeKnowledgeBase?.documents.length || 0 }}</strong>
            <small>份资料</small>
          </div>
          <div class="metric-card">
            <span>回答模式</span>
            <strong>{{ isRagMode ? 'RAG' : 'Chat' }}</strong>
            <small>{{ currentModeLabel }}</small>
          </div>
          <div class="metric-card action" @click="router.push('/knowledge')">
            <span>进入知识空间</span>
            <strong>管理资料</strong>
            <small>上传 / 查看 / 切换</small>
          </div>
        </div>
      </section>

      <section class="chat-board">
        <div class="message-stream" ref="messageStreamRef">
          <template v-if="messages.length">
            <article
              v-for="message in messages"
              :key="message.id"
              class="message-card"
              :class="message.role"
            >
              <div class="message-topline">
                <span class="message-role">{{ message.role === 'user' ? '你' : 'AssistGen' }}</span>
                <span class="message-mode" v-if="message.role === 'assistant'">{{ currentModeLabel }}</span>
              </div>
              <div class="message-body" v-html="renderMarkdown(message.content)"></div>

              <div v-if="message.sources?.length" class="source-bar">
                <button class="source-entry" type="button" @click="toggleMessageSources(message)">
                  <span>引用来源（{{ message.sources.length }}）</span>
                  <span>{{ message.showSources ? '收起' : '展开' }}</span>
                </button>

                <div v-if="message.showSources" class="source-grid compact">
                  <div v-for="(source, sourceIndex) in message.sources" :key="source.chunk_id || sourceIndex" class="source-card">
                    <div class="source-head">
                      <span>{{ source.title || source.section || `片段 ${sourceIndex + 1}` }}</span>
                      <span v-if="source.page">P.{{ source.page }}</span>
                    </div>
                    <p class="source-preview full-preview">{{ source.content || source.snippet || source.original_snippet || '暂无片段内容' }}</p>
                  </div>
                </div>
              </div>
            </article>
          </template>
          <div v-else class="empty-chat-hint">
            <p>试试这些问题：</p>
            <div class="prompt-list">
              <button @click="fillPrompt('墨卡托海图有什么特点？')">墨卡托海图有什么特点？</button>
              <button @click="fillPrompt('导航的定义是什么？')">导航的定义是什么？</button>
              <button @click="fillPrompt('请概括本章关于海图投影的关键区别')">请概括本章关于海图投影的关键区别</button>
            </div>
          </div>
        </div>

        <div class="composer-card">
          <div class="mode-switches">
            <button class="mode-chip" :class="{ active: isRagMode }" @click="isRagMode = !isRagMode">知识库问答</button>
            <button class="mode-chip" :class="{ active: isDeepThinking }" @click="isDeepThinking = !isDeepThinking">深度思考</button>
            <button class="mode-chip" :class="{ active: isSearching }" @click="isSearching = !isSearching">联网搜索</button>
            <button class="mode-chip secondary" @click="router.push('/knowledge')">管理知识库</button>
          </div>

          <div class="active-kb-bar">
            <div>
              <span class="eyebrow">已连接知识空间</span>
              <strong>{{ activeKnowledgeBase?.name || '暂无可用知识库' }}</strong>
            </div>
            <select v-model="activeKnowledgeBaseId" @change="handleKnowledgeBaseChange">
              <option v-for="base in knowledgeBases" :key="base.id" :value="base.id">
                {{ base.name }}
              </option>
            </select>
          </div>

          <textarea
            v-model="userInput"
            class="composer-input"
            placeholder="输入你的问题，支持教材问答、普通对话与联网搜索..."
            @keydown.enter.exact.prevent="sendMessage"
          />

          <div class="composer-footer">
            <div class="footer-tip">内容由模型生成，建议结合知识片段进行判断。</div>
            <button class="send-button" :disabled="isSending || !userInput.trim()" @click="sendMessage">
              {{ isSending ? '生成中...' : '发送消息' }}
            </button>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'
import { ApiService, AuthService, type BookRagSource, type KnowledgeBaseItem } from '../services/api'
import { useConversationStore } from '../stores/conversation'
import { useUserStore } from '../stores/user'
import type { ChatMessage, KnowledgeSource } from '../types'

interface UiMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: KnowledgeSource[]
  showSources?: boolean
}

const router = useRouter()
const conversationStore = useConversationStore()
const userStore = useUserStore()
const md = new MarkdownIt({ breaks: true, linkify: true, typographer: true })

const userInput = ref('')
const messages = ref<UiMessage[]>([])
const isDeepThinking = ref(false)
const isSearching = ref(false)
const isRagMode = ref(true)
const isSending = ref(false)
const messageStreamRef = ref<HTMLElement | null>(null)
const knowledgeBases = ref<KnowledgeBaseItem[]>([])
const activeKnowledgeBaseId = ref('')

const activeKnowledgeBase = computed(() =>
  knowledgeBases.value.find(base => base.id === activeKnowledgeBaseId.value) || knowledgeBases.value[0] || null
)

const currentModeLabel = computed(() => {
  if (isRagMode.value) return '知识库增强问答'
  if (isSearching.value) return '联网搜索问答'
  if (isDeepThinking.value) return '深度思考模式'
  return '标准对话模式'
})

const userInitial = computed(() => (userStore.username || 'A').slice(0, 1).toUpperCase())

const renderMarkdown = (content: string) => DOMPurify.sanitize(md.render(content || ''))

const scrollToBottom = async () => {
  await nextTick()
  messageStreamRef.value?.scrollTo({ top: messageStreamRef.value.scrollHeight, behavior: 'smooth' })
}

const formatTime = (value: string) => {
  const date = new Date(value)
  return new Intl.DateTimeFormat('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date)
}

const createMessageId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

const normalizeSources = (rawSources?: BookRagSource[]): KnowledgeSource[] => {
  if (!rawSources?.length) return []
  return rawSources.slice(0, 4).map((item, index) => ({
    chunk_id: item.chunk_id,
    title:
      item.title ||
      item.metadata?.section_title ||
      item.metadata?.book_name ||
      `片段 ${index + 1}`,
    section: item.section || item.metadata?.section_title,
    content:
      item.content ||
      item.snippet ||
      item.original_snippet ||
      item.metadata?.text ||
      item.metadata?.content ||
      '暂无片段内容',
    snippet: item.snippet,
    original_snippet: item.original_snippet,
    page:
      item.page ||
      item.metadata?.page ||
      item.metadata?.page_start ||
      item.metadata?.start_page,
    score: item.score,
    rerank_score: item.rerank_score
  }))
}

const toggleMessageSources = (message: UiMessage) => {
  message.showSources = !message.showSources
}

const syncKnowledgeBases = () => {
  knowledgeBases.value = ApiService.getKnowledgeBases()
  activeKnowledgeBaseId.value = ApiService.getActiveKnowledgeBaseId()
}

const handleKnowledgeBaseChange = () => {
  ApiService.setActiveKnowledgeBase(activeKnowledgeBaseId.value)
}

const startNewChat = async () => {
  await conversationStore.createNewConversation()
  messages.value = []
}

const handleConversationClick = async (conversationId: number) => {
  await conversationStore.loadConversationMessages(conversationId)
  messages.value = conversationStore.currentMessages.map(message => ({
    id: String(message.id),
    role: message.sender,
    content: message.content,
    showSources: false
  }))
  await scrollToBottom()
}

const fillPrompt = (prompt: string) => {
  userInput.value = prompt
}

const ensureConversation = async () => {
  if (!conversationStore.currentConversationId) {
    await conversationStore.createNewConversation()
  }
  return conversationStore.currentConversationId as number
}

const sendStandardStream = async (payload: ChatMessage[], mode: 'chat' | 'reason' | 'search') => {
  const conversationId = await ensureConversation()
  const reader = await (mode === 'reason'
    ? ApiService.reason(payload, conversationId)
    : mode === 'search'
      ? ApiService.search(payload, conversationId)
      : ApiService.chat(payload, conversationId))

  if (!reader) throw new Error('未获取到流式响应')

  const assistantMessage: UiMessage = { id: createMessageId(), role: 'assistant', content: '' }
  messages.value.push(assistantMessage)

  await ApiService.handleChatStream(reader, chunk => {
    if (chunk.type === 'response' || !chunk.type) {
      assistantMessage.content = chunk.content
    }
  })
}

const sendBookRagMessage = async (prompt: string) => {
  const activeBase = activeKnowledgeBase.value
  if (!activeBase) throw new Error('请先选择知识库')

  const result = await ApiService.bookRagGenerate({
    query: prompt,
    index_id: activeBase.indexId,
    top_k: 5,
    document_type: activeBase.category === '教材' ? 'textbook' : 'general'
  })

  console.log('[BookRAG result]', result)

  const normalizedSources = normalizeSources(
    result.references || result.sources || result.retrieved_chunks || result.chunks || result.retrieval_results
  )
  console.log('[BookRAG normalizedSources]', normalizedSources)

  const assistantMessage: UiMessage = {
    id: createMessageId(),
    role: 'assistant',
    content: result.answer || result.response || result.content || '未返回答案',
    sources: normalizedSources,
    showSources: false
  }
  messages.value.push(assistantMessage)
}

const sendMessage = async () => {
  const prompt = userInput.value.trim()
  if (!prompt || isSending.value) return

  isSending.value = true
  messages.value.push({ id: createMessageId(), role: 'user', content: prompt, showSources: false })
  userInput.value = ''
  await scrollToBottom()

  try {
    const payload: ChatMessage[] = [{ role: 'user', content: prompt }]

    if (isRagMode.value) {
      await sendBookRagMessage(prompt)
    } else if (isSearching.value) {
      await sendStandardStream(payload, 'search')
    } else if (isDeepThinking.value) {
      await sendStandardStream(payload, 'reason')
    } else {
      await sendStandardStream(payload, 'chat')
    }

    await conversationStore.loadUserConversations()
  } catch (error: any) {
    messages.value.push({
      id: createMessageId(),
      role: 'assistant',
      content: `请求失败：${error?.message || '未知错误'}`,
      showSources: false
    })
  } finally {
    isSending.value = false
    await scrollToBottom()
  }
}

const handleLogout = async () => {
  await AuthService.logout()
}

watch(
  () => conversationStore.currentMessages,
  currentMessages => {
    if (!currentMessages.length) return
    messages.value = currentMessages.map(message => ({
      id: String(message.id),
      role: message.sender,
      content: message.content,
      showSources: false
    }))
  },
  { deep: true }
)

onMounted(async () => {
  syncKnowledgeBases()

  try {
    const userInfo = await AuthService.getUserInfo()
    userStore.setUserInfo(userInfo)
  } catch (error) {
    console.error('获取用户信息失败', error)
  }

  try {
    await conversationStore.loadUserConversations()
    if (conversationStore.isNewConversation) {
      await conversationStore.createNewConversation()
    }
  } catch (error) {
    console.error('初始化会话失败', error)
  }
})
</script>

<style scoped>
:global(body) {
  margin: 0;
  font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
  background:
    radial-gradient(circle at top left, rgba(84, 126, 255, 0.16), transparent 28%),
    radial-gradient(circle at top right, rgba(0, 204, 255, 0.12), transparent 24%),
    #07111f;
  color: #e8f1ff;
}

.page-shell {
  height: 100vh;
  display: grid;
  grid-template-columns: 320px 1fr;
  background: linear-gradient(180deg, rgba(7, 17, 31, 0.92), rgba(3, 10, 20, 0.98));
  overflow: hidden;
}

.left-rail {
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 28px 22px;
  border-right: 1px solid rgba(150, 185, 255, 0.12);
  background: rgba(5, 12, 24, 0.9);
  backdrop-filter: blur(20px);
}

.brand-card,
.nav-panel,
.conversation-panel,
.user-card,
.hero-card,
.chat-board,
.composer-card,
.metric-card,
.message-card,
.source-card,
.active-kb-bar {
  border: 1px solid rgba(133, 168, 255, 0.14);
  background: rgba(12, 22, 38, 0.72);
  box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
}

.brand-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 18px;
  border-radius: 24px;
}

.brand-mark {
  width: 48px;
  height: 48px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, #78dce8, #7f7dff);
  color: #03101c;
  font-weight: 800;
  font-size: 22px;
}

.eyebrow {
  margin: 0 0 6px;
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #7cb8ff;
}

.brand-card h1,
.workspace-header h2,
.hero-copy h3 {
  margin: 0;
}

.nav-panel,
.conversation-panel,
.user-card {
  padding: 18px;
  border-radius: 24px;
}

.nav-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-decoration: none;
  color: #b8cfff;
  padding: 14px 16px;
  border-radius: 18px;
  transition: 0.25s ease;
}

.nav-item.active,
.nav-item:hover,
.conversation-item:hover,
.conversation-item.active,
.mode-chip.active,
.send-button:hover,
.metric-card.action:hover {
  background: linear-gradient(135deg, rgba(87, 147, 255, 0.22), rgba(38, 223, 255, 0.16));
  color: #f5fbff;
}

.section-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

.ghost-btn,
.logout-btn,
.send-button,
.mode-chip,
.conversation-item,
select {
  border: none;
  outline: none;
  cursor: pointer;
}

.ghost-btn,
.logout-btn,
.mode-chip,
select {
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  color: #d5e6ff;
}

.ghost-btn,
.logout-btn {
  padding: 10px 14px;
}

.conversation-panel {
  min-height: 0;
}

.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 320px;
  overflow: auto;
}

.conversation-item {
  text-align: left;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.03);
  color: #d8e7ff;
}

.conversation-title,
.message-role,
.user-name,
.metric-card strong {
  display: block;
  font-weight: 700;
}

.conversation-time,
.user-email,
.metric-card small,
.footer-tip,
.source-card p,
.message-mode {
  color: #89a5d6;
  font-size: 12px;
}

.user-card {
  margin-top: auto;
}

.user-meta {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 14px;
}

.avatar {
  width: 44px;
  height: 44px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, rgba(120, 220, 232, 0.85), rgba(127, 125, 255, 0.85));
  color: #04101d;
  font-weight: 800;
}

.workspace {
  height: 100vh;
  min-height: 0;
  overflow: hidden;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.workspace-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.header-badges {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.header-badge {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  color: #dbe9ff;
  font-size: 13px;
}

.header-badge.active {
  background: linear-gradient(135deg, rgba(120, 220, 232, 0.2), rgba(127, 125, 255, 0.22));
}

.hero-card {
  padding: 26px;
  border-radius: 28px;
  display: grid;
  grid-template-columns: 1.6fr 1fr;
  gap: 22px;
}

.hero-copy p:last-child {
  color: #acc6ee;
  line-height: 1.75;
}

.hero-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}

.metric-card {
  padding: 18px;
  border-radius: 22px;
}

.metric-card.action {
  cursor: pointer;
}

.chat-board {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  gap: 18px;
  border-radius: 28px;
  padding: 18px;
  min-height: 0;
  flex: 1;
  overflow: hidden;
}

.message-stream {
  overflow: auto;
  padding-right: 6px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.empty-chat-hint {
  margin: auto 0;
  text-align: center;
  color: #9db8e6;
}

.prompt-list {
  margin-top: 18px;
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.prompt-list button {
  padding: 12px 16px;
  border-radius: 999px;
  border: 1px solid rgba(124, 184, 255, 0.18);
  background: rgba(255, 255, 255, 0.04);
  color: #dceaff;
  cursor: pointer;
}

.message-card {
  padding: 20px;
  border-radius: 24px;
}

.message-card.user {
  align-self: flex-end;
  width: min(78%, 760px);
  background: linear-gradient(135deg, rgba(76, 119, 255, 0.24), rgba(26, 36, 64, 0.82));
}

.message-card.assistant {
  align-self: flex-start;
  width: min(86%, 860px);
}

.message-topline {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
}

.message-body {
  color: #edf4ff;
  line-height: 1.8;
}

.source-bar {
  margin-top: 18px;
}

.source-entry {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  border: 1px solid rgba(124, 184, 255, 0.18);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
  color: #dceaff;
  cursor: pointer;
}

.source-entry:hover {
  background: rgba(124, 184, 255, 0.14);
}

.source-grid.compact {
  margin-top: 12px;
}

.source-grid {
  margin-top: 18px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}

.source-card {
  padding: 14px;
  border-radius: 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.source-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
  color: #cfe2ff;
  font-size: 13px;
}

.source-preview {
  margin: 0;
  color: #89a5d6;
  font-size: 12px;
  line-height: 1.7;
}

.full-preview {
  display: block;
}

.composer-card {
  padding: 18px;
  border-radius: 24px;
}

.mode-switches {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.mode-chip {
  padding: 10px 14px;
}

.mode-chip.secondary {
  margin-left: auto;
}

.active-kb-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  border-radius: 18px;
  margin-bottom: 16px;
}

select {
  min-width: 220px;
  padding: 10px 12px;
}

.composer-input {
  width: 100%;
  min-height: 140px;
  resize: vertical;
  border: 1px solid rgba(132, 168, 255, 0.16);
  border-radius: 20px;
  background: rgba(5, 12, 24, 0.72);
  color: #eff6ff;
  padding: 16px 18px;
  box-sizing: border-box;
  font-size: 15px;
  line-height: 1.7;
}

.composer-footer {
  margin-top: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.send-button {
  padding: 12px 18px;
  border-radius: 16px;
  background: linear-gradient(135deg, #78dce8, #7f7dff);
  color: #03101c;
  font-weight: 800;
}

.send-button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

@media (max-width: 1200px) {
  .page-shell {
    grid-template-columns: 1fr;
  }

  .hero-card,
  .hero-metrics {
    grid-template-columns: 1fr;
  }
}
</style>
