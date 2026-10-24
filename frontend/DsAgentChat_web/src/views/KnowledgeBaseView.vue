<template>
  <div class="knowledge-page">
    <aside class="knowledge-sidebar">
      <div class="sidebar-top">
        <p class="eyebrow">知识空间</p>
        <h1>资料中心</h1>
        <button class="create-btn" :disabled="preview" @click="createKnowledgeBase">+ 新建知识库</button>
        <p v-if="preview" class="preview-note">只读预览 · 合成资料<br />不连接后端、不创建真实索引</p>
      </div>

      <div class="knowledge-base-list">
        <button
          v-for="base in knowledgeBases"
          :key="base.id"
          class="base-item"
          :class="[base.accent, { active: base.id === activeKnowledgeBaseId }]"
          @click="selectKnowledgeBase(base.id)"
        >
          <div>
            <strong>{{ base.name }}</strong>
            <p>{{ base.description }}</p>
          </div>
          <span>{{ base.documents.length }} 份资料</span>
        </button>
      </div>

      <RouterLink v-if="!preview" to="/chat" class="back-link">← 返回对话工作台</RouterLink>
    </aside>

    <main class="knowledge-main" v-if="activeKnowledgeBase">
      <header class="knowledge-header">
        <div>
          <p class="eyebrow">当前知识库</p>
          <h2>{{ activeKnowledgeBase.name }}</h2>
          <p class="desc">{{ activeKnowledgeBase.description }}</p>
        </div>
        <div class="header-actions">
          <label v-if="!preview" class="upload-btn">
            上传资料
            <input type="file" multiple hidden @change="handleFileChange" />
          </label>
          <button class="primary-btn" :disabled="preview" @click="goToChat">用于对话</button>
        </div>
      </header>

      <section class="overview-grid">
        <div class="overview-card">
          <span>资料总数</span>
          <strong>{{ activeKnowledgeBase.documents.length }}</strong>
          <small>教材 / 讲义 / 论文 / 报告</small>
        </div>
        <div class="overview-card">
          <span>当前索引</span>
          <strong>{{ activeKnowledgeBase.indexId || '尚未建立' }}</strong>
          <small>{{ preview ? '展示数据，未执行向量化' : '上传资料后由后端生成索引' }}</small>
        </div>
        <div class="overview-card">
          <span>最后更新</span>
          <strong>{{ formatDate(activeKnowledgeBase.updatedAt) }}</strong>
          <small>上传后可立即切换使用</small>
        </div>
      </section>

      <section class="document-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">资料列表</p>
            <h3>支持教材与各种研究资料</h3>
          </div>
          <div class="panel-tip">当前为前端初版，已先打通上传入口与会话知识库切换。</div>
        </div>

        <div v-if="activeKnowledgeBase.documents.length" class="document-grid">
          <article v-for="doc in activeKnowledgeBase.documents" :key="doc.id" class="document-card">
            <div class="doc-meta">
              <span class="doc-type">{{ doc.type.toUpperCase() }}</span>
              <span class="doc-status" :class="doc.status">{{ statusLabelMap[doc.status] }}</span>
            </div>
            <h4>{{ doc.name }}</h4>
            <p>上传时间：{{ formatDate(doc.uploadedAt) }}</p>
            <p>大小：{{ formatSize(doc.size) }}</p>
            <p v-if="doc.indexId">索引：{{ doc.indexId }}</p>
          </article>
        </div>

        <div v-else class="empty-state">
          <h4>还没有资料</h4>
          <p>先上传教材、讲义、论文或课题组文档，后续问答时即可作为知识来源。</p>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { ApiService, type KnowledgeBaseItem, type KnowledgeDocument } from '../services/api'

const router = useRouter()
const props = withDefaults(defineProps<{ preview?: boolean }>(), { preview: false })
const knowledgeBases = ref<KnowledgeBaseItem[]>(props.preview ? [{
  id: 'preview-space', name: 'RAG 工程资料库', category: '工程', accent: 'cyan',
  description: '从文档结构到检索证据：教材、设计笔记与评测问题的统一入口。',
  indexId: 'preview-only', updatedAt: '2026-01-01T08:00:00Z',
  documents: [
    { id: 'preview-1', name: '示例教材：文档切分与章节上下文', type: 'md', size: 4200, status: 'ready', uploadedAt: '2026-01-01T08:00:00Z' },
    { id: 'preview-2', name: '示例笔记：混合检索与候选融合', type: 'md', size: 2800, status: 'ready', uploadedAt: '2026-01-01T08:00:00Z' },
    { id: 'preview-3', name: '示例问题：引用可追溯性检查', type: 'json', size: 1600, status: 'processing', uploadedAt: '2026-01-01T08:00:00Z' }
  ]
}] : ApiService.getKnowledgeBases())
const activeKnowledgeBaseId = ref(props.preview ? 'preview-space' : ApiService.getActiveKnowledgeBaseId())

const activeKnowledgeBase = computed(
  () => knowledgeBases.value.find(base => base.id === activeKnowledgeBaseId.value) || knowledgeBases.value[0] || null
)

const statusLabelMap: Record<KnowledgeDocument['status'], string> = {
  ready: '已就绪',
  uploading: '上传中',
  processing: '处理中',
  error: '失败'
}

const refreshKnowledgeBases = () => {
  knowledgeBases.value = ApiService.getKnowledgeBases()
  activeKnowledgeBaseId.value = ApiService.getActiveKnowledgeBaseId()
}

const selectKnowledgeBase = (id: string) => {
  activeKnowledgeBaseId.value = id
  if (!props.preview) ApiService.setActiveKnowledgeBase(id)
}

const createKnowledgeBase = () => {
  if (props.preview) return
  const count = knowledgeBases.value.length + 1
  ApiService.createKnowledgeBase({
    name: `知识库 ${count}`,
    description: '适合上传教材、课程讲义、实验指导、论文与课题资料。',
    category: '综合资料',
    accent: count % 2 === 0 ? 'violet' : 'cyan'
  })
  refreshKnowledgeBases()
}

const formatDate = (value: string) => {
  const date = new Date(value)
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date)
}

const formatSize = (size: number) => {
  if (!size) return '系统内置'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(2)} MB`
}

const handleFileChange = async (event: Event) => {
  if (props.preview) return
  const files = (event.target as HTMLInputElement).files
  const base = activeKnowledgeBase.value
  const userId = localStorage.getItem('user_id')
  if (!files?.length || !base || !userId) return

  for (const file of Array.from(files)) {
    const localDoc: KnowledgeDocument = {
      id: `doc-${Date.now()}-${file.name}`,
      name: file.name,
      type: file.name.split('.').pop() || 'file',
      size: file.size,
      status: 'uploading',
      uploadedAt: new Date().toISOString()
    }

    ApiService.addDocumentToKnowledgeBase(base.id, localDoc)
    refreshKnowledgeBases()

    try {
      const result = await ApiService.uploadKnowledgeDocument(file, userId)
      const updatedDoc: KnowledgeDocument = {
        ...localDoc,
        status: 'ready',
        indexId: result?.index_result?.index_id || base.indexId,
        path: result?.path
      }

      const bases = ApiService.getKnowledgeBases().map(item => {
        if (item.id !== base.id) return item
        return {
          ...item,
          updatedAt: new Date().toISOString(),
          documents: item.documents.map(doc => (doc.id === localDoc.id ? updatedDoc : doc)),
          indexId: updatedDoc.indexId || item.indexId
        }
      })
      ApiService.saveKnowledgeBases(bases)
      refreshKnowledgeBases()
    } catch (error) {
      const bases = ApiService.getKnowledgeBases().map(item => {
        if (item.id !== base.id) return item
        return {
          ...item,
          documents: item.documents.map(doc =>
            doc.id === localDoc.id ? { ...doc, status: 'error' as const } : doc
          )
        }
      })
      ApiService.saveKnowledgeBases(bases)
      refreshKnowledgeBases()
      console.error('上传失败', error)
    }
  }

  ;(event.target as HTMLInputElement).value = ''
}

const goToChat = () => {
  if (props.preview) return
  ApiService.setActiveKnowledgeBase(activeKnowledgeBaseId.value)
  router.push('/chat')
}
</script>

<style scoped>
.preview-note { color: #9ab9d6; line-height: 1.8; font-size: 13px; }
button:disabled { cursor: default; opacity: .5; }
:global(body) {
  margin: 0;
  font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
  background:
    radial-gradient(circle at top left, rgba(87, 147, 255, 0.14), transparent 28%),
    radial-gradient(circle at bottom right, rgba(127, 125, 255, 0.14), transparent 32%),
    #07111f;
  color: #eaf2ff;
}

.knowledge-page {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 320px 1fr;
}

.knowledge-sidebar {
  padding: 28px 22px;
  border-right: 1px solid rgba(146, 179, 255, 0.12);
  background: rgba(6, 13, 25, 0.88);
  backdrop-filter: blur(16px);
}

.eyebrow {
  margin: 0 0 8px;
  font-size: 12px;
  color: #7cb8ff;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.sidebar-top h1,
.knowledge-header h2,
.document-panel h3 {
  margin: 0;
}

.create-btn,
.upload-btn,
.primary-btn,
.base-item {
  border: none;
  cursor: pointer;
}

.create-btn,
.upload-btn,
.primary-btn {
  padding: 12px 16px;
  border-radius: 16px;
  background: linear-gradient(135deg, #78dce8, #7f7dff);
  color: #04101d;
  font-weight: 700;
}

.knowledge-base-list {
  margin-top: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.base-item,
.overview-card,
.document-panel,
.document-card,
.knowledge-header {
  border: 1px solid rgba(133, 168, 255, 0.14);
  background: rgba(12, 22, 38, 0.72);
  box-shadow: 0 18px 56px rgba(0, 0, 0, 0.2);
}

.base-item {
  text-align: left;
  padding: 16px;
  border-radius: 20px;
  color: #dceaff;
}

.base-item p,
.desc,
.panel-tip,
.document-card p,
.empty-state p {
  color: #91addb;
}

.base-item.active,
.base-item:hover {
  background: linear-gradient(135deg, rgba(87, 147, 255, 0.2), rgba(38, 223, 255, 0.12));
}

.back-link {
  display: inline-block;
  margin-top: 24px;
  color: #d8e7ff;
  text-decoration: none;
}

.knowledge-main {
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.knowledge-header {
  padding: 24px;
  border-radius: 28px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.overview-card {
  padding: 20px;
  border-radius: 24px;
}

.overview-card strong {
  display: block;
  margin: 12px 0 6px;
  font-size: 20px;
}

.document-panel {
  border-radius: 28px;
  padding: 22px;
}

.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 18px;
}

.document-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 14px;
}

.document-card {
  padding: 18px;
  border-radius: 20px;
}

.doc-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
}

.doc-type,
.doc-status {
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
}

.doc-status.ready {
  color: #8ef0b8;
}

.doc-status.uploading,
.doc-status.processing {
  color: #ffd479;
}

.doc-status.error {
  color: #ff9191;
}

.empty-state {
  padding: 42px 24px;
  text-align: center;
}

@media (max-width: 1200px) {
  .knowledge-page {
    grid-template-columns: 1fr;
  }

  .overview-grid {
    grid-template-columns: 1fr;
  }

  .knowledge-header,
  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>

