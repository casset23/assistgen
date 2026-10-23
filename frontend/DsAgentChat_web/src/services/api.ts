import { ChatMessage } from '../types'
import axios from './axios'
import router from '../router'
import { sha256 } from '../utils/crypto'

interface StreamChunk {
  type?: 'think' | 'response'
  content: string
}

export interface UserCreate {
  username: string
  email: string
  password: string
}

export interface UserLogin {
  email: string
  password: string
}

export interface Token {
  access_token: string
  token_type: string
}

export interface Conversation {
  id: number
  created_at: string
  title: string
  status: 'ongoing' | 'completed'
  dialogue_type: string
}

export interface Message {
  id: number
  conversation_id: number
  sender: 'user' | 'assistant'
  content: string
  created_at: string
  message_type: string
}

export interface BookRagSource {
  chunk_id?: string
  title?: string
  section?: string
  content?: string
  page?: number | string
  score?: number
  rerank_score?: number
  snippet?: string
  original_snippet?: string
  metadata?: Record<string, any>
}

export interface KnowledgeDocument {
  id: string
  name: string
  type: string
  size: number
  status: 'ready' | 'uploading' | 'processing' | 'error'
  uploadedAt: string
  indexId?: string
  path?: string
}

export interface KnowledgeBaseItem {
  id: string
  name: string
  description: string
  category: string
  accent: string
  indexId: string
  updatedAt: string
  documents: KnowledgeDocument[]
}

export interface BookRagGenerateResult {
  status: string
  answer?: string
  response?: string
  content?: string
  references?: BookRagSource[]
  sources?: BookRagSource[]
  retrieved_chunks?: BookRagSource[]
  chunks?: BookRagSource[]
  retrieval_results?: BookRagSource[]
  [key: string]: any
}

const defaultKnowledgeBases: KnowledgeBaseItem[] = [
  {
    id: 'textbook-space',
    name: '教材知识库',
    description: '用于教材问答与课程知识检索，可继续上传教材、讲义与专题资料。',
    category: '教材',
    accent: 'cyan',
    indexId: 'index_cfcdb69d4ead19dcc50768cdb37f1c6d',
    updatedAt: new Date().toISOString(),
    documents: [
      {
        id: 'seed-book-rag',
        name: '教材主索引',
        type: 'system',
        size: 0,
        status: 'ready',
        uploadedAt: new Date().toISOString(),
        indexId: 'index_cfcdb69d4ead19dcc50768cdb37f1c6d'
      }
    ]
  }
]

const KNOWLEDGE_BASES_KEY = 'knowledge_bases'
const ACTIVE_KNOWLEDGE_BASE_KEY = 'active_knowledge_base_id'

function readStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function writeStorage<T>(key: string, value: T) {
  localStorage.setItem(key, JSON.stringify(value))
}

export class ApiService {
  private static baseUrl = import.meta.env.VITE_API_BASE_URL || ''

  static async handleChatStream(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    onChunk: (chunk: StreamChunk) => void
  ) {
    const decoder = new TextDecoder()
    let thinkContent = ''
    let responseContent = ''
    let isInThinkingMode = false
    let pendingBuffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        pendingBuffer += decoder.decode(value, { stream: true })
        const lines = pendingBuffer.split('\n')
        pendingBuffer = lines.pop() ?? ''

        for (const rawLine of lines) {
          const line = rawLine.trim()
          if (!line.startsWith('data: ')) continue

          const content = line.slice(6).trim()
          if (!content || content === '[DONE]') continue

          let cleanContent = content
          try {
            const parsed = JSON.parse(content)
            if (typeof parsed === 'string') {
              cleanContent = parsed
            } else if (parsed?.content) {
              cleanContent = parsed.content
            } else if (parsed?.answer) {
              cleanContent = parsed.answer
            } else {
              cleanContent = JSON.stringify(parsed, null, 2)
            }
          } catch {
            cleanContent = content
              .replace(/^"|"$/g, '')
              .replace(/\\"/g, '"')
              .replace(/\\n/g, '\n')
          }

          if (cleanContent.includes('<think>')) {
            isInThinkingMode = true
            const thinkMatch = cleanContent.match(/<think>([\s\S]*?)<\/think>/)

            if (thinkMatch?.[1]) {
              thinkContent = thinkMatch[1].trim()
              onChunk({ type: 'think', content: thinkContent })
              isInThinkingMode = false
            } else {
              const startContent = cleanContent.split('<think>')[1]
              if (startContent) {
                thinkContent = startContent.trim()
                onChunk({ type: 'think', content: thinkContent })
              }
            }
          } else if (cleanContent.includes('</think>')) {
            isInThinkingMode = false
            const endContent = cleanContent.split('</think>')[0]
            if (endContent) {
              thinkContent += endContent.trim()
              onChunk({ type: 'think', content: thinkContent })
            }

            const afterThink = cleanContent.split('</think>')[1]
            if (afterThink?.trim()) {
              responseContent += afterThink.trim()
              onChunk({ type: 'response', content: responseContent })
            }
          } else if (isInThinkingMode) {
            thinkContent += cleanContent
            onChunk({ type: 'think', content: thinkContent })
          } else {
            responseContent += cleanContent
            onChunk({ type: 'response', content: responseContent })
          }
        }
      }
    } catch (error) {
      console.error('Error reading stream:', error)
      throw error
    }
  }

  static async createConversation(): Promise<number> {
    const response = await fetch(`${this.baseUrl}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        user_id: localStorage.getItem('user_id')
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data.conversation_id
  }

  static async chat(messages: ChatMessage[], conversationId: number) {
    if (!conversationId) throw new Error('Missing conversation_id')

    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        messages,
        user_id: localStorage.getItem('user_id'),
        conversation_id: conversationId
      })
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.body?.getReader()
  }

  static async reason(messages: ChatMessage[], conversationId: number) {
    if (!conversationId) throw new Error('Missing conversation_id')

    const response = await fetch(`${this.baseUrl}/api/reason`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        messages,
        user_id: localStorage.getItem('user_id'),
        conversation_id: conversationId
      })
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.body?.getReader()
  }

  static async search(messages: ChatMessage[], conversationId: number) {
    if (!conversationId) throw new Error('Missing conversation_id')

    const response = await fetch(`${this.baseUrl}/api/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        messages,
        user_id: localStorage.getItem('user_id'),
        conversation_id: conversationId
      })
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.body?.getReader()
  }

  static async bookRagGenerate(payload: {
    query: string
    index_id: string
    top_k?: number
    document_type?: string
    noise_level?: string
    answer_style?: string
    retrieval_mode?: string
  }): Promise<BookRagGenerateResult> {
    const response = await fetch(`${this.baseUrl}/api/book-rag/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        top_k: 5,
        document_type: 'textbook',
        noise_level: 'medium',
        answer_style: 'balanced',
        retrieval_mode: 'balanced',
        ...payload
      })
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.json()
  }

  static async uploadKnowledgeDocument(file: File, userId: string | number) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('user_id', String(userId))

    const response = await fetch(`${this.baseUrl}/api/upload`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      body: formData
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.json()
  }

  static getKnowledgeBases(): KnowledgeBaseItem[] {
    const stored = readStorage<KnowledgeBaseItem[]>(KNOWLEDGE_BASES_KEY, [])
    const fallbackIndexId = 'index_cfcdb69d4ead19dcc50768cdb37f1c6d'
    const normalizedStored = stored.map(base => ({
      ...base,
      indexId: base.indexId === 'educate_rag_book_minerU' ? fallbackIndexId : base.indexId,
      documents: base.documents.map(doc => ({
        ...doc,
        indexId: doc.indexId === 'educate_rag_book_minerU' ? fallbackIndexId : doc.indexId
      }))
    }))

    if (!normalizedStored.length) {
      writeStorage(KNOWLEDGE_BASES_KEY, defaultKnowledgeBases)
      return defaultKnowledgeBases
    }

    writeStorage(KNOWLEDGE_BASES_KEY, normalizedStored)
    return normalizedStored
  }

  static saveKnowledgeBases(bases: KnowledgeBaseItem[]) {
    writeStorage(KNOWLEDGE_BASES_KEY, bases)
  }

  static getActiveKnowledgeBaseId(): string {
    const storedId = localStorage.getItem(ACTIVE_KNOWLEDGE_BASE_KEY)
    const bases = this.getKnowledgeBases()
    return storedId || bases[0]?.id || defaultKnowledgeBases[0].id
  }

  static setActiveKnowledgeBase(id: string) {
    localStorage.setItem(ACTIVE_KNOWLEDGE_BASE_KEY, id)
  }

  static createKnowledgeBase(partial: Partial<KnowledgeBaseItem>) {
    const bases = this.getKnowledgeBases()
    const newBase: KnowledgeBaseItem = {
      id: `kb-${Date.now()}`,
      name: partial.name || '未命名知识库',
      description: partial.description || '新建知识空间，适合上传教材、讲义、论文和实验资料。',
      category: partial.category || '综合资料',
      accent: partial.accent || 'violet',
      indexId: partial.indexId || `kb-index-${Date.now()}`,
      updatedAt: new Date().toISOString(),
      documents: []
    }
    const next = [newBase, ...bases]
    this.saveKnowledgeBases(next)
    this.setActiveKnowledgeBase(newBase.id)
    return newBase
  }

  static addDocumentToKnowledgeBase(baseId: string, document: KnowledgeDocument) {
    const bases = this.getKnowledgeBases().map(base =>
      base.id === baseId
        ? {
            ...base,
            updatedAt: new Date().toISOString(),
            documents: [document, ...base.documents]
          }
        : base
    )
    this.saveKnowledgeBases(bases)
    return bases
  }

  static async getUserConversations(userId: string): Promise<Conversation[]> {
    const response = await fetch(`${this.baseUrl}/api/conversations/user/${userId}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.json()
  }

  static async getConversationMessages(conversationId: number): Promise<Message[]> {
    const userId = localStorage.getItem('user_id')
    const response = await fetch(`${this.baseUrl}/api/conversations/${conversationId}/messages?user_id=${userId}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.json()
  }

  static async deleteConversation(conversationId: number): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/conversations/${conversationId}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
  }

  static async updateConversationName(conversationId: number, name: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/conversations/${conversationId}/name`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({ name })
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
  }
}

export const AuthService = {
  async register(data: UserCreate): Promise<Token> {
    const hashedPassword = await sha256(data.password)
    const response = await axios.post('/api/register', {
      username: data.username,
      email: data.email,
      password: hashedPassword
    })
    return response.data
  },

  async login(data: UserLogin): Promise<Token> {
    const hashedPassword = await sha256(data.password)
    const response = await axios.post(
      '/api/token',
      {
        email: data.email,
        password: hashedPassword
      },
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    )
    return response.data
  },

  async logout() {
    localStorage.removeItem('token')
    router.push('/login')
  },

  async validateToken() {
    try {
      await axios.get('/api/validate-token')
      return true
    } catch {
      return false
    }
  },

  async getUserInfo() {
    try {
      const response = await axios.get('/api/users/me')
      return response.data
    } catch {
      const response = await axios.get('/api/users/me/mock')
      return response.data
    }
  }
}
