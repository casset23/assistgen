export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  isSearching?: boolean
  sources?: KnowledgeSource[]
}

export interface ChatHistory {
  id: number
  title: string
  time: Date
  messages: ChatMessage[]
}

export interface SearchResult {
  title: string
  url: string
  snippet: string
  date?: string
  source?: string
  isExpanded?: boolean
}

export interface KnowledgeSource {
  chunk_id?: string
  title?: string
  section?: string
  content?: string
  page?: number | string
  score?: number
  rerank_score?: number
  snippet?: string
  original_snippet?: string
}
