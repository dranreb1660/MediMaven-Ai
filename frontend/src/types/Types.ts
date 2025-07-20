// src/types/chat.ts

/** A single turn in the conversation, for replay or history. */
export interface ConversationMessage {
  user: string;
  assistant: string;
}

export interface ChatRequest {
  query: string;
  conversation_id?: string | null;
  history?: ConversationMessage[];   // <-- now known
}

export interface Citation {
  id: string;
  source: string;
  url?: string | null;
  rank: number;
}

export interface HistoryTurn {
  user: string;
  assistant: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  latency: number;
  conversation_id: string;
  messages: HistoryTurn[];
  model_version?: string;
}

export interface AssistantMeta {
  latency?: number;
  citations?: Citation[];
  modelVersion?: string;
  error?: boolean;
  streaming?: boolean;
}

/** Chat message in the UI store */
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  meta?: AssistantMeta;
}
