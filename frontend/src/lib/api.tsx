import axios from 'axios';

/**
 * Shared Axios instance.
 *
 * 👉 Timeout is set to **180 000 ms (3 min)** because the model can take up to
 *    two minutes to generate an answer.  Adjust if you optimise latency later.
 */


// ---------------- API types ------------------
export interface ChatRequest {
  query: string;
  conversation_id?: string | null;
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
