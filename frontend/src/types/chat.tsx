export interface AssistantMeta {
    latency?: number;
    modelVersion?: string;
    error?: boolean;
  }
  
  export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    meta?: AssistantMeta;
  }
  