
/** Extra info carried only by assistant bubbles */

import { Citation } from '../lib/api';
export interface AssistantMeta {
  latency?: number;
  citations?: Citation[];
  modelVersion?: string;
  error?: boolean;
  streaming?: boolean;

}
  
/** Chat message (user or assistant) */
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  meta?: AssistantMeta;
}
  