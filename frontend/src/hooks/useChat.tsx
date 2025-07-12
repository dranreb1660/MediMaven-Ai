import { useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { api, ChatRequest, ChatResponse } from '../lib/api';
import { useChatStore } from '../store/useChatStore';   // 👈 NEW
import { type Message } from '../types/chat';

const STORAGE_KEY = 'medimaven.chat';

function loadHistory(): Message[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Message[]) : [];
  } catch {
    return [];
  }
}

export function useChat() {
  // ──────────────────────────────── 1. Zustand state
  const { cid, setCid, messages, addMessage, reset } = useChatStore();

  // ──────────────────────────────── 2. local UI flags
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);

  // ──────────────────────────────── 3. hydrate store from LS on first mount
  useEffect(() => {
    if (messages.length === 0) {
      const hist = loadHistory();
      hist.forEach(addMessage);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ──────────────────────────────── 4. persist to LS whenever messages change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  // ──────────────────────────────── 5. reset helper
  const clearChat = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    reset();
    addMessage({
      id: uuidv4(),
      role: 'assistant',
      content: 'Hi! I’m your medical assistant. How can I help you today?',
    });
  }, [reset, addMessage]);

  // ──────────────────────────────── 6. main send
  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;

      setLastQuery(trimmed);
      addMessage({ id: uuidv4(), role: 'user', content: trimmed });
      setIsTyping(true);
      setError(null);

      try {
        const req: ChatRequest = { query: trimmed, conversation_id: cid ?? null };
        const res = await api.post<ChatResponse>('/chat', req);

        // cache cid for future turns
        setCid(res.data.conversation_id);

        const { answer, latency, citations } = res.data;
        const id = uuidv4();
        addMessage({
          id,
          role: 'assistant',
          content: '',
          meta: { latency, citations },
        });

        // fake streaming
        for (const word of answer.split(' ')) {
        useChatStore.getState().editMessage(id, (m) => ({
          ...m,
          content: m.content ? `${m.content} ${word}` : word,
        }));
        await new Promise((r) => setTimeout(r, 40)); 
   }
      } catch (err: any) {
        console.error('[Chat API]', err?.response ?? err);
        addMessage({
          id: uuidv4(),
          role: 'assistant',
          content: '⚠️ Something went wrong.',
          meta: { error: true },
        });
      } finally {
        setIsTyping(false);
      }
    },
    [addMessage, cid, setCid],
  );

  // ──────────────────────────────── 7. retry helper
  const retryLast = useCallback(() => {
    if (lastQuery) sendMessage(lastQuery);
  }, [lastQuery, sendMessage]);

  return { messages, sendMessage, retryLast, clearChat, isTyping, error };
}
