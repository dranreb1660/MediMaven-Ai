import { useRef, useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChatStore } from '../store/useChatStore';
import { streamChat } from '../lib/streamChat';
import { type Message } from '../types/chat';

const STORAGE_KEY = 'medimaven.chat';
// const API_BASE    = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const API_BASE = "https://217d86f59c2e.ngrok-free.app"; // for testing

/* ---------- utils --------- */
function loadHistory(): Message[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Message[]) : [];
  } catch {
    return [];
  }
}
function saveHistory(msgs: Message[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(msgs));
}

/* ---------- hook ---------- */
export function useChat() {
  const { cid, setCid, messages, addMessage, editMessage, reset } =
    useChatStore();

  const [isTyping, setIsTyping] = useState(false);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* hydrate once */
  const didHydrate = useRef(false);
  useEffect(() => {
    if (didHydrate.current) return;
    didHydrate.current = true;

    const hist = loadHistory();
    if (hist.length) hist.forEach(addMessage);
    else
      addMessage({
        id: uuidv4(),
        role: 'assistant',
        content: 'Hi! I’m your medical assistant. How can I help you today?',
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* persist */
  useEffect(() => saveHistory(messages), [messages]);

  /* clear chat */
  const clearChat = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    reset();
    addMessage({
      id: uuidv4(),
      role: 'assistant',
      content: 'Hi! I’m your medical assistant. How can I help you today?',
    });
  }, [reset, addMessage]);

  /* main send */
  const sendMessage = useCallback(
    async (raw: string) => {
      const query = raw.trim();
      if (!query) return;

      setLastQuery(query);
      addMessage({ id: uuidv4(), role: 'user', content: query });
      setIsTyping(true);
      setError(null);

      try {
        const req = { query, conversation_id: cid ?? null };

        const aid = uuidv4();
        addMessage({ id: aid, role: 'assistant', content: '', meta: { streaming: true } });

        for await (const chunk of streamChat(req, API_BASE)) {
          if (chunk.type === 'token') {
            editMessage(aid, m => ({ ...m, content: m.content + chunk.token }));
          } else {
            /* done */
            editMessage(aid, m => ({
              ...m,
              content: chunk.meta.answer,
              meta: {
                latency: chunk.meta.latency,
                citations: chunk.meta.citations,
                streaming: false,
              },
            }));
            setCid(chunk.meta.conversation_id ?? undefined);
          }
        }
      } catch (err) {
        console.error('[chat]', err);
        addMessage({
          id: uuidv4(),
          role: 'assistant',
          content: '⚠️ Something went wrong.',
          meta: { error: true },
        });
        setError('request_failed');
      } finally {
        setIsTyping(false);
      }
    },
    [cid, setCid, addMessage, editMessage],
  );

  const retryLast = useCallback(
    () => lastQuery && sendMessage(lastQuery),
    [lastQuery, sendMessage],
  );

  return { messages, sendMessage, retryLast, clearChat, isTyping, error };
}
