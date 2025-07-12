// frontend/src/hooks/useChat.tsx
import { useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChatStore } from '../store/useChatStore';
import { paths } from '../types/openapi'; 
import { postChat } from '../lib/typedFetch';

import { type Message } from '../types/chat';

const STORAGE_KEY = 'medimaven.chat';

type ChatPostRequest  =
  paths['/chat']['post']['requestBody']['content']['application/json'];

type ChatPostResponse =
  paths['/chat']['post']['responses']['200']['content']['application/json'];


/* -------- utils -------- */
function loadHistory(): Message[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Message[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(messages: Message[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

/* -------- hook -------- */
export function useChat() {
  /* 1 ▶ Zustand global state */
  const {
    cid,
    setCid,
    messages,
    addMessage,
    editMessage,
    reset,
  } = useChatStore();

  /* 2 ▶ local ui flags */
  const [isTyping, setIsTyping] = useState(false);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* 3 ▶ hydrate on first mount */
  useEffect(() => {
    if (messages.length === 0) {
      const hist = loadHistory();
      if (hist.length) hist.forEach(addMessage);
      else {
        // seed with welcome message
        addMessage({
          id: uuidv4(),
          role: 'assistant',
          content: 'Hi! I’m your medical assistant. How can I help you today?',
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* 4 ▶ persist every change */
  useEffect(() => {
    saveHistory(messages);
  }, [messages]);

  /* 5 ▶ clear chat helper */
  const clearChat = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    reset();
    addMessage({
      id: uuidv4(),
      role: 'assistant',
      content: 'Hi! I’m your medical assistant. How can I help you today?',
    });
  }, [reset, addMessage]);

  /* 6 ▶ core send function */
  const sendMessage = useCallback(
    async (text: string) => {
      const query = text.trim();
      if (!query) return;

      // optimistic user bubble
      setLastQuery(query);
      addMessage({ id: uuidv4(), role: 'user', content: query });
      setIsTyping(true);
      setError(null);

      try {
        /* ─ call backend ─ */
        const req: ChatPostRequest = {
          query,
          conversation_id: cid ?? null,
        };
        const res: ChatPostResponse = await postChat(req);

        /* ─ cache cid for next turns ─ */
        setCid(res.conversation_id ?? undefined);

        /* ─ create placeholder assistant bubble ─ */
        const aid = uuidv4();
        addMessage({
          id: aid,
          role: 'assistant',
          content: '',
          meta: { latency: res.latency, citations: res.citations },
        });

        /* ─ fake streaming word-by-word ─ */
        for (const word of res.answer.split(' ')) {
          editMessage(aid, (m) => ({
            ...m,
            content: m.content ? `${m.content} ${word}` : word,
          }));
          await new Promise((r) => setTimeout(r, 40));
        }
      } catch (err: any) {
        console.error('[Chat API]', err);
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

  /* 7 ▶ retry helper */
  const retryLast = useCallback(() => {
    if (lastQuery) sendMessage(lastQuery);
  }, [lastQuery, sendMessage]);

  return {
    messages,
    sendMessage,
    retryLast,
    clearChat,
    isTyping,
    error,
  };
}
