import { useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { api, ChatRequest, ChatResponse } from '../lib/api';
import { Message } from '../types/chat';


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
  // 1) initial history
  const [messages, setMessages] = useState<Message[]>(
    loadHistory().length
      ? loadHistory()
      : [
          {
            id: uuidv4(),
            role: 'assistant',
            content: 'Hi! I’m your medical assistant. How can I help you today?'
          }
        ]
  );

  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);

  // 2) persist history every change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  // 3) helper: reset chat
  const clearChat = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setMessages([
      {
        id: uuidv4(),
        role: 'assistant',
        content: 'Hi! I’m your medical assistant. How can I help you today?'
      }
    ]);
  }, []);

  // 4) main send
  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;

      setLastQuery(trimmed);
      setMessages(prev => [...prev, { id: uuidv4(), role: 'user', content: trimmed }]);
      setIsTyping(true);
      setError(null);

      try {
        const res = await api.post<ChatResponse>('/chat', { query: trimmed } as ChatRequest);
        const { answer, latency, model_version } = res.data;

        const id = uuidv4();
        setMessages(prev => [
          ...prev,
          { id, role: 'assistant', content: '', meta: { latency, modelVersion: model_version } }
        ]);

        // fake word‑by‑word streaming
        for (const word of answer.split(' ')) {
          setMessages(prev =>
            prev.map(m =>
              m.id === id ? { ...m, content: `${m.content}${m.content ? ' ' : ''}${word}` } : m
            )
          );
          await new Promise(r => setTimeout(r, 40));
        }
      } catch (err: any) {
        console.error('[Chat API]', err?.response ?? err);
        setMessages(prev => [
          ...prev,
          {
            id: uuidv4(),
            role: 'assistant',
            content: '⚠️ Something went wrong.',
            meta: { error: true }
          }
        ]);
      } finally {
        setIsTyping(false);
      }
    },
    []
  );

  // 5) retry helper
  const retryLast = useCallback(() => {
    if (lastQuery) sendMessage(lastQuery);
  }, [lastQuery, sendMessage]);

  return { messages, sendMessage, retryLast, clearChat, isTyping, error };
}
