// src/hooks/useChat.tsx
import { useRef, useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChatStore } from '../store/useChatStore';
import { streamChat } from '../lib/streamChat';
import type { ConversationMessage } from '../types/Types';
import { useAuth } from '../hooks/useAuth';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export function useChat() {
  const { cid, setCid, messages, addMessage, editMessage, reset, isLoadingFromHistory } = useChatStore();
  const { isAuthenticated, getAccessToken } = useAuth();

  const [isTyping, setIsTyping]   = useState(false);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [error, setError]         = useState<string | null>(null);

  // Only show welcome once, if the *store* is initially empty AND not loading from history
  const didInit = useRef(false);
  useEffect(() => {
    if (didInit.current || isLoadingFromHistory) return;
    
    // Double check we're not loading from history
    const currentState = useChatStore.getState();
    if (currentState.isLoadingFromHistory) return;
    
    didInit.current = true;

    if (messages.length === 0) {
      addMessage({
        id: uuidv4(),
        role: 'assistant',
        content: "Hi! I'm your medical assistant. How can I help you today?",
      });
    }
  }, [messages.length, addMessage, isLoadingFromHistory]);

  // "New Chat" clears the store (and localStorage) via the store's reset()
  const clearChat = useCallback(() => {
    reset();
    addMessage({
      id: uuidv4(),
      role: 'assistant',
      content: "Hi! I'm your medical assistant. How can I help you today?",
    });
  }, [reset, addMessage]);

  const sendMessage = useCallback(
    async (raw: string) => {
      const query = raw.trim();
      if (!query) return;

      // Prevent sending messages while loading from history
      if (isLoadingFromHistory) return;

      // Don't add duplicate user messages
      const tail = useChatStore.getState().messages.slice(-1)[0];
      const isDuplicate = tail?.role === 'user' && tail.content.trim() === query;
      
      if (!isDuplicate) {
        addMessage({ id: uuidv4(), role: 'user', content: query });
      }

      setLastQuery(query);
      setIsTyping(true);
      setError(null);

      const currentCid = cid ?? null;
      const aid = uuidv4();
      addMessage({ id: aid, role: 'assistant', content: '', meta: { streaming: true } });

      try {
        const token = isAuthenticated
          ? await getAccessToken({
              authorizationParams: { audience: import.meta.env.VITE_AUTH0_AUDIENCE },
            })
          : undefined;

        // build the history array of { user, assistant } turns
        const all = useChatStore.getState().messages;
        const history: ConversationMessage[] = [];
        let pendingUser: string | undefined;
        all.forEach((m) => {
          if (m.role === 'user') {
            pendingUser = m.content;
          } else if (m.role === 'assistant' && pendingUser !== undefined) {
            history.push({ user: pendingUser, assistant: m.content });
            pendingUser = undefined;
          }
        });
      
        // include history in the payload
        const req = {
          query,
          conversation_id: currentCid,
          history,
        };
        for await (const chunk of streamChat(req, API_BASE, token)) {
          if (chunk.type === 'token') {
            editMessage(aid, (m) => ({ ...m, content: m.content + chunk.token }));
          } else {
            editMessage(aid, (m) => ({
              ...m,
              content: chunk.meta.answer,
              meta: {
                latency: chunk.meta.latency,
                citations: chunk.meta.citations,
                streaming: false,
              },
            }));
            setCid(chunk.meta.conversation_id);
          }
        }
      } catch (err) {
        console.error(err);
        editMessage(aid, (m) => ({
          ...m,
          content: '⚠️ Something went wrong.',
          meta: { ...m.meta, streaming: false, error: true },
        }));
        setError('request_failed');
      } finally {
        setIsTyping(false);
      }
    },
    [cid, isAuthenticated, getAccessToken, addMessage, editMessage, setCid, isLoadingFromHistory]
  );

  const retryLast = useCallback(() => lastQuery && sendMessage(lastQuery), [lastQuery, sendMessage]);

  return { messages, sendMessage, retryLast, clearChat, isTyping, error };
}