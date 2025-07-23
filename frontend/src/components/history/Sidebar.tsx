// src/components/history/Sidebar.tsx
import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { useChatStore } from '../../store/useChatStore';
import { fetchJson } from '../../lib/fetchJson';

const API_BASE    = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const AUDIENCE    = import.meta.env.VITE_AUTH0_AUDIENCE;
const STORAGE_KEY = 'medimaven.chatHistory';

interface ConvMeta {
  cid: string;
  preview: string;
  messages: Array<{
    user?: string;
    assistant?: string;
    citations?: { id: string; source?: string; url?: string; rank: number }[];
    latency?: number;
  }>;
}

export default function Sidebar({ onClose }: { onClose: () => void }) {
  const { isAuthenticated, getAccessToken } = useAuth();
  const [items, setItems]   = useState<ConvMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const didFetch = useRef(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

  // 1️⃣ Hydrate from localStorage for instant UI
  useEffect(() => {
    if (!isAuthenticated) return;
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      try { setItems(JSON.parse(raw)); }
      catch { /* ignore */ }
    }
  }, [isAuthenticated]);

  // 2️⃣ Fetch fresh from backend once per open
  useEffect(() => {
    if (!isAuthenticated || didFetch.current) return;
    didFetch.current = true;
    setLoading(true);

    (async () => {
      const token = await getAccessToken({ authorizationParams: { audience: AUDIENCE } });
      const fresh: ConvMeta[] = await fetchJson(
        `${API_BASE}/chat/list`,
        { method: 'GET' },
        token
      );
      setItems(fresh);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(fresh));
    })()
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [isAuthenticated, getAccessToken]);

  // 3️⃣ Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (sidebarRef.current && !sidebarRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  // 4️⃣ Load a conversation (with its meta) into your chat store
const loadChat = (it: ConvMeta) => {
  const { reset, setCid, addMessage, setLoadingFromHistory } = useChatStore.getState();

  // Set the flag BEFORE doing anything
  setLoadingFromHistory(true);

  // Use setTimeout to ensure the flag is set before any effects run
  setTimeout(() => {
    reset();
    setCid(it.cid);

    it.messages.forEach((turn, idx) => {
      // 1) replay the user utterance
      addMessage({
        id:   `${it.cid}-u-${idx}`,
        role: 'user',
        content: turn.user!,
      });

      // 2) replay the assistant reply
      addMessage({
        id:      `${it.cid}-a-${idx}`,
        role:    'assistant',
        content: turn.assistant!,
        meta: {
          streaming: false,
          latency:   turn.latency,
          citations: turn.citations?.map(c => ({
            id:     c.id,
            source: c.source!,
            url:    c.url  ?? null,
            rank:   c.rank,
          })),
        },
      });
    });

    // Clear the flag after loading is complete
    setTimeout(() => {
      setLoadingFromHistory(false);
    }, 100);
  }, 0);

  onClose();
  window.scrollTo({ top: 0 });
};

  return (
    <div className="fixed inset-0 z-50 flex justify-end pointer-events-auto">
      {/* backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
        onClick={onClose}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onClose();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Close sidebar"
      />

      {/* panel */}
      <aside
        ref={sidebarRef}
        className="relative z-50 w-72 h-full bg-white dark:bg-gray-900
                   border-l-4 border-mm-accent-500 shadow-2xl animate-slide-in"
      >
        <div className="flex items-center justify-between p-4 border-b dark:border-gray-700">
          <h3 className="font-semibold text-gray-700 dark:text-gray-200">
            Chat History
          </h3>
          <button
            onClick={onClose}
            className="text-gray-600 dark:text-gray-400 hover:text-red-500"
          >
            ✕
          </button>
        </div>

        <div className="overflow-y-auto p-2 space-y-1 bg-gray-50 dark:bg-gray-800">
          {items.map((it) => (
            <button
              key={it.cid}
              onClick={() => loadChat(it)}
              className="w-full p-2 rounded cursor-pointer truncate text-left
                         hover:bg-mm-blue-100 dark:hover:bg-mm-blue-900
                         text-sm text-gray-900 dark:text-gray-200
                         focus:outline-none focus:ring-2 focus:ring-mm-accent-500"
              aria-label={`Load chat: ${it.preview}`}
            >
              {it.preview}
            </button>
          ))}

          {loading && (
            <p className="p-2 text-xs text-center text-gray-500 animate-pulse">
              Retrieving chats…
            </p>
          )}

          {!loading && items.length === 0 && (
            <p className="p-4 text-gray-400 text-sm">
              You have no chat history.
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}