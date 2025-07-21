// src/store/useChatStore.ts
import { create } from 'zustand';
import type { Message } from '../types/Types';

const LS_KEY = 'medimaven.chat';

interface ChatState {
  cid?: string;
  messages: Message[];
  isLoadingFromHistory: boolean; // Add this flag
  setCid: (id?: string) => void;
  addMessage: (msg: Message) => void;
  reset: () => void;
  editMessage: (id: string, updater: (m: Message) => Message) => void;
  setLoadingFromHistory: (loading: boolean) => void; // Add this method
}

// ─── Load whatever's in localStorage, old or new format ───
function loadInitial(): { cid?: string; messages: Message[] } {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { cid: undefined, messages: [] };
    const data = JSON.parse(raw);

    // You previously stored just the messages array → migrate that:
    if (Array.isArray(data)) {
      return { cid: undefined, messages: data as Message[] };
    }
    // If you've already moved to { cid, messages }:
    if (data && Array.isArray(data.messages)) {
      return {
        cid: typeof data.cid === 'string' ? data.cid : undefined,
        messages: data.messages as Message[],
      };
    }
  } catch {
    // parse error → fall through
  }
  return { cid: undefined, messages: [] };
}

// ─── Persist helper ───────────────────────────────────────
function persist(get: () => ChatState) {
  const { cid, messages } = get();
  localStorage.setItem(LS_KEY, JSON.stringify({ cid, messages }));
}

const initial = loadInitial();

export const useChatStore = create<ChatState>((set, get) => ({
  // set from migrated initial state
  cid: initial.cid,
  messages: initial.messages,
  isLoadingFromHistory: false,

  setCid: (id) => {
    set({ cid: id });
    persist(get);
  },

  addMessage: (msg) => {
    set((s) => ({ messages: [...s.messages, msg] }));
    persist(get);
  },

  reset: () => {
    set({ cid: undefined, messages: [] });
    // Don't clear the isLoadingFromHistory flag here - let the caller manage it
    persist(get);
  },

  editMessage: (id, updater) =>
    set((s) => {
      const updated = s.messages.map((m) => (m.id === id ? updater(m) : m));
      persist(get);
      return { messages: updated };
    }),

  setLoadingFromHistory: (loading) => {
    set({ isLoadingFromHistory: loading });
  },
}));