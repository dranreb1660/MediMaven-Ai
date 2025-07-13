// frontend/src/store/useChatStore.ts
import { create } from 'zustand';
import { type Message } from '../types/chat';

const LS_KEY = 'medimaven.chat';

interface ChatState {
  cid?: string;
  messages: Message[];
  setCid: (id?: string) => void;
  addMessage: (msg: Message) => void;
  /** wipe cid + messages + localStorage */
  reset: () => void;
  /** mutate existing message (for streaming updates) */
  editMessage: (id: string, updater: (m: Message) => Message) => void;
}

export const useChatStore = create<ChatState>()((set) => ({
  cid: undefined,
  messages: [],

  setCid: (id) => set({ cid: id }),

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  reset: () => {
    localStorage.removeItem(LS_KEY);          // ← clears persisted chat
    set({ cid: undefined, messages: [] });
  },

  editMessage: (id, updater) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? updater(m) : m,
      ),
    })),
}));
