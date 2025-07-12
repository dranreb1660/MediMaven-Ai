import { create } from 'zustand';
import { type Message } from '../types/chat';  

interface ChatState {
  cid?: string;
  messages: Message[];
  setCid: (id?: string) => void;
  addMessage: (msg: Message) => void;
  reset: () => void;
  editMessage: (id: string, updater: (m: Message) => Message) => void;

}

export const useChatStore = create<ChatState>()((set) => ({
  cid: undefined,
  messages: [],
  setCid: (id) => set({ cid: id }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  reset: () => set({ cid: undefined, messages: [] }),
  editMessage: (id: string, updater: (m: Message) => Message) =>
  set((s) => ({
    messages: s.messages.map((m) => (m.id === id ? updater(m) : m)),
    })),
}));
