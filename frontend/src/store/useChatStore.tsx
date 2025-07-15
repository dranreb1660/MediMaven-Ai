import { create } from 'zustand';
import { type Message } from '../types/chat';
import { persistChat, rehydrateChat } from '../lib/cache';

/* ---------- types ---------- */
interface ChatState {
  cid?: string;
  messages: Message[];
  hydrated: boolean;
  setCid: (id?: string) => void;
  addMessage: (msg: Message) => void;
  reset: () => void;
  editMessage: (id: string, updater: (m: Message) => Message) => void;
}

interface CachePayload {
  cid?: string;
  messages: Message[];
}

/* ---------- store ---------- */
export const useChatStore = create<ChatState>((set, get) => ({
  cid: undefined,
  messages: [],
  hydrated: false,

  addMessage: (msg) => {
    set((s) => ({ messages: [...s.messages, msg] }));
    const { hydrated, cid, messages } = get();
    if (hydrated) persistChat(cid, messages).catch(console.error);
  },

  setCid: (id) => {
    set({ cid: id });
    const { hydrated, messages } = get();
    if (hydrated) persistChat(id, messages).catch(console.error);
  },

  reset: () => {
    set({ cid: undefined, messages: [] });
    const { hydrated } = get();
    if (hydrated) persistChat(undefined, []).catch(console.error);
  },

  editMessage: (id, updater) =>
    set((s) => {
      const updated = s.messages.map((m) =>
        m.id === id ? updater(m) : m,
      );
      if (s.hydrated) persistChat(s.cid, updated).catch(console.error);
      return { messages: updated };
    }),
}));

/* ---------- one-time hydrate ---------- */
rehydrateChat()
  .then((cache: CachePayload | null) => {
    if (cache) {
      useChatStore.setState({
        cid: cache.cid,
        messages: cache.messages,
        hydrated: true,
      });
    } else {
      useChatStore.setState({ hydrated: true });
    }
  })
  .catch((e) => {
    console.error('IDB hydrate error', e);
    useChatStore.setState({ hydrated: true });
  });
