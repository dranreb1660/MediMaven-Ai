import { get, set } from 'idb-keyval';
import { type Message } from '../types/Types';

const CACHE_KEY = 'mm.chat.v1';

export async function persistChat(cid: string | undefined, messages: Message[]) {
  const trimmed = messages.slice(-50);
  await set(CACHE_KEY, { cid, messages: trimmed });
}

export async function rehydrateChat() {
  return (await get<{ cid?: string; messages: Message[] }>(CACHE_KEY)) ?? null;
}
