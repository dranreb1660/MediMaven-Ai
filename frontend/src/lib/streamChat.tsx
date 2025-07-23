// src/lib/streamChat.tsx
import { createParser } from 'eventsource-parser';

export interface TokenChunk { type: 'token'; token: string }
export interface DoneChunk  { type: 'done';  meta: unknown }

const DEFAULT_TIMEOUT = 180_000; // 3 minutes
const MAX_RETRIES = 2;

export async function* streamChat(
  body: unknown,
  apiBase = import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  token?: string
) {
  let lastError: Error | null = null;
  
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
    
    try {
      const res = await fetch(`${apiBase}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      
      if (!res.ok || !res.body) {
        throw new Error(`SSE ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const queue: (TokenChunk | DoneChunk)[] = [];

      const parser = (createParser as (config: { onEvent: (evt: { data?: string }) => void }) => { feed: (data: string) => void })({
        onEvent(evt: { data?: string }) {
          if (!evt.data) return;
          const data = JSON.parse(evt.data);
          if (data.token) queue.push({ type: 'token', token: data.token });
          if (data.done) queue.push({ type: 'done', meta: data });
        },
      });

      clearTimeout(timeoutId);

      // Stream processing
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        parser.feed(decoder.decode(value));
        while (queue.length) yield queue.shift()!;
      }

      while (queue.length) yield queue.shift()!;
      return; // Success - exit retry loop
      
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      lastError = err;
      
      // Don't retry on client errors or abort
      if (err instanceof Error && err.name === 'AbortError') {
        throw new Error('Stream timeout');
      }
      if (err instanceof Error && err.message?.includes('SSE 4')) { // 4xx errors
        throw err;
      }
      
      // Retry on server errors
      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 1000 * attempt));
        continue;
      }
    }
  }
  
  throw lastError || new Error('Stream failed');
}