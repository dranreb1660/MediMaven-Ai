
import { createParser } from 'eventsource-parser';

export interface TokenChunk { type: 'token'; token: string }
export interface DoneChunk  { type: 'done';  meta: any }

/**
 * Yield `{type:'token', token}` for every streamed chunk,
 * followed by one `{type:'done', meta}` object at the end.
 */
export async function* streamChat(
  body: unknown,
//   apiBase = import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
apiBase = "https://217d86f59c2e.ngrok-free.app", // for testing

) {
  const res = await fetch(`${apiBase}/chat/stream`, {
    method : 'POST',
    headers: {
      'Content-Type' : 'application/json',
      'Accept'       : 'text/event-stream',
    },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body)
    throw new Error(`SSE ${res.status}`);

  const reader   = res.body.getReader();
  const decoder  = new TextDecoder();
  const queue: (TokenChunk | DoneChunk)[] = [];

  /* ---------- parser (v1 API) ---------- */
  const parser = (createParser as any)({
    onEvent(evt: any) {
      if (!evt.data) return;                       // <— accept all event types
      const data = JSON.parse(evt.data);
      if (data.token) queue.push({ type: 'token', token: data.token });
      if (data.done)  queue.push({ type: 'done',  meta: data });
    },
  }); // cast ≡ silence TS/JS version-mismatch

  /* ---------- pump reader ---------- */
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.feed(decoder.decode(value));
    while (queue.length) yield queue.shift()!;
  }

  /* flush anything parsed after the final chunk */
  while (queue.length) yield queue.shift()!;
}
