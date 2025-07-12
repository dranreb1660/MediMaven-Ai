import { z } from 'zod';
import { paths } from '../types/openapi';

/* -------------------------Base URL----------------------------------------- */
const API_BASE =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000'; // dev fallback
/* ------------------------------------------------------------------ */


type ChatPostRequest =
  paths['/chat']['post']['requestBody']['content']['application/json'];
type ChatPostResponse =
  paths['/chat']['post']['responses']['200']['content']['application/json'];

const ChatResponseSchema = z.object({
  answer: z.string(),
  latency: z.number(),
  conversation_id: z.string().nullable(),   // ← updated
  citations: z.array(
    z.object({
      id: z.string(),
      source: z.string(),
      url: z.string().url().nullable(),     // ← updated
      rank: z.number(),
    }),
  ),
  messages: z.array(
    z.object({
      user: z.string(),
      assistant: z.string(),
    }),
  ),
});


export async function postChat(
  body: ChatPostRequest,
): Promise<ChatPostResponse> {
  const res = await fetch(`${API_BASE}/chat`, {      // 👈 prepend base
    method: 'POST',
    mode: 'cors',                                    // good practice
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) throw new Error(`Chat API ${res.status}`);
  const data = await res.json();
  return ChatResponseSchema.parse(data);
}
