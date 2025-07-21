// src/lib/typedFetch.tsx
import { z } from 'zod';
import { paths } from '../types/openapi';
import { fetchJson } from './fetchJson';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

type ChatPostRequest =
  paths['/chat']['post']['requestBody']['content']['application/json'];
type ChatPostResponse =
  paths['/chat']['post']['responses']['200']['content']['application/json'];

const ChatResponseSchema = z.object({
  answer: z.string(),
  latency: z.number(),
  conversation_id: z.string().nullable(),
  citations: z.array(
    z.object({
      id: z.string(),
      source: z.string(),
      url: z.string().url().nullable(),
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
  const data = await fetchJson<ChatPostResponse>(
    `${API_BASE}/chat`,
    {
      method: 'POST',
      mode: 'cors',
      body: JSON.stringify(body)
    }
  );
  return ChatResponseSchema.parse(data);
}