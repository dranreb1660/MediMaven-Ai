// src/lib/fetchJson.ts
const DEFAULT_TIMEOUT = 180_000; // 3 minutes
const DEFAULT_RETRIES = 2;

export async function fetchJson<T = any>(
  url: string,
  options: RequestInit = {},
  token?: string,
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
    ...options.headers as Record<string, string>,
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);

  const attemptFetch = async (attempt = 1): Promise<Response> => {
    try {
      const res = await fetch(url, { 
        ...options, 
        headers,
        signal: controller.signal 
      });
      
      if (!res.ok && res.status >= 500 && attempt < DEFAULT_RETRIES) {
        await new Promise(r => setTimeout(r, 1000 * attempt));
        return attemptFetch(attempt + 1);
      }
      
      return res;
    } catch (err: any) {
      if (err.name === 'AbortError') throw new Error('Request timeout');
      if (attempt < DEFAULT_RETRIES) {
        await new Promise(r => setTimeout(r, 1000 * attempt));
        return attemptFetch(attempt + 1);
      }
      throw err;
    }
  };

  try {
    const res = await attemptFetch();
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } finally {
    clearTimeout(timeoutId);
  }
}