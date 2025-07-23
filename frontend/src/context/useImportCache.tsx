import { useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { rehydrateChat } from '../lib/cache';
import { useChatStore } from '../store/useChatStore';

export function useImportCache() {
  const { isAuthenticated, isLoading, getAccessToken } = useAuth();

  useEffect(() => {
    if (!isAuthenticated || isLoading) return;

    (async () => {
      const cache = await rehydrateChat();
      if (!cache || !cache.messages.length) return;

      const token = await getAccessToken();
      await fetch('/chat/import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(cache),
      }).then(r => r.ok && useChatStore.getState().reset());
    })();
  }, [isAuthenticated, isLoading, getAccessToken]);
}

