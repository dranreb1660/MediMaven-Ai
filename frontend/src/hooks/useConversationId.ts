// hooks/useConversationId.ts
import { useState } from 'react';
// This hook manages the conversation ID, storing it in localStorage
// and providing a way to update it.

export function useConversationId() {
  const [cid, setCid] = useState<string | undefined>(() =>
    localStorage.getItem('medimaven:cid') || undefined
  );

  const updateCid = (val: string | undefined) => {
    setCid(val);
    if (val) localStorage.setItem('medimaven:cid', val);
    else localStorage.removeItem('medimaven:cid');
  };

  return { cid, updateCid };
}
