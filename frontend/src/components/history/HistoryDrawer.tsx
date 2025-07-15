import { useState, useEffect } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { useAuth } from '../../context/AuthContext';
import { useChatStore } from '../../store/useChatStore';
import { XMarkIcon } from '@heroicons/react/24/solid';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

interface HistItem {
  cid: string;
  preview: string;
  messages: any[];
}

export default function HistoryDrawer() {
  const { isAuthenticated, getAccessToken, login } = useAuth();
  const [items, setItems] = useState<HistItem[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [open, setOpen] = useState(false);

  /* fetch list only when drawer opens + user logged-in */
  useEffect(() => {
    if (!open || !isAuthenticated) return;

    setStatus('loading');
    (async () => {
      try {
        const tk = await getAccessToken();
        const res = await fetch(`${API_BASE}/chat/list?page=1`, {
          headers: { Authorization: `Bearer ${tk}` },
        });

        /* 🔹 dev-stub: treat 404 as “no history yet” */
        if (res.status === 404) {
          setItems([]);
          setStatus('idle');
          return;
        }

        const ctype = res.headers.get('content-type') ?? '';
        if (!res.ok || !ctype.includes('application/json'))
          throw new Error(`Bad response ${res.status}`);

        setItems(await res.json());
        setStatus('idle');
      } catch (err) {
        console.error('[history]', err);
        setStatus('error');
      }
    })();
  }, [open, isAuthenticated, getAccessToken]);

  const loadChat = (it: HistItem) => {
    useChatStore.setState({ cid: it.cid, messages: it.messages });
    setOpen(false);
    window.scrollTo({ top: 0 });
  };

  /* always visible — greyed when logged-out */
  const triggerStyle = isAuthenticated
    ? 'text-xs text-mm-accent hover:underline'
    : 'text-xs text-gray-400 hover:text-gray-500 cursor-pointer';

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger
        className={triggerStyle}
        onClick={() => !isAuthenticated && login()}
      >
        My&nbsp;History
      </Dialog.Trigger>

      {/* Drawer only renders when open */}
      {open && (
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/30" />
          <Dialog.Content
            aria-label="History drawer"                
            className="fixed right-0 top-0 h-full w-80 bg-white dark:bg-gray-900
                       p-4 shadow-lg flex flex-col"
          >
            <Dialog.Title className="mb-3 font-semibold">
              Past Conversations
            </Dialog.Title>

            {status === 'loading' && (
              <p className="text-xs text-gray-500">Loading…</p>
            )}
            {status === 'error' && (
              <p className="text-xs text-red-500">Server error.</p>
            )}
            {!items.length && status === 'idle' && (
              <p className="text-xs text-gray-500">No history yet.</p>
            )}

            <ul className="flex-1 overflow-auto space-y-2">
              {items.map(it => (
                <li
                  key={it.cid}
                  onClick={() => loadChat(it)}
                  className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer"
                >
                  <p className="text-[10px] text-gray-500">{it.cid}</p>
                  <p className="text-sm truncate">{it.preview}</p>
                </li>
              ))}
            </ul>

            <Dialog.Close asChild>
              <button
                className="self-end mt-4 text-xs flex items-center gap-1 hover:underline"
              >
                <XMarkIcon className="h-4 w-4" /> Close
              </button>
            </Dialog.Close>
          </Dialog.Content>
        </Dialog.Portal>
      )}
    </Dialog.Root>
  );
}
