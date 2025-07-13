// frontend/components/ui/SiteHeader.tsx
import { Link } from 'react-router-dom';
import {
  TrashIcon,
  HomeIcon,
  ChatBubbleLeftRightIcon,
  ArrowPathIcon,       // ↻ new-chat icon
} from '@heroicons/react/24/outline';
import DarkToggle from './DarkToggle';
import { useChatStore } from '../../store/useChatStore';  // Zustand state

type HeaderVariant = 'welcome' | 'chat';

interface HeaderProps {
  variant: HeaderVariant;
  onClear?: () => void;   // still used for hard-wipe (optional)
}

export default function SiteHeader({ variant, onClear }: HeaderProps) {
  /* ─ Zustand ─ */
  const { cid, reset } = useChatStore();

  /* ─ route toggle (left) ─ */
  const LeftIcon =
    variant === 'chat' ? HomeIcon : ChatBubbleLeftRightIcon;
  const leftHref = variant === 'chat' ? '/' : '/chat';

  /* ─ new-chat handler ─ */
  const handleNewChat = async () => {
    if (cid) {
      // cleanly close the thread on backend (non-blocking)
      fetch(
        `${import.meta.env.VITE_API_URL ?? 'http://localhost:8000'}/chat/end`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ conversation_id: cid }),
        },
      ).catch(console.error);
    }
    reset();                       // wipe cid + messages + LS
    window.scrollTo({ top: 0 });
  };

  return (
    <div className="sticky top-0 z-20 w-full bg-gray-50 dark:bg-gray-900 border-b dark:border-gray-700">
      <div className="mx-auto max-w-screen-lg flex items-center justify-between px-4 py-2">
        {/* ───────── left block ───────── */}
        <div className="flex items-center gap-2">
          <Link to={leftHref} title={variant === 'chat' ? 'Home' : 'Chat'}>
            <LeftIcon className="h-5 w-5 text-gray-400 hover:text-mm-accent" />
          </Link>
          <div>
            <h1 className="font-bold text-sm sm:text-base text-gray-900 dark:text-mm-accent">
              MediMaven&nbsp;AI
            </h1>
            <p className="text-[10px] leading-none text-gray-500 dark:text-gray-400">
              …your virtual&nbsp;AI&nbsp;doctor
            </p>
          </div>
        </div>

        {/* ───────── right block ───────── */}
        <div className="flex items-center gap-3">
          {variant === 'chat' && (
            <>
              {/* New-chat ↻ */}
              <button
                onClick={handleNewChat}
                title="New chat"
                className="p-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                <ArrowPathIcon className="h-5 w-5 text-gray-400 hover:text-mm-accent" />
              </button>

              {/* Optional “clear chat” (hard delete) */}
              {onClear && (
                <button onClick={onClear} title="Clear chat">
                  <TrashIcon className="h-5 w-5 text-gray-400 dark:text-gray-500 hover:text-red-500" />
                </button>
              )}
            </>
          )}

          {/* Dark-mode toggle */}
          <DarkToggle />
        </div>
      </div>
    </div>
  );
}
