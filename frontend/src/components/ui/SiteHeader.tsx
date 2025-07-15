import { Link } from 'react-router-dom';
import {
  TrashIcon,
  HomeIcon,
  ChatBubbleLeftRightIcon,
  ArrowPathIcon,
  ArrowLeftEndOnRectangleIcon as LogInIcon,
  ArrowRightOnRectangleIcon as LogOutIcon,
} from '@heroicons/react/24/outline';

import DarkToggle from './DarkToggle';
import HistoryDrawer from '../history/HistoryDrawer';
import { useChatStore } from '../../store/useChatStore';
import { useAuth } from '../../context/AuthContext';

type HeaderVariant = 'welcome' | 'chat';
interface HeaderProps {
  variant: HeaderVariant;
  onClear?: () => void;
}

export default function SiteHeader({ variant, onClear }: HeaderProps) {
  const { cid, reset } = useChatStore();
  const { isAuthenticated, login, logout, isLoading } = useAuth();

  /* nav icon */
  const LeftIcon =
    variant === 'chat' ? HomeIcon : ChatBubbleLeftRightIcon;
  const leftHref = variant === 'chat' ? '/' : '/chat';

  /* new-chat handler */
  const handleNewChat = async () => {
    if (cid) {
      fetch(`${import.meta.env.VITE_API_URL}/chat/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: cid }),
      }).catch(console.error);
    }
    reset();
    window.scrollTo({ top: 0 });
  };

  return (
    <div className="sticky top-0 z-20 w-full bg-gray-50 dark:bg-gray-900 border-b dark:border-gray-700">
      <div className="mx-auto max-w-screen-lg flex items-center justify-between px-4 py-2">
        {/* left */}
        <div className="flex items-center gap-2">
          <Link to={leftHref}>
            <LeftIcon className="h-5 w-5 text-gray-400 hover:text-mm-accent" />
          </Link>
          <h1 className="font-bold text-sm sm:text-base text-gray-900 dark:text-mm-accent">
            MediMaven&nbsp;AI
          </h1>
        </div>

        {/* right cluster */}
        <div className="flex items-center gap-3">
          {variant === 'chat' && (
            <>
              <button onClick={handleNewChat} title="New chat">
                <ArrowPathIcon className="h-5 w-5 text-gray-400 hover:text-mm-accent" />
              </button>
              {onClear && (
                <button onClick={onClear} title="Clear chat">
                  <TrashIcon className="h-5 w-5 text-gray-400 hover:text-red-500" />
                </button>
              )}
            </>
          )}

          {/* Dark mode */}
          <DarkToggle />

          {/* History drawer trigger (handles its own auth) */}
          <HistoryDrawer />

          {/* Login / Logout */}
          {!isLoading && (
            isAuthenticated ? (
              <button onClick={logout} title="Logout">
                <LogOutIcon className="h-5 w-5 text-gray-400 hover:text-red-500" />
              </button>
            ) : (
              <button onClick={login} title="Login">
                <LogInIcon className="h-5 w-5 text-gray-400 hover:text-mm-accent" />
              </button>
            )
          )}
        </div>
      </div>
    </div>
  );
}



