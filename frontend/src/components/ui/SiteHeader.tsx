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
import { useChatStore } from '../../store/useChatStore';
import { useAuth } from '../../hooks/useAuth';
import MyHistoryControl from '../history/MyHistoryControl';

type HeaderVariant = 'welcome' | 'chat';
interface HeaderProps {
  variant: HeaderVariant;
  onClear?: () => void;
}

export default function SiteHeader({ variant, onClear }: HeaderProps) {
  const { cid, reset } = useChatStore();
  const { isAuthenticated, login, logout, isLoading } = useAuth();

  const LeftIcon = variant === 'chat' ? HomeIcon : ChatBubbleLeftRightIcon;
  const leftHref = variant === 'chat' ? '/' : '/chat';

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

  const IconButton = ({ icon: Icon, onClick, title, className = "" }: {
    icon: typeof HomeIcon;
    onClick: () => void;
    title: string;
    className?: string;
  }) => (
    <button onClick={onClick} title={title} className={`p-1 ${className}`}>
      <Icon className="h-5 w-5 text-gray-400 hover:text-mm-accent transition-colors" />
    </button>
  );

  return (
    <div className="sticky top-0 z-20 w-full bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm">
      <div className="max-w-screen-lg mx-auto flex items-center justify-between px-4 py-2">
        
        {/* Left: Navigation + Logo */}
        <div className="flex items-center gap-3">
          <Link to={leftHref}>
            <LeftIcon className="h-5 w-5 text-gray-400 hover:text-mm-accent transition-colors" />
          </Link>
          <h1 className="font-bold text-sm sm:text-base text-gray-900 dark:text-mm-accent">
            MediMaven AI
          </h1>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          {variant === 'chat' && (
            <>
              <IconButton 
                icon={ArrowPathIcon} 
                onClick={handleNewChat} 
                title="New chat" 
              />
              {onClear && (
                <IconButton 
                  icon={TrashIcon} 
                  onClick={onClear} 
                  title="Clear chat"
                  className="hover:text-red-500"
                />
              )}
            </>
          )}

          <DarkToggle />
          <MyHistoryControl />

          {!isLoading && (
            <IconButton
              icon={isAuthenticated ? LogOutIcon : LogInIcon}
              onClick={isAuthenticated ? logout : login}
              title={isAuthenticated ? "Logout" : "Login"}
              className={isAuthenticated ? "hover:text-red-500" : ""}
            />
          )}
        </div>
      </div>
    </div>
  );
}