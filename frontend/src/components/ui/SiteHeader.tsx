import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Home, MessageSquare, MessageCirclePlus, Trash2, Menu} from 'lucide-react';
import { useDrawer } from '../../context/DrawerContext';
import DarkToggle from './DarkToggle';

type HeaderVariant = 'welcome' | 'chat';
interface HeaderProps {
  variant: HeaderVariant;
  onClear?: () => void;
}

export default function SiteHeader({ variant, onClear }: HeaderProps) {
  const { pathname } = useLocation();
  const navigate     = useNavigate();
  const { toggle }   = useDrawer();

  /* left icon swaps automatically */
  const isOnHome   = pathname === '/';
  const LeftIcon   = isOnHome ? MessageSquare : Home;
  const leftHref   = isOnHome ? '/chat' : '/';

  /* chat-only helpers */
  const handleNewChat = () => navigate(0);   // quick hard-refresh

  const IconButton = ({ icon: Icon, onClick, title, className = "" }: {
    icon: typeof Home | typeof MessageSquare | typeof MessageCirclePlus | typeof Trash2;
    onClick: () => void;
    title: string;
    className?: string;
  }) => (
    <button onClick={onClick} title={title} className={`p-1 ${className}`}>
      <Icon className="h-5 w-5 text-gray-400 hover:text-mm-accent transition-colors" />
    </button>
  );

  return (
    <header className="sticky top-0 z-30 flex items-center gap-3
                       px-4 py-2 bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-white shadow-sm shadow-sm">
      {/* ––––– left cluster ––––– */}
      <Link to={leftHref} aria-label={isOnHome ? 'Open chat' : 'Home'}>
        <LeftIcon size={20} className="text-mm-accent" />
      </Link>

      <h1 className="flex-1 font-extrabold text-mm-accent text-sm sm:text-base">
        MediMaven&nbsp;AI
      </h1>

      {/* ––––– centre actions – chat page only ––––– */}
      {variant === 'chat' && (
        <>
          {onClear && (
            <button onClick={onClear} title="New chat">
              <MessageCirclePlus size={18} className="text-gray-400 hover:text-mm-accent" />
            </button>
          )}
          {onClear && (
            <button onClick={onClear} title="Clear chat">
              <Trash2 size={18} className="text-gray-400 hover:text-red-500" />
            </button>
          )}
        </>
      )}

      {/* ––––– dark-mode toggle (always visible) ––––– */}
      <DarkToggle className="hidden sm:inline-flex" />

      {/* ––––– drawer toggle ––––– */}
      <button onClick={toggle} title="Menu" className="ml-1">
        <Menu size={22} className="text-gray-400 hover:text-red-500"/>
      </button>
    </header>
  );
}
