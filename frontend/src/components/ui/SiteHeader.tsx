import { Link, useLocation } from 'react-router-dom';
import { Home, MessageSquare, MessageCirclePlus, Menu} from 'lucide-react';
import { useDrawer } from '../../hooks/useDrawer';

type HeaderVariant = 'welcome' | 'chat';
interface HeaderProps {
  variant: HeaderVariant;
  onClear?: () => void;
}

export default function SiteHeader({ variant, onClear }: HeaderProps) {
  const { pathname } = useLocation();
  const { toggle }   = useDrawer();

  /* left icon swaps automatically */
  const isOnHome   = pathname === '/';
  const LeftIcon   = isOnHome ? MessageSquare : Home;
  const leftHref   = isOnHome ? '/chat' : '/';

  return (
    <header className="sticky top-0 z-30 flex items-center gap-2 sm:gap-3
                       px-4 py-4 bg-white dark:bg-gray-900 text-gray-900 dark:text-white 
                       shadow-sm border-b border-gray-200 dark:border-gray-800">
      {/* ––––– left cluster ––––– */}
      <Link to={leftHref} aria-label={isOnHome ? 'Open chat' : 'Home'} 
            className="p-2 -m-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
        <LeftIcon size={22} className="text-mm-accent" />
      </Link>

      <h1 className="flex-1 font-bold text-mm-accent text-lg">
        MediMaven
      </h1>

      {/* ––––– centre actions – chat page only ––––– */}
      {variant === 'chat' && onClear && (
        <button onClick={onClear} title="New chat" 
                className="p-2 -m-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
          <MessageCirclePlus size={20} className="text-gray-600 dark:text-gray-400" />
        </button>
      )}

      {/* ––––– dark-mode toggle (always visible) ––––– */}
      {/* <DarkToggle className="hidden sm:inline-flex" /> */}

      {/* ––––– drawer toggle ––––– */}
      <button onClick={toggle} title="Menu" 
              className="p-2 -m-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
        <Menu size={24} className="text-gray-600 dark:text-gray-400"/>
      </button>
    </header>
  );
}
