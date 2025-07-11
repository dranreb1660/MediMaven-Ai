import { Link } from 'react-router-dom';
import { TrashIcon, HomeIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline';
import DarkToggle from './DarkToggle';

type HeaderVariant = 'welcome' | 'chat';
interface HeaderProps {
  variant: HeaderVariant;
  onClear?: () => void;          // only needed on chat page
}


export default function SiteHeader({ variant, onClear }: HeaderProps) {
  const LeftIcon =
    variant === 'chat' ? HomeIcon                            // ↩︎ back to Welcome
                       : ChatBubbleLeftRightIcon;            // ↦ go to Chat

  const leftHref = variant === 'chat' ? '/' : '/chat';

  return (
    <div className="sticky top-0 z-20 w-full bg-gray-50 dark:bg-gray-900 border-b dark:border-gray-700">
      <div className="mx-auto max-w-screen-lg flex items-center justify-between px-4 py-2">
        {/* left block */}
        <div className="flex items-center gap-2">
          <Link to={leftHref} title={variant === 'chat' ? 'Home' : 'Chat'}>
            <LeftIcon className="w-5 h-5 text-gray-400 hover:text-mm-accent" />
          </Link>
          <div>
            <h1 className=" font-bold text-sm sm:text-base text-gray-900 dark:text-mm-accent">
                MediMaven&nbsp;AI
            </h1>
            <p className=" text-[10px] leading-none text-gray-500 dark:text-gray-400  // subtle but visible">
              …your virtual&nbsp;AI&nbsp;doctor
            </p>

          </div>
        </div>

        {/* right block */}
        <div className="flex items-center gap-3">
          
          {variant === 'chat' && onClear && (
            <button onClick={onClear} title="Clear chat">
              <TrashIcon className="w-5 h-5 w-5 text-gray-400 dark:text-gray-500 hover:text-red-500" />
            </button>
          )}
          <DarkToggle />
        </div>
      </div>
    </div>
  );
}
