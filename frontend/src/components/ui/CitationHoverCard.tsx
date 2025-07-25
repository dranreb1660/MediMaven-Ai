import * as HoverCard from '@radix-ui/react-hover-card';
import { useState, useEffect } from 'react';

interface CitationHoverCardProps {
  url: string;
  source: string;
  children: React.ReactNode;
}

function formatUrlTitle(url: string) {
  const segments = url.split('/').filter(Boolean);
  if (!segments.length) return url;

  const lastSegment = segments[segments.length - 1].replace(/-/g, ' ');
  return lastSegment.charAt(0).toUpperCase() + lastSegment.slice(1);
}

export default function CitationHoverCard({
  url,
  source,
  children,
}: CitationHoverCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isTouchDevice, setIsTouchDevice] = useState(false);
  const formattedTitle = formatUrlTitle(url);

  useEffect(() => {
    setIsTouchDevice('ontouchstart' in window || navigator.maxTouchPoints > 0);
  }, []);

  const handleCardClick = () => {
    window.open(url, '_blank', 'noopener,noreferrer');
    setIsOpen(false);
  };

  // For desktop, handle hover
  const handleOpenChange = (open: boolean) => {
    if (!isTouchDevice) {
      setIsOpen(open);
    }
  };

  return (
    <HoverCard.Root 
      open={isOpen}
      onOpenChange={handleOpenChange}
      openDelay={150} 
      closeDelay={150}
    >
      <HoverCard.Trigger asChild>
        <button
          type="button"
          onClick={() => {
            if (isTouchDevice) {
              setIsOpen(prev => !prev);
            }
          }}
          className="text-xs underline cursor-pointer text-mm-accent-600 dark:text-mm-accent-500 
                     hover:text-mm-accent-700 dark:hover:text-mm-accent-400 transition-colors
                     bg-transparent border-none p-0 m-0 font-inherit inline"
        >
          {children}
        </button>
      </HoverCard.Trigger>

      <HoverCard.Portal>
        <HoverCard.Content
          sideOffset={8}
          align="center"
          className="max-w-xs rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 
                     p-4 shadow-xl animate-fade-in z-50"
          onInteractOutside={() => setIsOpen(false)}
        >
          <div 
            onClick={handleCardClick}
            className="cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 -m-4 p-4 rounded-lg transition-colors"
          >
            <div className="flex items-start gap-2">
              <svg className="w-4 h-4 text-mm-accent-600 dark:text-mm-accent-500 mt-0.5 flex-shrink-0" 
                   fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <div className="flex-1">
                <strong className="text-sm font-semibold text-gray-900 dark:text-white block">
                  {formattedTitle}
                </strong>
                <span className="text-xs text-mm-accent-600 dark:text-mm-accent-500">
                  {source}
                </span>
              </div>
            </div>
            <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              {isTouchDevice ? 'Tap' : 'Click'} to view source
            </p>
          </div>
          <HoverCard.Arrow className="fill-white dark:fill-gray-800" />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}