import * as HoverCard from '@radix-ui/react-hover-card';

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
  const formattedTitle = formatUrlTitle(url);

  return (
    <HoverCard.Root openDelay={150} closeDelay={150}>
      <HoverCard.Trigger asChild>
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs underline cursor-pointer text-mm-accent-600"
        >
          {children}
        </a>
      </HoverCard.Trigger>

      <HoverCard.Portal>
        <HoverCard.Content
          sideOffset={6}
          align="center"
          className="max-w-xs rounded-md bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-3 text-xs text-gray-800 dark:text-gray-200 shadow-xl animate-fade-in"
        >
          <strong className="text-sm font-medium">{formattedTitle} - <span className="text-mm-accent-600 dark:text-mm-accent-600">{source}</span></strong>
          <p className="mt-1 text-[11px] text-gray-600 dark:text-gray-400">
            Click to view detailed information from <strong>{source}</strong>.
          </p>
          <HoverCard.Arrow className="fill-current text-gray-200 dark:text-gray-700" />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}
