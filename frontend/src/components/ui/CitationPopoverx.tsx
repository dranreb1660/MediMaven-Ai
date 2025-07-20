import * as Popover from '@radix-ui/react-popover';

interface CitationPopoverProps {
  citationNumber: number;
  title: string;
  url?: string;
}

export default function CitationPopover({ citationNumber, title, url }: CitationPopoverProps) {
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <a
          href={url ?? '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs underline cursor-pointer text-mm-accent-600"
        >
          [{citationNumber}]
        </a>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 text-sm max-w-xs z-50 border dark:border-gray-700"
          side="top"
          align="center"
          sideOffset={4}
        >
          {title}
          <Popover.Arrow className="fill-current text-white dark:text-gray-800" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
