import {type Message } from '../../types/chat';
import { Citation } from '../../lib/api';
import TypingDots from '../ui/TypingDots';

interface BubbleProps {
  role: Message['role'];
  content: string;
  meta?: Message['meta'];
  retry?: () => void;
}

function Citations({ cits }: { cits?: Citation[] }) {
  if (!cits?.length) return null;
  return (
    <sup className="ml-1">
      {cits.map((c, i) => (
        <a
          key={c.id}
          href={c.url ?? '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs hover:underline"
        >
          [{i + 1}]
        </a>
      ))}
    </sup>
  );
}

export default function ChatBubble({ role, content, meta, retry }: BubbleProps) {
  const isUser = role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} my-2`}>
      <div
        className={`relative w-full sm:max-w-[80%] p-4 rounded-3xl text-base leading-relaxed
          ${isUser
            ? 'bg-mm-bubble-user text-gray-900 dark:bg-mm-bubble/80'
            : 'bg-mm-bubble text-gray-800 dark:bg-gray-700'}
          before:absolute before:top-3 before:w-0 before:h-0
          ${isUser
            ? 'before:right-full before:border-y-8 before:border-r-8 before:border-l-0 before:border-r-transparent'
            : 'before:left-full before:border-y-8 before:border-l-8 before:border-r-0 before:border-l-transparent'}
        `}
      >
        {content}
        {meta?.streaming && (
          <span className="inline-flex items-center gap-1">
            <span className="text-xs text-gray-400">Thinking</span>
            <TypingDots />
          </span>
        )}
        {role === 'assistant' && <Citations cits={meta?.citations} />}

        {meta?.latency && (
          <span className="block mt-1 text-xs text-gray-500">{meta.latency}s</span>
        )}

        {meta?.error && retry && (
          <button
            onClick={retry}
            className="mt-2 text-xs text-blue-600 hover:underline"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
