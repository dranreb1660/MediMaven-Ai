import type { Message } from '../../types/chat';

type BubbleProps = Message & { retry: () => void };

export default function ChatBubble({ role, content, meta, retry }: BubbleProps) {
  const isUser = role === 'user';

  return (
    <div
      className={`flex ${
        isUser ? 'justify-end sm:justify-end' : 'justify-start sm:justify-start'
      } px-0 sm:px-4`}
    >
      <div
        className={`
          relative w-full sm:max-w-[80%] p-4 my-2 rounded-3xl text-base leading-relaxed
          ${isUser
            ? 'bg-mm-bubble text-gray-900 dark:bg-mm-bubble/80'
            : 'bg-mm-bg text-gray-800 dark:bg-gray-700'}
          before:absolute before:top-3 before:w-0 before:h-0
          ${isUser
            ? 'before:right-full before:border-y-8 before:border-r-8 before:border-y-transparent before:border-r-mm-bubble'
            : 'before:left-full before:border-y-8 before:border-l-8 before:border-y-transparent before:border-l-mm-bg dark:before:border-l-gray-700'}
        `}
      >
        {content}

        {meta?.latency && (
          <span className="block mt-1 text-xs text-gray-500">{meta.latency}s</span>
        )}

        {meta?.error && (
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
