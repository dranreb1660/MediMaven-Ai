import ChatBubble from './ChatBubble';
import type { Message } from '../../types/Types';
import { useAutoScroll } from '../../hooks/useAutoScroll';

export interface MessageListProps {
  messages: Message[];
  isTyping: boolean;
  retry: () => void;
}

export default function MessageList({
  messages,
  isTyping,
  retry,
}: MessageListProps) {
  const bottomRef = useAutoScroll({
    dependencies: [messages, isTyping]
  });

  return (
    <div className="flex flex-col flex-1 overflow-y-auto px-0 sm:px-4">
      {messages.map((m, idx) => (
        /*  fallback to array index when m.id is missing  */
        <ChatBubble key={m.id ?? `row-${idx}`} {...m} retry={retry} />
      ))}

      {isTyping && (
        <div className="flex justify-start px-4 my-2">
          <div className="w-32 h-6 bg-gray-200 dark:bg-gray-700 rounded-3xl animate-pulse" />
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
