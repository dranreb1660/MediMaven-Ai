import { useEffect, useRef } from 'react';
import ChatBubble from './ChatBubble';
import type { Message } from '../../types/chat';

type Props = {
  messages: Message[];
  isTyping: boolean;
  retry: () => void;
};

export default function MessageList({ messages, isTyping, retry }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="flex flex-col flex-1 overflow-y-auto px-0 sm:px-4">
      {messages.map(m => (
        <ChatBubble key={m.id} {...m} retry={retry} />
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
