import { useEffect, useRef } from 'react';
import ChatBubble from './ChatBubble';
import { type Message } from '../../types/chat';

interface MessageListProps {
  messages: Message[];
}

export default function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col gap-2 px-4 py-3 overflow-auto flex-1">
      {messages.map((msg) => (
        <ChatBubble
          key={msg.id}
          role={msg.role}
          content={msg.content}
          meta={msg.meta}
        />
      ))}
      <div ref={bottomRef} /> {/* 👈 Scroll target */}
    </div>
  );
}
