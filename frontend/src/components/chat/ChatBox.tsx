import { useRef, useState, useEffect } from 'react';
import { useChat } from '../../hooks/useChat';
import MessageList from './MessageList';
import SiteHeader from '../ui/SiteHeader';
import Button from '../ui/Button';
import Container from '../layout/Container';




type Props = { prefill?: string };

export default function ChatBox({ prefill }: Props) {
  const { messages, sendMessage, retryLast, clearChat, isTyping, error } = useChat();
  const [input, setInput] = useState(prefill ?? '');
  const bottomRef = useRef<HTMLDivElement>(null);

  // one‑shot prefill
  useEffect(() => {
    if (prefill) sendMessage(prefill);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // auto‑scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const onSend = () => {
    if (input.trim()) {
      sendMessage(input);
      setInput('');
    }
  };

  return (
    <Container>
      <div className="flex justify-center w-full dark:bg-gray-900">
        {/* centred & capped column */}
        <div className="relative min-h-screen w-full max-w-screen-lg flex flex-col bg-gray-50 dark:bg-gray-900">
          {/* sticky header controls */}
          <SiteHeader variant="chat" onClear={clearChat} />

        <div className="flex-1 overflow-y-auto px-2 sm:px-4 pb-6">
          <MessageList messages={messages} isTyping={isTyping} retry={retryLast} />
          {error && <div className="mt-2 text-red-600 text-sm">{error}</div>}
          <div ref={bottomRef} />
        </div>

        {/* input row */}
      <div className="flex items-center gap-2 px-4 py-3 border-t dark:border-gray-700">
        <input
          aria-label='Ask your health question...'
          type="text"
          placeholder="Ask your health question..."
          className="flex-1 border dark:border-gray-600 rounded-full px-4 py-2 text-m dark:text-gray-300 bg-white dark:bg-gray-800 focus:outline-none"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && onSend()}
        />
        <Button disabled={isTyping} onClick={onSend}>
          Send
        </Button>
      </div>
      </div>
    </div>
  </Container>
  );
}
