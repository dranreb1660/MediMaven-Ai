import { useRef, useState, useEffect } from 'react';
import { useChat } from '../../hooks/useChat';
import MessageList from './MessageList';
import SiteHeader from '../ui/SiteHeader';



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
            type="text"
            placeholder="Ask your Heaalth question..."
            className="flex-1 border dark:border-gray-600 rounded-full px-4 py-2 text-sm bg-white dark:bg-gray-800 focus:outline-none"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onSend()}
          />
          <button
            disabled={isTyping}
            onClick={onSend}
            className={`rounded-full px-6 py-2 text-sm font-semibold transition ${
              isTyping
                ? 'bg-gray-300 dark:bg-gray-700 cursor-not-allowed'
                : 'bg-mm-accent hover:bg-mm-accentDark text-white'
            }`}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
