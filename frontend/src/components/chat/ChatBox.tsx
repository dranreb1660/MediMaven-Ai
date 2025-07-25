import { useRef, useState, useEffect } from 'react';
import { useChat } from '../../hooks/useChat';
import MessageList from './MessageList';
import SiteHeader from '../ui/SiteHeader';
import Button from '../ui/Button';
import Container from '../layout/Container';
import Disclaimer from '../ui/Disclaimer';

type Props = { sendMessage?: string };

export default function ChatBox({ sendMessage: initialMessage }: Props) {
  const { messages, sendMessage, retryLast, clearChat, isTyping, error } = useChat();
  const [input, setInput] = useState('');
  const [online, setOnline] = useState(navigator.onLine);
  
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sentRef = useRef(false);

  // Send initial message once
  useEffect(() => {
    if (initialMessage && sendMessage && !sentRef.current) {
      sentRef.current = true;
      setTimeout(() => sendMessage(initialMessage), 100);
      window.history.replaceState(null, '', '/chat');
    }
  }, [initialMessage, sendMessage]);

  // Online/offline detection
  useEffect(() => {
    const handleOnline = () => setOnline(true);
    const handleOffline = () => setOnline(false);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isTyping || !online) return;
    sendMessage(text);
    setInput('');
    setTimeout(() => inputRef.current?.focus(), 80);
  };

  return (
    <Container>
      <div className="flex justify-center w-full min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="relative min-h-screen w-full max-w-2xl mx-auto flex flex-col bg-white dark:bg-gray-800">
          
          <SiteHeader variant="chat" onClear={clearChat} />

          {!online && (
            <div className="bg-red-100 dark:bg-red-900 px-4 py-2 text-center text-red-700 dark:text-red-300 text-sm">
              🔴 No internet connection
            </div>
          )}

          <div className="flex-1 flex flex-col overflow-y-auto">
            <div className="flex-1 px-4 pb-4 overflow-y-auto">
              <MessageList messages={messages} isTyping={isTyping} retry={retryLast} />
              {error && (
                <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                  <p className="text-red-700 dark:text-red-300 text-sm">⚠️ {error}</p>
                  <Button variant="outline" size="sm" onClick={retryLast} className="mt-2">
                    Try Again
                  </Button>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
            {messages.length === 0 && <Disclaimer />}
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 p-4 pb-safe">
            <div className="flex gap-3 items-end">
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                placeholder="Ask your health question..."
                disabled={isTyping || !online}
                maxLength={1000}
                rows={1}
                className="flex-1 max-h-32 resize-none border-2 border-gray-300 dark:border-gray-600 rounded-full px-5 py-3.5 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-base focus:outline-none focus:ring-2 focus:ring-mm-accent focus:border-transparent disabled:opacity-50 transition-all duration-200"
                style={{ minHeight: '52px' }}
              />
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isTyping || !online}
                className="h-[52px] w-[52px] rounded-full p-0 flex items-center justify-center shadow-lg hover:shadow-xl transition-shadow"
                aria-label="Send message"
              >
                {isTyping ? (
                  <div className="animate-pulse">⏳</div>
                ) : (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                )}
              </Button>
            </div>
            {(isTyping || !online) && (
              <p className="text-center text-xs text-gray-500 mt-2 animate-fade-in">
                {isTyping ? '🤖 AI is thinking…' : '🔴 Check your connection'}
              </p>
            )}
          </div>
        </div>
      </div>
    </Container>
  );
}
