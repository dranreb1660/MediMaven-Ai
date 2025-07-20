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
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Handle online/offline and prefill
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    if (prefill) setTimeout(() => sendMessage(prefill), 100);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const onSend = () => {
    const message = input.trim();
    if (!message || isTyping || !isOnline) return;
    
    sendMessage(message);
    setInput('');
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const handleClearChat = () => {
    if (messages.length && confirm('Clear chat? This cannot be undone.')) {
      clearChat();
    }
  };

  return (
    <Container>
      <div className="flex justify-center w-full min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="relative min-h-screen w-full max-w-2xl mx-auto flex flex-col bg-white dark:bg-gray-800">
          
          <SiteHeader variant="chat" onClear={handleClearChat} />

          {/* Connection status */}
          {!isOnline && (
            <div className="bg-red-100 dark:bg-red-900 px-4 py-2 text-center text-red-700 dark:text-red-300 text-sm">
              🔴 No internet connection
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 pb-4">
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

          {/* Input */}
          <div className="border-t border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-end gap-3">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), onSend())}
                placeholder="Ask your health question..."
                disabled={isTyping || !isOnline}
                maxLength={1000}
                rows={1}
                className="flex-1 max-h-32 resize-none border-2 border-gray-300 dark:border-gray-600 rounded-2xl px-4 py-3 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-mm-accent disabled:opacity-50"
                style={{ minHeight: '48px' }}
              />
              
              <Button 
                onClick={onSend}
                disabled={!input.trim() || isTyping || !isOnline}
                className="px-6 py-3"
              >
                {isTyping ? '⏳' : '📤'}
              </Button>
            </div>
            
            {(isTyping || !isOnline) && (
              <p className="text-center text-xs text-gray-500 mt-2">
                {isTyping ? '🤖 AI is thinking...' : '🔴 Check your connection'}
              </p>
            )}
          </div>
        </div>
      </div>
    </Container>
  );
}