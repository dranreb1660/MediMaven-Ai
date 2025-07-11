import { useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import ChatBox from '../components/chat/ChatBox';

export default function ChatPage() {
  const location = useLocation();
  // grab once; after welcome→chat this state is null on reload
  const prefill = useMemo(
    () => (location.state as { prefill?: string } | null)?.prefill,
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  return <ChatBox prefill={prefill} />;
}
