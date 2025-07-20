import { useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import ChatBox from '../components/chat/ChatBox';

export default function ChatPage() {
  const location = useLocation();
  const state = useMemo(
    () => location.state as { prefill?: string } | null,
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  return <ChatBox prefill={state?.prefill} />;
}