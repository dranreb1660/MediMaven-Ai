import { useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import ChatBox from '../components/chat/ChatBox';

export default function ChatPage() {
  const location = useLocation();

  /** Grab the message passed from WelcomeCard (only on first mount). */
  const sendMsgOnce = useMemo(
    () => (location.state as { sendMessage?: string } | null)?.sendMessage,
    // dependency intentionally empty – we only want initial value
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  return <ChatBox sendMessage={sendMsgOnce} />;
}
