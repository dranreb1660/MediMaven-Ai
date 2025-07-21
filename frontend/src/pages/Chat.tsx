import { useLocation } from 'react-router-dom';
import ChatBox from '../components/chat/ChatBox';

export default function Chat() {
  const location = useLocation();
  const message = (location.state as { sendMessage?: string } | null)?.sendMessage;
  
  return <ChatBox sendMessage={message} />;
}
