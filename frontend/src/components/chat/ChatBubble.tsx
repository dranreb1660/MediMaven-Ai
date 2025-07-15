import { Message } from '../../types/chat';
import { Citation } from '../../lib/api';
import TypingDots from '../ui/TypingDots';
import Bubble from '../ui/Bubble';
import Badge from '../ui/Badge';

interface BubbleProps {
  role: Message['role'];
  content: string;
  meta?: Message['meta'];
  retry?: () => void;
}

const Citations = ({ cits }: { cits?: Citation[] }) =>
  cits?.length ? (
    <sup className="ml-1">
      {cits.map((c, i) => (
        <a
          key={c.id}
          href={c.url ?? '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs hover:underline"
        >
          [{i + 1}]
        </a>
      ))}
    </sup>
  ) : null;

export default function ChatBubble({ role, content, meta, retry }: BubbleProps) {
  const isUser = role === 'user';
  return (
    <Bubble variant={isUser ? 'user' : 'assistant'}>
      {content}
      {meta?.streaming && (
        <span className="inline-flex items-center gap-1">
          <span className="text-xs text-gray-400">Thinking</span>
          <TypingDots />
        </span>
      )}
      {!isUser && meta?.citations && <Citations cits={meta.citations} />}
      {meta?.latency && <Badge>{meta.latency}s</Badge>}
      {meta?.error && retry && (
        <button onClick={retry} className="mt-2 text-xs text-blue-600 hover:underline">
          Retry
        </button>
      )}
    </Bubble>
  );
}
