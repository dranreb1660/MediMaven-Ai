import { Message } from '../../types/Types';
import { Citation } from '../../types/Types';
import TypingDots from '../ui/TypingDots';
import Bubble from '../ui/Bubble';
import Badge from '../ui/Badge';
import CitationHoverCard from '../ui/CitationHoverCard';



interface BubbleProps {
  role: Message['role'];
  content: string;
  meta?: Message['meta'];
  retry?: () => void;
}

const Citations = ({ cits }: { cits?: Citation[] }) =>
  cits?.length ? (
    <span className="inline-flex items-baseline ml-1">
      {cits.map((c, i) => (
        <sup key={c.id} className="ml-0.5">
          <CitationHoverCard url={c.url ?? '#'} source={c.source}>
            [{i + 1}]
          </CitationHoverCard>
        </sup>
      ))}
    </span>
  ) : null;


export default function ChatBubble({ role, content, meta, retry }: BubbleProps) {
  const isUser = role === 'user';
  return (
    <Bubble variant={isUser ? 'user' : 'assistant'}>
      <div className="text-base leading-relaxed">
        {content}
        {meta?.streaming && (
          <span className="inline-flex items-center gap-1 ml-2">
            <TypingDots />
          </span>
        )}
        {!isUser && meta?.citations && <Citations cits={meta.citations} />}
      </div>
      {meta?.latency && (
        <div className="mt-2">
          <Badge>{meta.latency}s</Badge>
        </div>
      )}
      {meta?.error && retry && (
        <button 
          onClick={retry} 
          className="mt-3 px-4 py-2 text-sm bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/30 transition-colors"
        >
          Try Again
        </button>
      )}
    </Bubble>
  );
}
