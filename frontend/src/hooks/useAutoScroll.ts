import { useRef, useEffect } from 'react';

interface UseAutoScrollOptions {
  /** Dependencies that trigger scroll */
  dependencies: readonly unknown[];
  /** Delay before scrolling in ms. Default: 0 */
  delay?: number;
  /** Scroll behavior. Default: 'smooth' */
  behavior?: ScrollBehavior;
}

export function useAutoScroll<T extends HTMLElement = HTMLDivElement>({
  dependencies,
  delay = 0,
  behavior = 'smooth'
}: UseAutoScrollOptions) {
  const scrollRef = useRef<T>(null);

  useEffect(() => {
    const scrollToTarget = () => {
      scrollRef.current?.scrollIntoView({ behavior });
    };

    if (delay > 0) {
      const timeoutId = setTimeout(scrollToTarget, delay);
      return () => clearTimeout(timeoutId);
    } else {
      scrollToTarget();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, behavior, delay]);

  return scrollRef;
}
