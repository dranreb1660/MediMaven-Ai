import { motion } from 'framer-motion';

export default function TypingDots() {
  return (
    <motion.span
      className="inline-flex gap-[2px] ml-1"
      aria-label="assistant is typing"
      initial={{ opacity: 0.2 }}
      animate={{ opacity: 1 }}
      transition={{ repeat: Infinity, repeatType: 'reverse', duration: 0.6 }}
    >
      <span className="w-1 h-1 bg-current rounded-full" />
      <span className="w-1 h-1 bg-current rounded-full" />
      <span className="w-1 h-1 bg-current rounded-full" />
    </motion.span>
  );
}
