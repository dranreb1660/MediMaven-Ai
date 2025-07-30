// src/context/DrawerContext.tsx        🆕 NEW
import { createContext, useState, ReactNode, useCallback, useEffect } from 'react';

export interface DrawerCtx {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

export const DrawerCtx = createContext<DrawerCtx | null>(null);

export function DrawerProvider({ children }: { children: ReactNode }) {
  const [isOpen, set] = useState(false);
  const open  = useCallback(() => set(true), []);
  const close = useCallback(() => set(false), []);
  const toggle = useCallback(() => set(s => !s), []);
  
  useEffect(()=>{ document.body.classList.toggle('drawer-open',isOpen); },[isOpen]);

  return <DrawerCtx.Provider value={{ isOpen, open, close, toggle }}>{children}</DrawerCtx.Provider>;
}

