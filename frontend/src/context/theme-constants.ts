// src/context/theme-constants.ts
import { createContext } from 'react';

export const ThemeCtx = createContext({ dark: false, toggle: () => {} });
