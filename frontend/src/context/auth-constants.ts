// src/context/auth-constants.ts

export const AUTH_CONFIG = {
  domain: import.meta.env.VITE_AUTH0_DOMAIN!,
  clientId: import.meta.env.VITE_AUTH0_CLIENT_ID!,
  audience: import.meta.env.VITE_AUTH0_AUDIENCE!,
} as const;
