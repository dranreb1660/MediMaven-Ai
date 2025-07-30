// src/context/AuthContext.tsx
import React from 'react';
import { Auth0Provider } from '@auth0/auth0-react';
import { useNavigate } from 'react-router-dom';
import { AUTH_CONFIG } from './auth-constants';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();

  return (
    <Auth0Provider
      domain={AUTH_CONFIG.domain}
      clientId={AUTH_CONFIG.clientId}
      authorizationParams={{
        audience: AUTH_CONFIG.audience,
        redirect_uri: window.location.origin + '/chat',
      }}
      onRedirectCallback={() => {
        navigate('/chat');
      }}
      cacheLocation="localstorage"
      useRefreshTokens={true}
    >
      {children}
    </Auth0Provider>
  );
}

