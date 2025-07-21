// src/context/AuthContext.tsx
import React from 'react';
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react';
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '../store/useChatStore';

const domain   = import.meta.env.VITE_AUTH0_DOMAIN!;
const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID!;
const audience = import.meta.env.VITE_AUTH0_AUDIENCE!;

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();

  return (
    <Auth0Provider
      domain={domain}
      clientId={clientId}
      authorizationParams={{
        audience,
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

export const useAuth = () => {
  const { isAuthenticated, isLoading,
          loginWithRedirect, logout: rawLogout,
          getAccessTokenSilently, user } = useAuth0();
  const navigate = useNavigate();

  const login = () => {
    // <— simply call the provider’s configured redirect/audience
    return loginWithRedirect();
  };

  const logout = () => {
    rawLogout({
      logoutParams: {
        federated: true,
        // next return to /chat
        returnTo: window.location.origin + '/chat',
      },
    });
    useChatStore.getState().reset();
  };

  return {
    isAuthenticated,
    isLoading,
    user,
    login,
    logout,
    getAccessToken: getAccessTokenSilently,
  };
};