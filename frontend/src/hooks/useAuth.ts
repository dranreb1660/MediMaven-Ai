// src/hooks/useAuth.ts
import { useAuth0 } from '@auth0/auth0-react';
import { useChatStore } from '../store/useChatStore';

export const useAuth = () => {
  const { isAuthenticated, isLoading,
          loginWithRedirect, logout: rawLogout,
          getAccessTokenSilently, user } = useAuth0();

  const login = () => {
    // ← simply call the provider's configured redirect/audience
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
