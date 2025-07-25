// src/hooks/useAuth.ts
import { useAuth0 } from '@auth0/auth0-react';
import { useChatStore } from '../store/useChatStore';

export const useAuth = () => {
  const {
    isAuthenticated, isLoading, user,
    loginWithRedirect,
    logout: auth0Logout,
    getAccessTokenSilently,
  } = useAuth0();

  /** Hard logout without nuking Google account */
  const logout = () =>
    auth0Logout({
      logoutParams: {
        returnTo: window.location.origin,   // ← where to land
        // ❌ NO “federated: true” ⇒ leaves Google session intact
      },
    });

  /** Always show the Auth0 UI */
  const login = () =>
    loginWithRedirect({
      authorizationParams: {
        audience:   import.meta.env.VITE_AUTH0_AUDIENCE,
        prompt:     'login',          // or 'select_account'
        // max_age: 0,                // alternative to prompt
      },
    });

  return {
    isAuthenticated,
    isLoading,
    user,
    login,
    logout,
    getAccessToken: getAccessTokenSilently,
  };
};