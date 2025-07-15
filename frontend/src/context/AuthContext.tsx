import React from 'react';
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react';
import { useNavigate } from 'react-router-dom';

const domain     = import.meta.env.VITE_AUTH0_DOMAIN!;
const clientId   = import.meta.env.VITE_AUTH0_CLIENT_ID!;
const audience   = import.meta.env.VITE_AUTH0_AUDIENCE!;

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();

  return (
    <Auth0Provider
      domain={domain}
      clientId={clientId}
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience,
      }}
      onRedirectCallback={(appState) => navigate(appState?.returnTo || '/')}
      cacheLocation="localstorage"       /* keeps session across tabs */
    >
      {children}
    </Auth0Provider>
  );
}

/* typed helpers the rest of the app can import */
export const useAuth = () => {
  const {
    isAuthenticated,
    isLoading,
    loginWithRedirect,
    logout,
    getAccessTokenSilently,
    user,
  } = useAuth0();

  return {
    isAuthenticated,
    isLoading,
    user,
    login: () => loginWithRedirect(),
    logout: () => logout({ logoutParams: { returnTo: window.location.origin } }),
    getAccessToken: getAccessTokenSilently,
  };
};
