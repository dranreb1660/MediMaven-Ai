# src/backend/auth.py
import os, httpx
from jose import jwt
from jose.exceptions import JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
import cachetools
from backend.app import config

from fastapi import Header
from typing import Optional

AUTH0_DOMAIN   = config.AUTH0_DOMAIN
AUTH0_AUDIENCE = config.AUTH0_AUDIENCE 


bearer = HTTPBearer()
_jwks_cache = cachetools.TTLCache(maxsize=1, ttl=12 * 60 * 60)

def _get_jwks():
    if "jwks" in _jwks_cache:
        return _jwks_cache["jwks"]

    url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    try:
        r = httpx.get(url, timeout=5)
        r.raise_for_status()
        jwks = r.json()
    except Exception as e:
        print("[AUTH] JWKS fetch failed:", e)
        raise HTTPException(503, "Auth metadata fetch error")
    _jwks_cache["jwks"] = jwks
    return jwks


def get_current_user(token=Depends(bearer)):
    try:
        payload = jwt.decode(
            token.credentials,
            _get_jwks(),
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
        return payload                     # contains `sub`
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


def get_current_user_optional(
    authorization: Optional[str] = Header(None)
):
    """Return user dict if JWT present & valid, else None."""
    if authorization is None:
        print("No Authorization header provided")
        return None
    try:
        print(f"Authorization header--------------------------:\n {authorization}")
        scheme, _, token = authorization.partition(" ")
        user = jwt.decode(
            token,
            _get_jwks(),
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
        print("[AUTH] decoded user--------------------:\n", user["sub"]) 
        return user
    except JWTError as e:
        print("⛔️ ⛔️ ⛔️ ⛔️ [AUTH] JWTError:", e) 
        return None
