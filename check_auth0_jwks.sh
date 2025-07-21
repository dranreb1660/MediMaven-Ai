#!/bin/bash

# Script to safely fetch Auth0 JWKS with proper error handling
# This prevents the jq parse error you encountered

AUTH0_DOMAIN="${1:-medimaven-dev.us.auth0.com}"
JWKS_URL="https://$AUTH0_DOMAIN/.well-known/jwks.json"

echo "Checking Auth0 JWKS for domain: $AUTH0_DOMAIN"
echo "URL: $JWKS_URL"
echo "----------------------------------------"

# Step 1: Test if the domain resolves
if ! nslookup "$AUTH0_DOMAIN" >/dev/null 2>&1; then
    echo "❌ ERROR: Domain '$AUTH0_DOMAIN' does not resolve"
    echo "   This is likely why you're getting the jq parse error"
    exit 1
fi

# Step 2: Fetch the response with proper error handling
echo "Fetching JWKS..."
response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$JWKS_URL" 2>/dev/null)

# Extract HTTP status and body
http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')

echo "HTTP Status: $http_code"

# Step 3: Check if the request was successful
if [ "$http_code" -ne 200 ]; then
    echo "❌ ERROR: HTTP $http_code response"
    echo "Raw response: $body"
    exit 1
fi

# Step 4: Validate JSON before piping to jq
if echo "$body" | jq . >/dev/null 2>&1; then
    echo "✅ SUCCESS: Valid JSON response"
    echo "JWKS content:"
    echo "$body" | jq .
else
    echo "❌ ERROR: Invalid JSON response"
    echo "Raw response: $body"
    exit 1
fi
