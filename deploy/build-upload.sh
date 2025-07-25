#!/usr/bin/env bash
set -e
VER=$(git rev-parse --short HEAD)
npm ci
npm run build                      # outputs to dist/
aws s3 sync dist/ s3://medimaven-web/$VER --delete
aws s3 cp  dist/index.html s3://medimaven-web/ --cache-control "max-age=60" --content-type text/html
