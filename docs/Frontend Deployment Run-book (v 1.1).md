# 🌐 MediMaven Frontend Deployment Runbook (v1.1)

![CI](https://img.shields.io/badge/Built_with-Docker-blue) ![AWS](https://img.shields.io/badge/Cloud-AWS-%23FF9900) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

_Last updated 25 Jul 2025_

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Local Build](#local-build)
3.  [Manual S3 Sync](#manual-s3-sync)
4.  [CI/CD Workflow](#cicd-workflow)
5.  [CloudFront Behaviors](#cloudfront-behaviors)
6.  [Security Headers](#security-headers)
7.  [Troubleshooting](#troubleshooting)
8.  [Routine Deploy Flow](#routine-deploy-flow)

---

## 0 · Prerequisites (one-time AWS setup)

| Resource | Purpose | Key Settings |
|----------|---------|--------------|
| **S3 bucket** `medimaven-web` | Stores built assets | • _Block_ **all** public access<br>• Static-website hosting **disabled** |
| **CloudFront distribution** `E3CAF6LNQB5FF0` | Global TLS edge cache | Origin ⇒ the S3 bucket<br>Origin Access **OAC** |
| **ACM cert** (`us-east-1`) | HTTPS for `www.medimaven-ai.com` | Attach to CloudFront |
| **Route 53 record** | DNS | A/AAAA (alias) → CF distribution |
| **IAM deploy user** | GitHub Actions upload rights | `s3:PutObject/DeleteObject` on bucket + `cloudfront:CreateInvalidation` |
| **Repo secrets** | CI credentials | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

_Optional hardening_  
* Lambda@Edge **`addSecurityHeaders`** (viewer-response)  
* CF Function  **`spaRewrite`** (viewer-request → rewrite ↦ `/index.html`)

---

## 1 · Local build (fallback manual)

```bash
cd frontend
npm ci
npm run build          # → frontend/dist
```

## 2 · Manual S3 sync (fallback)
```bash
# Long-lived assets

aws s3 sync frontend/dist s3://medimaven-web \
  --delete --exclude "*.html" --exclude "service-worker.js" \
  --cache-control "public,max-age=31536000,immutable"

# HTML + service-worker
aws s3 sync frontend/dist s3://medimaven-web \
  --exclude "*" --include "*.html" --include "service-worker.js" \
  --cache-control "public,max-age=300,must-revalidate"
```

## 3 · CI/CD workflow
```github/workflows/deploy.yml```


## 4 · CloudFront behaviours

| Path pattern                        | Min TTL | Notes             |
| ----------------------------------- | ------- | ----------------- |
| `/index.html`, `/service-worker.js` | **0 s** | Always fresh      |
| `/*` (static)                       | 1 year  | Default behaviour |

## 5 · Security headers (Lambda@Edge – viewer-response)

```js
'use strict';
exports.handler = (event, ctx, cb) => {
  const r = event.Records[0].cf.response;
  const h = r.headers;
  h['strict-transport-security'] = [{ key:'Strict-Transport-Security', value:'max-age=63072000; includeSubDomains; preload' }];
  h['content-security-policy']   = [{ key:'Content-Security-Policy', value:"default-src 'self' https://*.auth0.com https://api.medimaven-ai.com; img-src 'self' data:; object-src 'none'" }];
  h['x-frame-options']           = [{ key:'X-Frame-Options', value:'DENY' }];
  h['x-content-type-options']    = [{ key:'X-Content-Type-Options', value:'nosniff' }];
  h['referrer-policy']           = [{ key:'Referrer-Policy', value:'strict-origin-when-cross-origin' }];
  cb(null, r);
};
```

## 6 · Troubleshooting quick table
| Symptom               | Likely cause                          | Remedy                                  |
| --------------------- | ------------------------------------- | --------------------------------------- |
| White screen, CSS 404 | `/index.html` cached too long         | Re-upload with short cache + invalidate |
| JS 403 `AccessDenied` | Bucket policy missing CF OAC          | Update policy                           |
| SPA routes 404 XML    | `spaRewrite` missing                  | Attach CF Function or add behaviour     |
| Auth redirect 403     | Wrong `AUTH0_DOMAIN` / `AUDIENCE`     | Fix env vars                            |
| CORS error            | FastAPI CORS not allowing prod origin | Update backend CORS list                |


## 7 · Routine deploy flow
1. Dev pushes to main.

2. GitHub Actions runs (tests-Frontend, backend), builds & uploads.

3. CloudFront invalidation completes (≈ 2 min).

4. Users worldwide receive v1.1 with zero downtime.