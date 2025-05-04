# MediMaven – Infrastructure Run‑book

> One stop reference for recreating / operating the **low‑cost always‑on‑ish** backend.

---
## 0 – Accounts / Regions
| Resource | Value |
|----------|-------|
| AWS Region | **us‑east‑1** |
| Account ID | **466287783087** |
| Domain Registrar | **Namecheap** |

---
## 1 – Compute layer
| Item | Details |
|------|---------|
| EC2 instance | **g4dn.xlarge Spot** (persistent request **sir‑a5af32…**) 8 vCPU • 16 GiB • T4 GPU |
| EBS | **gp3 30 GiB** @ 3 000 IOPS mounted **/mnt/gp3disk** – `UUID=… /mnt/gp3disk ext4 defaults,nofail 0 2` |
| Docker | Engine enabled @ boot `sudo systemctl enable docker` |
| Compose file | `~/gp3disk/Medimaven/docker‑compose.prod.yml` |
| Containers | `tgi` (port 8080), `rag-api` (port 8000) with `restart: always` |
| EIP | **34.201.xxx.xx** attached to instance SG `launch‑wizard‑9` |

---
## 2 – Networking & Security
### 2.1 Security groups
| SG | Inbound | Notes |
|----|---------|-------|
| **sg‑0a9971b2391cf3106** (Instance) | 22 / 0.0.0.0, 8000 **from ALB SG only** | no 80/443 exposed |
| **sg‑0b3a7484f85757ce8** (ALB) | 80, 443 / 0.0.0.0 | ALB → instance on 8000 |

### 2.2 VPC / Subnets
Public subnets **us‑east‑1a / 1c / 1f** associated with ALB. Instance lives in **subnet‑0000768896c487752 (1f)**.

---
## 3 – Application Load Balancer
| Setting | Value |
|---------|-------|
| Name | **medimaven‑api‑alb** |
| Scheme | **internet‑facing** |
| Listeners | **80 → redirect 443**, **443 → TG (TLS terminated)** |
| ACM cert | `medimaven-ai.com` (+ `api.medimaven-ai.com`) ARN `0039a497‑abb7‑…` |
| Target group | **medimaven-target-group** – instance target, protocol HTTP, port **8000**, health check `GET /health` |

---
## 4 – DNS & TLS
| Record | Type | Target |
|--------|------|--------|
| `medimaven-ai.com` | A (alias) | ALB |
| `api.medimaven-ai.com` | A (alias) | ALB |
| ACM validation | 2× CNAME (AWS added) |
| Registrar NS | pointed to Route 53 NS set **ns‑682 / 1290 / 1593 / 176** |

---
## 5 – Start / Stop automation
| Component | Logic |
|-----------|-------|
| **CloudWatch rule** `stop‑idle‑instance` | Runs every 15 min, triggers Lambda `StopIdleInstance` |
| Lambda `StopIdleInstance` | If **Max CPU ≤ 10 % during last 15 min** → `stop_instances()` |
| Lambda `StartInstance` | Triggered by **API Gateway HTTP => Lambda** or Slack slash‑command; `start_instances()` then polls ALB `/health` until ready |

---
## 6 – Logging & Monitoring
* **ALB access logs** → S3 bucket `medimaven-alb-logs` (lifecycle 30 days).
* **Container logs** gathered by Docker & sent to CloudWatch log‑group `medimaven‑ecs` via `awslogs` driver (compose).
* **Budget alert**: monthly threshold \$150 emailing phade160@…

---
## 7 – Re‑deployment steps
1. `ssh -i ~/.ssh/ec2_key.pem ubuntu@34.201.xxx.xx`
2. `cd /mnt/gp3disk/Medimaven`
3. `git pull && docker compose pull && docker compose up -d --remove-orphans`

---
## 8 – Cost (est.)
| Item | Qty | $/unit | Monthly \$ |
|------|-----|--------|-----------|
| g4dn.xlarge **Spot** | 50 hrs/mo (on‑demand time) | \$0.526 | **\$26** |
| gp3 30 GiB | 30 | \$0.08 | \$2.40 |
| EIP (attached) | 0 (attached) | \$0 | \$0 |
| ALB | 730 hrs | \$0.0225 | \$16 |
| ALB LCU | ~1 | \$0.008 | \$6 |
| Route 53 Hosted zone | 1 | \$0.50 | \$0.50 |
| DNS queries | ~1 M | \$0.40 | \$0.40 |
| ACM cert | N/A | \$0 | \$0 |
| Lambda (+ CW) | Negligible | | **—** |
| **Total (typical month)** | | | **≈ \$51** |

---
### Appendix A – Endpoints
* **API** (base): `https://api.medimaven-ai.com`  
  • `/health` – readiness  
  • `/chat` – body `{ "query": "…" }`
* **Swagger UI**: `https://api.medimaven-ai.com/docs`

---
_Last updated_ 25 Apr 2025



*End of file*
| **Layer** | **What we created** | **AWS service / place** | **Re-create** | **Notes** |
|-----------|--------------------|-------------------------|---------------|-----------|
| Images | phade160/medimaven-rag-api + TGI | Docker Hub + GHCR | `docker pull …` | Already public |
| Compose stack | docker-compose.prod.yml | EC2 | `docker compose up -d` | restart:always |
| Data disk | 30 GiB gp3 `/mnt/gp3disk` | EBS | create → attach → fstab | \$2.40 / mo |
| Compute | g4dn.xlarge **persistent Spot** | EC2 Spot | Launch template → Spot request | IAM role SSM |
| Auto start/stop | Lambda start / stopIdle | Lambda + CW Rule | zip upload, env `INSTANCE_ID` | Stop when 15 min CPU <10 % |
| Load balancer | medimaven-api-alb | ALB | wizard (3 AZ) | listeners 80→TG 8000, 443 cert |
| Target group | medimaven-target-group | ALB TG | HTTP 8000, health `/health` | |
| TLS cert | medimaven-ai.com + api.* | ACM | request DNS-validated | issued |
| DNS zone | medimaven-ai.com | Route 53 | create hosted zone | NS → Namecheap |
| CNAME validation | 2 records | Route 53 | pasted | done |
| API record | api.medimaven-ai.com → ALB | Route 53 A-alias | select ALB | |
| Root redirect | *(todo)* S3 bucket or CF Function | S3 or CF | see run-book | |
| ALB logs | *(todo)* | S3 bucket `medimaven-alb-logs` | enable on ALB | 30 day lifecycle |
| Container logs | *(todo)* | CloudWatch `/ecs/medimaven` | awslogs driver | |
| Budget | *(todo)* \$150 | AWS Budgets | billing console | email alert |

