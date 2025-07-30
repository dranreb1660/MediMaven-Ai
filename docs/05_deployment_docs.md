# 🚀 MediMaven Deployment Documentation

![CI](https://img.shields.io/badge/Built_with-Docker-blue) ![AWS](https://img.shields.io/badge/Cloud-AWS-%23FF9900) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

This document provides a detailed overview of the deployment process for the MediMaven project, covering both local and cloud-based deployments. It is intended for DevOps engineers, machine learning engineers, and developers responsible for deploying and maintaining the application.

## Table of Contents

1.  [Deployment Overview](#deployment-overview)
2.  [Local Deployment](#local-deployment)
3.  [AWS Deployment](#aws-deployment)
4.  [CI/CD Pipeline](#cicd-pipeline)
5.  [Configuration](#configuration)
6.  [Troubleshooting](#troubleshooting)

## 🎯 Deployment Overview

The MediMaven application is designed for both local development and production deployment on AWS. The deployment strategy is centered around Docker, with separate Docker Compose files for local and production environments. The production deployment is optimized for cost-efficiency and scalability, using a combination of AWS services to provide a robust and reliable service.

## 🐳 Local Deployment

For local development and testing, you can use the `docker-compose.yml` file to run the application on your local machine.

**Steps:**

1.  **Install Docker and Docker Compose**.
2.  **Create a `.env` file** in the root of the project and populate it with the necessary environment variables (see the [Configuration](#configuration) section).
3.  **Build and run the Docker containers**:

    ```bash
    docker-compose up --build
    ```

This will build the Docker image, download the necessary models, and start the FastAPI application on `http://localhost:8000`.

## ☁️ AWS Deployment

The production deployment is designed to be cost-effective and scalable, using a combination of AWS services:

| Service | Component | Purpose |
| :--- | :--- | :--- |
| **EC2** | `g4dn.xlarge` Spot Instance | Cost-effective GPU-accelerated compute for the backend. |
| **S3** | `medimaven-web` Bucket | Hosts the static frontend assets. |
| **CloudFront**| `Distribution_ID` | Caches the frontend and provides a global CDN for low-latency access. |
| **ALB**| `medimaven-api-alb` | Distributes traffic to the EC2 instance and provides SSL termination. |
| **Route 53**| `medimaven-ai.com` | Manages the DNS records for the application. |
| **Lambda**| `StopIdleInstance` | Automatically stops the EC2 instance to reduce costs. |

For a detailed breakdown of the AWS infrastructure, please refer to the [Infrastructure Runbook](infra-runbook.md).

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment. The CI/CD pipeline is defined in the `.github/workflows/deploy.yml` file and consists of the following steps:

| Step | Description |
| :--- | :--- |
| **Checkout** | The code is checked out from the repository. |
| **Install Dependencies** | The frontend and backend dependencies are installed. |
| **Run Tests** | The frontend and backend tests are run to ensure the code is working as expected. |
| **Build** | The frontend is built for production. |
| **Deploy** | The frontend is deployed to S3, and the Docker image is pushed to Docker Hub. |

## ⚙️ Configuration

The application is configured using environment variables. The following table lists the key environment variables and their purpose:

| Variable | Description |
| :--- | :--- |
| `HF_TOKEN` | Your Hugging Face Hub token, used to download the models. |
| `WANDB_API_KEY` | Your Weights & Biases API key, used for experiment tracking. |
| `DATABASE_URL` | The connection string for the database. |
| `AUTH0_DOMAIN` | Your Auth0 domain. |
| `AUTH0_AUDIENCE` | Your Auth0 audience. |
| `ALLOWED_ORIGINS` | A comma-separated list of allowed origins for CORS. |

## ❓ Troubleshooting

| Symptom | Likely Cause | Remedy |
| :--- | :--- | :--- |
| White screen, CSS 404 | `/index.html` cached too long | Re-upload with short cache + invalidate |
| JS 403 `AccessDenied` | Bucket policy missing CF OAC | Update policy |
| SPA routes 404 XML | `spaRewrite` missing | Attach CF Function or add behaviour |
| Auth redirect 403 | Wrong `AUTH0_DOMAIN` / `AUDIENCE` | Fix env vars |
| CORS error | FastAPI CORS not allowing prod origin | Update backend CORS list |

This concludes the overview of the deployment process for the MediMaven project. For more detailed information, please refer to the individual deployment and infrastructure files.
