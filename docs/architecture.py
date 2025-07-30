from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, EC2SpotInstance, AutoScaling, ECS
from diagrams.aws.database import RDS, ElastiCache
from diagrams.aws.network import ELB, Route53, CloudFront
from diagrams.aws.storage import S3
from diagrams.onprem.container import Docker
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.onprem.vcs import Github
from diagrams.programming.framework import FastAPI, React
from diagrams.programming.language import Python
from diagrams.generic.device import Mobile
from diagrams.aws.management import Cloudwatch
from diagrams.saas.analytics import Tableau

with Diagram("MediMaven Detailed Architecture", show=False, filename="docs/medimaven_architecture", outformat="svg"):
    user = Mobile("End User")

    with Cluster("AWS Cloud"):
        dns = Route53("DNS")
        cdn = CloudFront("CDN")

        with Cluster("VPC"):
            with Cluster("Public Subnet"):
                lb = ELB("ALB")

            with Cluster("Private Subnet"):
                with Cluster("Auto Scaling Group"):
                    spot_instances = [
                        EC2SpotInstance("Spot Instance 1"),
                        EC2SpotInstance("Spot Instance 2"),
                    ]
                    asg = AutoScaling("ASG")
                    asg >> Edge(label="scales based on load") >> spot_instances[0]


                with Cluster("Backend Services (Dockerized)"):
                    backend_app = FastAPI("FastAPI App")
                    spot_instances[0] >> backend_app

                    with Cluster("RAG Pipeline"):
                        retriever = Python("Hybrid Retriever")
                        reranker = Python("2-Stage Reranker")
                        generator = Python("Llama-3 8B")

                        backend_app >> retriever >> reranker >> generator

                    with Cluster("Models"):
                        qlora_model = S3("Fine-tuned Llama-3 (QLoRA)")
                        awq_model = S3("Quantized Llama-3 (AWQ)")
                        lgbm_model = S3("LambdaMART Model")
                        colbert_model = S3("ColBERT Model")

                        generator >> qlora_model
                        generator >> awq_model
                        reranker >> lgbm_model
                        retriever >> colbert_model

                with Cluster("Databases"):
                    postgres_db = RDS("Postgres DB (Conversations)")
                    redis_cache = ElastiCache("Redis (Cache)")
                    qdrant_db = EC2("Qdrant Vector DB")

                    backend_app >> postgres_db
                    backend_app >> redis_cache
                    retriever >> qdrant_db

                with Cluster("Monitoring"):
                    cloudwatch = Cloudwatch("CloudWatch")
                    spot_instances[0] >> cloudwatch


    with Cluster("CI/CD & Development"):
        github = Github("GitHub Repo")
        github_actions = Github("GitHub Actions")
        docker = Docker("Docker Hub")
# Readded WandB with correct import
        wandb = Wandb("WandB (Experiments)")

        github >> github_actions >> docker
        github_actions >> lb

    with Cluster("Model Training Pipeline (Offline)"):
        data_sources = S3("Medical Data Sources")
        with Cluster("ETL Pipeline (Spark)"):
            etl_job = Python("Spark ETL Job")
            data_sources >> etl_job
        processed_data = S3("Processed Parquet Files")
        etl_job >> processed_data
        
        with Cluster("Training Jobs"):
            qlora_training = Python("QLoRA Fine-tuning")
            awq_quantization = Python("AWQ Quantization")
            ltr_training = Python("LambdaMART Training")
            colbert_training = Python("ColBERT Training")
            
            processed_data >> qlora_training
            processed_data >> ltr_training
            processed_data >> colbert_training
            qlora_training >> awq_quantization
            
            # WandB connections for experiment tracking
            wandb >> Edge(label="logs experiments") >> qlora_training
            wandb >> Edge(label="logs experiments") >> ltr_training
            wandb >> Edge(label="logs experiments") >> colbert_training

    # Frontend
    with Cluster("User Interface"):
        react_app = React("React Web App")

    # Edges
    user >> react_app >> cdn >> lb >> spot_instances
    cdn >> S3("Frontend Assets")
    github_actions >> Edge(label="deploy") >> backend_app
