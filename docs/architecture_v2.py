from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, EC2SpotInstance, AutoScaling
from diagrams.aws.database import RDS, ElastiCache
from diagrams.aws.network import ELB, CloudFront
from diagrams.aws.storage import S3
from diagrams.onprem.container import Docker
from diagrams.onprem.vcs import Github
from diagrams.programming.framework import FastAPI, React
from diagrams.programming.language import Python
from diagrams.generic.device import Mobile
from diagrams.aws.management import Cloudwatch
from diagrams.onprem.analytics import Spark
from diagrams.generic.database import SQL
from diagrams.generic.storage import Storage

# Set graph attributes for better layout
graph_attr = {
    "fontsize": "45",
    "bgcolor": "white",
    "rankdir": "TB",
    "nodesep": "1",
    "ranksep": "1.5",
    "pad": "0.5"
}

with Diagram("MediMaven Architecture v1.1", show=False, filename="docs/medimaven_architecture_v2", 
             outformat="png", graph_attr=graph_attr, direction="TB"):
    
    # User Interface
    user = Mobile("End User")
    
    # Frontend
    frontend = React("React Web App")
    
    # AWS Infrastructure
    cdn = CloudFront("CloudFront CDN")
    alb = ELB("Application Load Balancer")
    
    # Auto Scaling Group with Spot Instances
    with Cluster("Auto Scaling Group"):
        ec2_instances = [
            EC2SpotInstance("EC2 Spot Instance"),
        ]
    
    # Backend Services
    with Cluster("Backend Services (Docker)"):
        fastapi = FastAPI("FastAPI Backend")
        
        with Cluster("RAG Pipeline"):
            retriever = Python("Hybrid Retriever\n(BM25 + Dense)")
            reranker = Python("2-Stage Reranker\n(LambdaMART + BGE)")
            generator = Python("Llama-3 8B\n(QLoRA/AWQ)")
    
    # Data Storage
    with Cluster("Data Storage"):
        postgres = RDS("PostgreSQL\n(Conversations)")
        redis = ElastiCache("Redis\n(Cache)")
        qdrant = SQL("Qdrant\n(Vector DB)")
        s3_models = S3("S3\n(Models)")
    
    # Model Training Pipeline
    with Cluster("Model Training Pipeline"):
        data_sources = S3("Medical Data\nSources")
        spark_etl = Spark("Spark ETL\nPipeline")
        processed_data = Storage("Processed\nParquet Files")
        
        with Cluster("Training Jobs"):
            qlora = Python("QLoRA\nFine-tuning")
            awq = Python("AWQ\nQuantization")
            lambdamart = Python("LambdaMART\nTraining")
            bge_cross = Python("BGE Cross-Encoder\nTraining")
    
    # CI/CD
    with Cluster("CI/CD"):
        github = Github("GitHub")
        github_actions = Github("GitHub Actions")
        docker_hub = Docker("Docker Hub")
    
    # Monitoring
    cloudwatch = Cloudwatch("CloudWatch\nMonitoring")
    
    # WandB for experiment tracking
    wandb = Storage("Weights & Biases\n(Experiment Tracking)")
    
    # Define connections
    user >> frontend >> cdn >> alb >> ec2_instances[0]
    ec2_instances[0] >> fastapi
    
    # Backend to RAG pipeline
    fastapi >> retriever
    retriever >> reranker
    reranker >> generator
    
    # Data connections
    retriever >> qdrant
    fastapi >> postgres
    fastapi >> redis
    generator >> s3_models
    reranker >> s3_models
    retriever >> s3_models
    
    # Training pipeline
    data_sources >> spark_etl >> processed_data
    processed_data >> [qlora, lambdamart, bge_cross]
    qlora >> awq
    
    # Model outputs
    qlora >> s3_models
    awq >> s3_models
    lambdamart >> s3_models
    bge_cross >> s3_models
    
    # WandB tracking
    [qlora, awq, lambdamart, bge_cross] >> wandb
    
    # CI/CD flow
    github >> github_actions >> docker_hub
    github_actions >> alb
    
    # Monitoring
    ec2_instances[0] >> cloudwatch
    
    # Frontend assets
    cdn >> S3("Frontend Assets")
