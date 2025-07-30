from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2SpotInstance
from diagrams.aws.database import RDS, ElastiCache
from diagrams.aws.network import ELB, CloudFront
from diagrams.aws.storage import S3
from diagrams.onprem.container import Docker
from diagrams.onprem.vcs import Github
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.analytics import Spark
from diagrams.programming.framework import FastAPI, React
from diagrams.programming.language import Python
from diagrams.generic.device import Mobile
from diagrams.aws.management import Cloudwatch
from diagrams.generic.blank import Blank
from diagrams.generic.database import SQL

# Set graph attributes for better visibility
graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "rankdir": "TB",
    "nodesep": "0.8",
    "ranksep": "1.2",
    "pad": "0.5",
    "dpi": "150"
}

node_attr = {
    "fontsize": "14",
    "height": "1.2",
    "width": "1.5"
}

edge_attr = {
    "fontsize": "12"
}

with Diagram("MediMaven Architecture", show=False, filename="docs/medimaven_architecture_final", 
             outformat="png", graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr):
    
    # User
    user = Mobile("End User")
    
    # Frontend & CDN
    with Cluster("Frontend"):
        react_app = React("React App")
        cdn = CloudFront("CloudFront")
    
    # AWS Infrastructure
    with Cluster("AWS Infrastructure"):
        alb = ELB("ALB")
        
        with Cluster("Auto Scaling"):
            backend = EC2SpotInstance("Spot Instance\n(Docker)")
            fastapi = FastAPI("FastAPI")
            backend >> fastapi
    
    # RAG Pipeline
    with Cluster("RAG Pipeline"):
        retriever = Python("Hybrid Retriever\nBM25 + Dense")
        reranker = Python("2-Stage Reranker\nLambdaMART + BGE")
        generator = Python("Llama-3 8B\nQLoRA/AWQ")
    
    # Databases
    with Cluster("Data Layer"):
        postgres = RDS("PostgreSQL")
        redis = ElastiCache("Redis Cache")
        qdrant = SQL("Qdrant Vector DB")
    
    # Model Training
    with Cluster("Training Pipeline"):
        scrapy = Python("Scrapy\nWeb Scraper")
        airflow = Airflow("Airflow ETL")
        
        with Cluster("Model Training"):
            train_jobs = Python("Training Jobs\n• QLoRA\n• AWQ\n• LambdaMART\n• BGE")
            wandb = Blank("WandB\nExperiment\nTracking")
    
    # Model Storage
    hf_hub = Blank("🤗 HuggingFace\nModels & Datasets")
    
    # CI/CD
    with Cluster("CI/CD"):
        github = Github("GitHub")
        actions = Github("Actions")
        docker = Docker("Docker Hub")
    
    # Monitoring
    monitoring = Cloudwatch("CloudWatch")
    
    # Main flow connections
    user >> react_app >> cdn >> alb >> backend
    
    # Backend to RAG
    fastapi >> retriever >> reranker >> generator
    
    # Data connections
    retriever >> qdrant
    fastapi >> [postgres, redis]
    
    # Model loading
    [retriever, reranker, generator] << Edge(label="load models") << hf_hub
    
    # Training pipeline
    scrapy >> airflow >> train_jobs
    train_jobs >> wandb
    train_jobs >> Edge(label="publish") >> hf_hub
    
    # CI/CD
    github >> actions >> docker >> backend
    
    # Monitoring
    backend >> monitoring
