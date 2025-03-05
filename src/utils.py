import os
import subprocess
from pymongo import MongoClient


def ensure_mongodb_running():
    """Checks if MongoDB is running, and starts it if not."""
    try:
        # Try connecting to MongoDB
        subprocess.run(["mongosh", "--eval", "db.runCommand({ ping: 1 })"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ MongoDB is already running.")
    except subprocess.CalledProcessError:
        print("⚠️ MongoDB is NOT running. Attempting to start it...")
        os.system("brew services start mongodb-community")
        print("✅ MongoDB is running now!!!")


def get_mongo_connection(mongo_uri:str = "mongodb://localhost:27017/", db_name:str = "medimaven_db"):
    client = MongoClient(mongo_uri)
    db = client[db_name]

    return db