"""
Configuration module - Load from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent

# LM Studio
LM_STUDIO_URL = 'http://localhost:1234/v1/chat/completions'
LM_STUDIO_MODEL = 'vistral-7b-chat@q8'  # Model đang load trong LM Studio

# Server
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 3737
SERVER_DEBUG = False

# RAG Parameters (for LM Studio)
RAG_MAX_TOKENS = int(os.getenv('RAG_MAX_TOKENS', '2000'))
RAG_TEMPERATURE = float(os.getenv('RAG_TEMPERATURE', '0.7'))

print(f"[CONFIG] LM Studio: {LM_STUDIO_URL}")
print(f"[CONFIG] Model: {LM_STUDIO_MODEL}")
print(f"[CONFIG] Server: {SERVER_HOST}:{SERVER_PORT}")

