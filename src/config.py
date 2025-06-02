import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
 
# Lichess API configuration
LICHESS_API_TOKEN = os.getenv('LICHESS_API_TOKEN')
if not LICHESS_API_TOKEN:
    raise ValueError("LICHESS_API_TOKEN environment variable is not set. Please create a .env file with your token.") 