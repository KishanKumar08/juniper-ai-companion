import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
COMPANION_MODEL = os.getenv("COMPANION_MODEL")          
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL")           
JUDGE_MODEL = os.getenv("JUDGE_MODEL")    
ORACLE_MODEL = os.getenv("ORACLE_MODEL")
DB_PATH = os.getenv("COMPANION_DB", "companion.db")


EPISODIC_TOP_K = 5          # how many past events we pull per turn
OPINION_TOP_K = 4           # how many of the companion's own past opinions we resurface
RECENT_TURNS = 8            # raw conversation turns kept verbatim in context
STATE_FACT_TTL = 25         # a "state" fact (mood, current task) expires after this many turns
RECENCY_HALF_LIFE = 30      # episodic salience halves every N turns


def require_credentials():
    if not BEDROCK_API_KEY:
        raise SystemExit("No Bedrock credentials found - can't reach Claude")
