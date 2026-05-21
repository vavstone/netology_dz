import time
from uuid import uuid4
from loguru import logger

from config import Settings
from models import AssistantResponse, SessionStats
from infrastructure.cache import RedisCache
