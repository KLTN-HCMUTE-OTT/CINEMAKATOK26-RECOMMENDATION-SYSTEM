from sqlalchemy import create_engine
from config.setting import POSTGRES_URL_AUDIT
from config.setting import POSTGRES_URL_CONTENT

engine_audit = create_engine(POSTGRES_URL_AUDIT)
engine_content = create_engine(POSTGRES_URL_CONTENT)
