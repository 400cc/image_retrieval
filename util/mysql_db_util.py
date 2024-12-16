import json
import mysql.connector
from mysql.connector import pooling
from typing import Dict

# JSON 파일에서 데이터베이스 설정 정보를 로드하는 함수
def load_db_config(config_path: str) -> Dict:
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# 커넥션 풀 생성 함수
def create_connection_pool(config_path: str, pool_name: str = "mypool", pool_size: int = 10):
    db_config = load_db_config(config_path)
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name=pool_name,
        pool_size=pool_size,  # 최대 연결 풀 크기
        **db_config
    )
    return connection_pool

# 커넥션 풀에서 커넥션을 가져오는 함수
def get_db_connection(pool):
    return pool.get_connection()