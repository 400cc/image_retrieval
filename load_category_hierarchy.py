import mysql.connector
from mysql.connector import pooling
from typing import List, Dict
from util.mysql_db_util import create_connection_pool, get_db_connection

# 데이터베이스 설정 파일 경로
config_path = 'util/mysql_config.json'

# 커넥션 풀 생성
connection_pool = create_connection_pool(config_path)

# 데이터베이스에서 데이터 가져오기
def fetch_data_from_db():
    # 커넥션 풀에서 커넥션을 가져오기
    conn = get_db_connection(connection_pool)
    # 재귀 쿼리
    query = """
    WITH RECURSIVE category_hierarchy AS (
        SELECT
            cs.style_id,
            c.category_id AS ancestor_id,
            c.name AS ancestor_name,
            cc.depth,
            cc.descendant_id,
            c2.name AS descendant_name
        FROM
            category_style cs
        JOIN
            category_closure cc ON cs.category_id = cc.descendant_id
        JOIN
            category c ON cc.ancestor_id = c.category_id
        JOIN
            category c2 ON cc.descendant_id = c2.category_id
    )
    SELECT
        style_id,
        ancestor_id,
        ancestor_name,
        depth,
        descendant_id,
        descendant_name
    FROM
        category_hierarchy
    ORDER BY
        style_id, depth;
    """
    
    # 커서 생성 및 쿼리 실행
    with conn.cursor(dictionary=True) as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()  # 모든 행을 가져옴

    # 커넥션 닫기
    conn.close()
    return rows

# 메모리에 데이터를 저장할 변수
category_hierarchy_cache = {}

# 데이터 로드 및 처리 함수
def load_category_hierarchy():
    rows = fetch_data_from_db()  # 데이터베이스에서 데이터 가져오기
    process_hierarchy_rows(rows)
    return category_hierarchy_cache


# 쿼리 결과를 처리하는 함수
def process_hierarchy_rows(rows):
    temp_cache = {}

    for row in rows:
        style_id = row['style_id']
        descendant_id = row['descendant_id']

        # style_id가 임시 캐시에 없으면 초기화
        if style_id not in temp_cache:
            temp_cache[style_id] = {}

        # descendant_id가 임시 캐시에 없으면 초기화
        if descendant_id not in temp_cache[style_id]:
            temp_cache[style_id][descendant_id] = []

        # 계층 구조 정보를 임시 캐시에 추가
        temp_cache[style_id][descendant_id].append({
            "ancestor_name": row["ancestor_name"],
            "depth": row["depth"]
        })

    # 최종 캐시 빌드
    build_category_hierarchy_cache(temp_cache)

# 최종 캐시를 빌드하는 함수
def build_category_hierarchy_cache(temp_cache):
    global category_hierarchy_cache

    for style_id, descendants in temp_cache.items():
        category_hierarchy_cache[style_id] = []
        for descendant_id, hierarchy in descendants.items():
            # depth를 기준으로 정렬
            sorted_hierarchy = sorted(hierarchy, key=lambda x: x['depth'])
            # 정렬된 계층 구조를 캐시에 추가
            category_hierarchy_cache[style_id].append([item['ancestor_name'] for item in sorted_hierarchy])

# 스타일 ID로 계층 구조를 가져오는 함수
def get_category_hierarchy(style_id):
    if style_id not in category_hierarchy_cache:
        raise ValueError(f"Style ID {style_id} not found")  # 예외 처리
    return category_hierarchy_cache[style_id]

