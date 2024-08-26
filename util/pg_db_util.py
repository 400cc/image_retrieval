import psycopg2
from sshtunnel import SSHTunnelForwarder

def get_pg_connection():
    ssh_host = '54.180.146.236'
    ssh_port = 22
    ssh_user = 'ubuntu'
    ssh_private_key = "/app/aws.ac.kwu.pem"

    pg_host = '127.0.0.1'
    pg_port = 5432
    pg_user = 'airflow'
    pg_password = 'airflow'
    pg_db = 'image_vector'

    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_private_key=ssh_private_key,
        remote_bind_address=(pg_host, pg_port),
        local_bind_address=('localhost', 5432)
    )
    tunnel.start()
    
    conn_pg = psycopg2.connect(
        host='localhost',
        port=tunnel.local_bind_port,
        user=pg_user,
        password=pg_password,
        dbname=pg_db
    )
    return conn_pg, tunnel
