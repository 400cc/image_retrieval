import json
from sshtunnel import SSHTunnelForwarder
import psycopg2

def get_pg_connection():
    # Load configuration from pg_config.json
    with open('util/pg_config.json', 'r') as config_file:
        config = json.load(config_file)
    
    ssh_config = config['ssh']
    pg_config = config['postgres']

    tunnel = SSHTunnelForwarder(
        (ssh_config['host'], ssh_config['port']),
        ssh_username=ssh_config['user'],
        ssh_private_key=ssh_config['private_key'],
        remote_bind_address=(pg_config['host'], pg_config['port']),
        local_bind_address=('localhost', 5432)
    )
    tunnel.start()
    
    conn_pg = psycopg2.connect(
        host='localhost',
        port=tunnel.local_bind_port,
        user=pg_config['user'],
        password=pg_config['password'],
        dbname=pg_config['db']
    )
    return conn_pg, tunnel
