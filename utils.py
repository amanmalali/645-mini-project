import pandas as pd
from configparser import ConfigParser
import psycopg2
from sklearn.preprocessing import MinMaxScaler


def load_config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return config

def normalize(df):
    scaler = MinMaxScaler((0,100))
    columns_scale=['rowc','colc', 'ra', 'field', 'fieldid', 'dec']
    df[columns_scale] = scaler.fit_transform(df[columns_scale])
    return df


def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def select_columns(path):
    df=pd.read_csv(path)
    columns=['objid','rowc','colc', 'ra', 'field', 'fieldid', 'dec']
    df=df[columns]
    df=normalize(df)
    df.to_csv("./data/small_sdss.csv",index=False)


def init_tables():
    """ Create tables in the PostgreSQL database"""
    commands = (
        """
        drop table IF EXISTS skyphoto cascade;
        """,
        """
        CREATE TABLE skyphoto (
            objid BIGINT PRIMARY KEY,
            rowc REAL,
            colc REAL, 
            ra FLOAT(8), 
            field FLOAT(8), 
            fieldid FLOAT(8),
            dec FLOAT(8)

        )
        """,
        )
    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                # execute the CREATE TABLE statement
                for command in commands:
                    cur.execute(command)

                    cur = conn.cursor()
                with open('./data/small_sdss.csv', 'r') as f:
                    next(f) # Skip the header row.
                    cur.copy_from(f, 'skyphoto', sep=',')
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def run_query(query_string):
    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                # execute the CREATE TABLE statement
                cur.execute(query_string)
                results = cur.fetchall()
        return results 
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def run_query_to_df(query_string):
    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            df = pd.read_sql_query(query_string, conn)
        return df
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)