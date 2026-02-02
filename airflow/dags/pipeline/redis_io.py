import redis, pyarrow as pa, pyarrow.parquet as pq
import pandas as pd

def _client(host="redis", port=6379, db=0):
    return redis.Redis(host=host, port=port, db=db, decode_responses=False)

def put_df(key: str, df: pd.DataFrame, ttl: int | None = 3600, host="redis", port=6379, db=0):
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    buf = sink.getvalue().to_pybytes()
    r = _client(host, port, db)
    r.set(key, buf)
    if ttl: r.expire(key, ttl)
    return True

def get_df(key: str, host="redis", port=6379, db=0) -> pd.DataFrame | None:
    r = _client(host, port, db)
    raw = r.get(key)
    if not raw: return None
    table = pq.read_table(pa.BufferReader(raw))
    return table.to_pandas()
