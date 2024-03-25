import json
from concurrent.futures import ThreadPoolExecutor

import backoff
import httpx
import pandas as pd


class AnnotateBatchException(Exception): ...


class TooManyRequestsException(Exception): ...


@backoff.on_exception(backoff.expo, TooManyRequestsException)
def annotate_batch(batch: pd.DataFrame, host: str) -> pd.DataFrame:
    predictions = []
    with httpx.Client() as client:
        for _, row in batch.iterrows():
            r = client.post(
                f"{host}/generate_docstring/", json={"code": row["functions"]}, timeout=httpx.Timeout(timeout=300)
            )

            if r.status_code == 429:
                raise TooManyRequestsException()
            if r.status_code != 200:
                raise AnnotateBatchException(f"Server returned error: {r.text}")
            predictions.append(json.loads(r.text)["result"])
    batch.loc[:, "predictions"] = predictions
    return batch


def annotate_corpus(host: str, data: pd.DataFrame, n_workers: int = 1):
    chunk_size = len(data) // n_workers
    batches = [data.iloc[i : i + chunk_size].copy() for i in range(0, len(data), chunk_size)]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(annotate_batch, batch, host) for batch in batches]
        results = [future.result() for future in futures]
    preds_df = pd.concat(results)
    return preds_df.sort_index()
