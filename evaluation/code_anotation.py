import json
from concurrent.futures import ThreadPoolExecutor
import backoff
import httpx
import pandas as pd
from model_loader import load_model


class AnnotateBatchException(Exception):
    ...


class TooManyRequestsException(Exception):
    ...


@backoff.on_exception(backoff.expo, TooManyRequestsException)
def annotate_batch(batch: pd.DataFrame, host: str) -> pd.DataFrame:
    """
    Annotates a batch of data using a specified host.

    Args:
        batch (pd.DataFrame): The input batch of data to be annotated.
        host (str): The host to which the data will be annotated.

    Returns:
        pd.DataFrame: The annotated batch of data with predictions.
    """
    predictions = []
    with httpx.Client() as client:
        for _, row in batch.iterrows():
            r = client.post(
                f"{host}/generate_docstring/",
                json={"code": row["functions"]},
                timeout=httpx.Timeout(timeout=300),
            )
            if r.status_code == 429:
                raise TooManyRequestsException()
            if r.status_code != 200:
                raise AnnotateBatchException(f"Server returned error: {r.text}")
            predictions.append(json.loads(r.text)["result"])
    batch.loc[:, "predictions"] = predictions
    return batch


def annotate_corpus_http(
    host: str, data: pd.DataFrame, n_workers: int = 1
) -> pd.DataFrame:
    """
    Annotate the given data using HTTP annotations on a specified host.

    Args:
        host (str): The host to annotate the data on.
        data (pd.DataFrame): The input data to be annotated.
        n_workers (int, optional): The number of workers to use for annotating the data. Defaults to 1.

    Returns:
        pd.DataFrame: The annotated data with the sorted index.
    """
    chunk_size = len(data) // n_workers
    batches = [
        data.iloc[i : i + chunk_size].copy() for i in range(0, len(data), chunk_size)
    ]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(annotate_batch, batch, host) for batch in batches]
        results = [future.result() for future in futures]
    preds_df = pd.concat(results)
    return preds_df.sort_index()


def annotate_corpus_model(model: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates the input data by generating predictions using the specified model.

    Args:
        model (str): The name of the model to be used for generating predictions.
        data (pd.DataFrame): The input data to be annotated.

    Returns:
        pd.DataFrame: The annotated data with predictions generated using the specified model.
    """
    model_cls = load_model(model)
    model_instance = model_cls(device="cuda")
    data.loc[:, "predictions"] = data["functions"].apply(
        lambda x: model_instance.generate(x)
    )
    return data
