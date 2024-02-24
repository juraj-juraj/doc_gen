import argparse
import logging
import sys
import time
from threading import Thread

import backoff
import openai
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential


def setup_logging(log_level: int = logging.WARNING):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )


class completion_worker(Thread):
    def __init__(self, series: pd.Series, api_key: str, worker_id: int):
        logging.debug(f"Worker {worker_id}: Initializing")
        Thread.__init__(self)
        self.series = series
        self.client = openai.OpenAI(api_key=api_key)
        self.result = None
        self.worker_id = worker_id

    # @retry(wait=wait_random_exponential(min=20, max=70), stop=stop_after_attempt(50))
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def _send_request(self, code: str):
        logging.info(f"Worker {self.worker_id}: Sending request")
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant to create google-style docstrings for python functions.",
                },
                {
                    "role": "user",
                    "content": f'Create google-style dosctring for function: \n "{code}" \n Return only docstring without quotes.',
                },
            ],
            n=1,
        )
        return response.choices[0].message.content.strip('"')

    def run(self):
        docstrings = [self._send_request(code) for code in self.series]
        logging.info(f"Worker {self.worker_id}: Finished")
        self.result = pd.DataFrame({"function": self.series, "docstring": docstrings})


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--range", nargs=2, type=int, default=[0, sys.maxsize], help="Range of rows to process")
    parser.add_argument("-l", "--log_level", type=logging.getLevelName, default="WARNING", help="Log level")
    parser.add_argument("-w", "--workers", type=int, default=5, help="Number of workers")
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, help="Output file")

    args = parser.parse_args()
    setup_logging(args.log_level)

    raw_data = pd.read_pickle(args.input)
    df = pd.DataFrame(raw_data)
    logging.info(f"Loaded {len(df)} rows")

    df = df.iloc[args.range[0] : args.range[1]]["function"].copy()
    logging.info(f"Processing {len(df)} rows")

    chunk_size = len(df) // args.workers
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    workers = [completion_worker(chunk, args.api_key, id) for chunk, id in zip(chunks, range(args.workers))]

    [worker.start() for worker in workers]
    [worker.join() for worker in workers]

    result = pd.concat([worker.result for worker in workers])
    result.reset_index(inplace=True, drop=True)

    result[["function", "docstring"]].to_pickle(args.output)
    logging.info(f"Finished in {time.time() - start_time} seconds")
    logging.info(f"Processes {len(result)} rows")
    logging.info(f"Processing rate: {len(result) / (time.time() - start_time)} functions per second")


if __name__ == "__main__":
    main()
