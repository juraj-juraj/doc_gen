import argparse
from threading import Thread
from typing import Callable

import openai
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from torch import dropout


class completion_worker(Thread):
    def __init__(self, series: pd.Series, api_key: str):
        Thread.__init__(self)
        self.series = series
        self.client = openai.OpenAI(api_key=api_key)
        self.result = None

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _send_request(self, code: str):
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
        self.result = pd.DataFrame({"function": self.series, "docstring": docstrings})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api_key", type=str, help="OpenAI API key")
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, help="Output file")

    args = parser.parse_args()

    raw_data = pd.read_pickle(args.input)
    df = pd.DataFrame(raw_data)
    df = df.iloc[0:16]["function"].copy()

    num_workers = 4
    chunks = [df.iloc[i::num_workers] for i in range(num_workers)]
    workers = [completion_worker(chunk, args.api_key) for chunk in chunks]

    [worker.start() for worker in workers]
    [worker.join() for worker in workers]

    result = pd.concat([worker.result for worker in workers])
    result.reset_index(inplace=True, drop=True)

    print(result)
    result[["function", "docstring"]].to_pickle(args.output)


if __name__ == "__main__":
    main()
