import argparse
import time

import httpx


def send_request(client: httpx.Client, host: str):
    start_time = time.time()
    try:
        client.get(host)
    except httpx.HTTPError as e:
        print(e)
    finally:
        print(f"Ping: {(time.time() - start_time) * 1000}")


def main():
    parser = argparse.ArgumentParser(description="Client ot measure ping at set location repeatedly.")
    parser.add_argument("-H", "--host", type=str, help="Endpoint to send requests")
    parser.add_argument("--cooldown", type=int, default=0, help="Milliseconds between each request")
    parser.add_argument("--num", type=int, default=None, help="How many times to ping")

    args = parser.parse_args()
    with httpx.Client() as client:
        num_request = args.num or 9999
        for _ in range(num_request):
            send_request(client, args.host)
            time.sleep(args.cooldown / 1000)


if __name__ == "__main__":
    main()
