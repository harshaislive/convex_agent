import argparse
import json

from tools import browse_beforest_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Beforest page fetching.")
    parser.add_argument("url", help="Beforest URL to fetch")
    parser.add_argument("query", nargs="?", default="", help="Optional focus query")
    args = parser.parse_args()

    result = browse_beforest_page.invoke({"url": args.url, "query": args.query})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
