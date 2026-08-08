"""One-off maintenance: set the public visit counter to a specific value.

Why this exists: while the map was served from a laptop through a tunnel, every
visit incremented that machine's counter and none of them reached the deployed
store. Moving back to cloud hosting exposed the deployed store's lower value.
This reconciles the two. It counts visits that genuinely happened; it is not a
way to invent traffic.

The counting system is unaffected afterwards. Redis INCR on an existing key
resumes from whatever value is there, so the next visit after setting 1095 is
1096 and the normal path takes over again.

Run it locally, never on the server, and never commit the credentials:

    export UPSTASH_REDIS_REST_URL=https://<your-db>.upstash.io
    export UPSTASH_REDIS_REST_TOKEN=<your-token>
    python backend/set_visit_count.py 1095

Both values are in the Render dashboard under Environment. Equivalent one-liner
in the Upstash console, if you prefer not to handle the token locally:

    SET shared_skies_visits 1095
"""
import os
import sys

import httpx

KEY = "shared_skies_visits"


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].lstrip("-").isdigit():
        sys.exit("usage: python backend/set_visit_count.py <integer>")
    target = int(sys.argv[1])
    if target < 0:
        sys.exit("count must be non-negative")

    url = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
    if not url or not token:
        sys.exit(
            "UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set.\n"
            "Copy them from the Render dashboard (Environment), export them in\n"
            "this shell, and re-run. Do not commit them."
        )

    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=10.0) as client:
        before = client.get(f"{url}/get/{KEY}", headers=headers).json().get("result")
        print(f"current: {before}")
        if before is not None and int(before) > target:
            # Guard against silently erasing real traffic with a typo.
            ok = input(f"{before} is HIGHER than {target}; lower it? [y/N] ").strip().lower()
            if ok != "y":
                sys.exit("aborted")
        client.post(f"{url}/set/{KEY}/{target}", headers=headers).raise_for_status()
        after = client.get(f"{url}/get/{KEY}", headers=headers).json().get("result")

    print(f"set to : {after}")
    print(
        "\nThe public endpoint caches for 10 minutes "
        "(PUBLIC_VISITS_TTL_MIN), so /api/metrics may show the old value "
        "briefly before it refreshes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
