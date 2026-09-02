import argparse
import json
import os
import time
from collections import defaultdict

from companion import config
from companion.chat import respond, seed_if_new
from companion.memory import store as store_mod
from . import scenarios, judge, oracle


def run_scenario(scenario, use_oracle=True, verbose=False):
    db_path = f"eval_{scenario['name']}_{int(time.time())}.db"
    store = store_mod.MemoryStore(db_path)
    seed_if_new(store)

    probe_results = []
    try:
        for kind, payload in scenario["steps"]:
            if kind == "say":
                respond(store, payload)
            elif kind == "probe":
                reply, _ = respond(store, payload["ask"])
                oracle_ans = oracle.ideal_answer(store, payload["ask"]) if use_oracle else None
                verdict = judge.judge(payload, reply, oracle_answer=oracle_ans)
                rec = {
                    "scenario": scenario["name"],
                    "category": payload["category"],
                    "ask": payload["ask"],
                    "reply": reply,
                    "oracle": oracle_ans,
                    "passed": verdict.get("passed", False),
                    "scores": verdict.get("scores", {}),
                    "rationale": verdict.get("rationale", ""),
                }
                probe_results.append(rec)
                if verbose:
                    mark = "PASS" if rec["passed"] else "FAIL"
                    print(f"[{mark}] {payload['category']}: {rec['rationale']}")
    finally:
        store.close()
        try:
            os.remove(db_path)
        except OSError:
            pass
    return probe_results


def main():
    # take argument in comments so we can run limited eval alos
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-oracle", action="store_true", help="skip the oracle baseline (faster/cheaper)")
    ap.add_argument("--only", help="comma-separated scenario names to run (subset)")
    ap.add_argument("--limit", type=int, help="run only the first N scenarios")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    config.require_credentials()

    scs = scenarios.all_scenarios()
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        scs = [s for s in scs if s["name"] in wanted]
    if args.limit:
        scs = scs[:args.limit]

    all_results = []
    for sc in scs:
        print(f"\n▶ {sc['name']} ({sum(1 for k,_ in sc['steps'] if k=='probe')} probe(s), "
              f"{len(sc['steps'])} turns)")
        all_results.extend(run_scenario(sc, use_oracle=not args.no_oracle, verbose=not args.quiet))

    _report(all_results)


def _report(results):
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][1] += 1
        if r["passed"]:
            by_cat[r["category"]][0] += 1

    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'category':<14}{'pass':>6}{'total':>7}{'rate':>8}")
    print("  " + "-" * 33)
    for cat, (p, t) in sorted(by_cat.items()):
        print(f"  {cat:<14}{p:>6}{t:>7}{(p/t*100):>7.0f}%")
    print("  " + "-" * 33)
    print(f"  {'OVERALL':<14}{total_pass:>6}{total:>7}{(total_pass/total*100 if total else 0):>7.0f}%")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n  Example failures (this is where the system is weakest):")
        for r in failures[:5]:
            print(f"\n  ✗ {r['scenario']} [{r['category']}]")
            print(f"      asked : {r['ask']}")
            print(f"      reply : {r['reply'][:220]}")
            print(f"      why   : {r['rationale']}")

    os.makedirs("eval/results", exist_ok=True)
    out = f"eval/results/results_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full transcript-level results written to {out}\n")


if __name__ == "__main__":
    main()
