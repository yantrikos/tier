#!/usr/bin/env python3
"""Tier Bridge."""
import sys, json, signal
from tier_engine.engine import TierEngine
_engine = None
def get_engine(config=None):
    global _engine
    if _engine is None: _engine = TierEngine(config)
    return _engine
def handle_command(command, args, config):
    engine = get_engine(config)
    try:
        if command == "route":
            return {"success": True, **engine.route_and_format(args.get("intent", ""), args.get("model_name", ""))}
        elif command == "resolve_mcq":
            tool = engine.resolve_mcq(args["choice"], args.get("tools", []), args.get("mcq_options", {}))
            return {"success": bool(tool), "tool": tool}
        elif command == "detect_tier":
            return {"success": True, **engine.detect_tier(args.get("model_name", ""))}
        elif command == "register_tool":
            return {"success": engine.register_tool(**{k:v for k,v in args.items()})}
        elif command == "record_usage":
            engine.record_usage(**{k:v for k,v in args.items()})
            return {"success": True}
        elif command == "stats": return {"success": True, **engine.stats()}
        elif command == "health_check": return engine.health_check()
        elif command == "list_tools": return {"success": True, "tools": engine.router.list_tools()}
        else: return {"success": False, "error": f"Unknown: {command}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
def main():
    if "--persistent" in sys.argv:
        signal.signal(signal.SIGTERM, lambda s,f: (_engine and _engine.close(), sys.exit(0)))
        for line in sys.stdin:
            if not line.strip(): continue
            try:
                d = json.loads(line)
                r = handle_command(d.get("command",""), d.get("args",{}), d.get("config",{}))
                r["request_id"] = d.get("request_id")
                print(json.dumps(r), flush=True)
            except Exception as e: print(json.dumps({"success": False, "error": str(e)}), flush=True)
    else:
        try: d = json.loads(sys.stdin.read())
        except: d = {}
        print(json.dumps(handle_command(d.get("command",""), d.get("args",{}), d.get("config",{}))))
if __name__ == "__main__": main()
