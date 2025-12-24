"""Generic metric worker - runs any metric class in an isolated subprocess."""
import gc
import importlib
import json
import sys
import traceback


def cleanup_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    metric = None

    def respond(msg):
        print(json.dumps(msg), flush=True)

    def log_error(msg):
        print(f"[metric_worker] ERROR: {msg}", file=sys.stderr, flush=True)

    try:
        for line in sys.stdin:
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                respond({"status": "error", "error": f"Invalid JSON: {line}"})
                continue

            cmd = msg.get("cmd")

            if cmd == "init":
                try:
                    module_name = msg["module"]
                    class_name = msg["class"]
                    kwargs = msg.get("kwargs", {})

                    module = importlib.import_module(module_name)
                    metric_cls = getattr(module, class_name)
                    metric = metric_cls(**kwargs)

                    respond({"status": "ready"})
                except Exception as e:
                    log_error(f"init failed: {e}\n{traceback.format_exc()}")
                    respond({"status": "error", "error": str(e)})

            elif cmd == "score":
                if metric is None:
                    respond({"status": "error", "error": "Metric not initialized"})
                    continue

                try:
                    result = metric.score(
                        msg["src_lang"],
                        msg["trg_lang"],
                        msg["source_texts"],
                        msg["translated_texts"],
                        msg["reference_texts"],
                    )
                    respond(
                        {
                            "status": "ok",
                            "name": result.name,
                            "corpus_score": result.corpus_score,
                            "segment_scores": result.segment_scores,
                            "details": result.details,
                        }
                    )
                except Exception as e:
                    log_error(f"score failed: {e}\n{traceback.format_exc()}")
                    respond({"status": "error", "error": str(e)})
                finally:
                    cleanup_memory()

            elif cmd == "supports_lang":
                try:
                    module_name = msg["module"]
                    class_name = msg["class"]

                    module = importlib.import_module(module_name)
                    metric_cls = getattr(module, class_name)
                    supported = metric_cls.supports_lang(msg["src_lang"], msg["trg_lang"])

                    respond({"status": "ok", "supported": supported})
                except Exception as e:
                    log_error(f"supports_lang failed: {e}\n{traceback.format_exc()}")
                    respond({"status": "error", "error": str(e)})

            elif cmd == "shutdown":
                break

    except Exception as e:
        log_error(f"Worker crashed: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
