"""Log an exported dataset folder as a versioned W&B artifact. Runs inside the repo .venv (has wandb).
Usage: wandb_log.py <dataset_dir> <result_json>   Prints the artifact URL on the last line."""
import json, os, sys
import wandb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vss import load_env
ENV = load_env()
os.environ.setdefault("WANDB_API_KEY", ENV.get("WANDB_API_KEY", ""))
os.environ.setdefault("WANDB_DIR", "/tmp")
root, result = sys.argv[1], json.loads(sys.argv[2])
team, project = ENV.get("WANDB_TEAM"), ENV.get("WANDB_PROJECT")
run = wandb.init(entity=team, project=project, job_type="edge-case-mining", name=result["name"],
                 config={"clips": result["clips"], "by_category": result["by_category"]})
art = wandb.Artifact("edge-case-dataset", type="dataset", metadata=result,
                     description="Rare driving scenarios mined by Haystack")
art.add_dir(root)
run.log({"clips": result["clips"], **{f"category/{k}": v for k, v in result["by_category"].items()}})
run.log_artifact(art)
run.finish()
print(run.url)
