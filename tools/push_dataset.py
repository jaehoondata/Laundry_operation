"""Upload a finalized VAI_AIWORKER dataset to a private Hugging Face Hub repo and tag it.

Run on the Mac (VAI_AIWORKER's ri_motion_v5_py312 env already has lerobot 0.6.1),
after review_dataset.ipynb and tools/select_joints.py have produced the `*_final_arm16` dataset:

    python tools/push_dataset.py ~/aiworker_data/laundry_fold_tshirt_eef_final_arm16 \
        --repo-id jaehoon2da/laundry_fold_tshirt_arm16 --tag v0.1

Training then pins the exact data with --dataset.repo_id / --dataset.revision=<tag>.
The recorder writes with repo_id="local/...", so the Hub repo_id is given here instead.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="finalized LeRobot v3.0 dataset root (has meta/info.json)")
    parser.add_argument("--repo-id", required=True, help="e.g. jaehoon2da/laundry_fold_tshirt_arm16")
    parser.add_argument("--tag", required=True, help="dataset version tag, e.g. v0.1")
    parser.add_argument("--public", action="store_true", help="create a public repo (default: private)")
    args = parser.parse_args()

    root = args.root.expanduser()
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit("%s 에 meta/info.json 이 없습니다. finalize 된 데이터셋 경로인지 확인하세요." % root)

    ds = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    print("%s: %d episodes, %d frames, %s fps" % (root, ds.meta.total_episodes, ds.meta.total_frames, ds.meta.fps))

    api = HfApi()
    if api.repo_exists(args.repo_id, repo_type="dataset"):
        tags = {t.name for t in api.list_repo_refs(args.repo_id, repo_type="dataset").tags}
        if args.tag in tags:
            raise SystemExit("tag %s 가 이미 있습니다. 새 버전 태그를 지정하세요." % args.tag)

    # tag_version=True keeps lerobot's codebase-version tag (v3.0), which the loader expects.
    ds.push_to_hub(private=not args.public, tag_version=True, upload_large_folder=True)
    api.create_tag(args.repo_id, tag=args.tag, repo_type="dataset")
    print("pushed https://huggingface.co/datasets/%s (tag %s)" % (args.repo_id, args.tag))


if __name__ == "__main__":
    main()
