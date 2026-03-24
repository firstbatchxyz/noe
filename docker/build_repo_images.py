#!/usr/bin/env python3
"""Build Docker images for SWE-bench repos.

Pre-builds one image per repo (~12 repos) to avoid setup time during episodes.
~2-4 hours for all repos.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common SWE-bench repos
DEFAULT_REPOS = [
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
]


def build_repo_image(repo: str, base_image: str = "noe-sandbox:base") -> bool:
    """Build Docker image for a specific repo."""
    org, name = repo.split("/")
    tag = f"noe-sandbox:{name}"

    dockerfile_content = f"""FROM {base_image}
RUN git clone --depth=1 https://github.com/{repo}.git /workspace/repo
WORKDIR /workspace/repo
RUN pip install --no-cache-dir -e . 2>/dev/null || true
"""

    try:
        result = subprocess.run(
            ["docker", "build", "-t", tag, "-"],
            input=dockerfile_content,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode == 0:
            logger.info(f"Built: {tag}")
            return True
        else:
            logger.error(f"Failed: {tag}\n{result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout building: {tag}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build repo Docker images")
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--base-image", type=str, default="noe-sandbox:base")
    parser.add_argument("--build-base", action="store_true")
    args = parser.parse_args()

    # Build base image first
    if args.build_base:
        logger.info("Building base image...")
        result = subprocess.run(
            ["docker", "build", "-t", "noe-sandbox:base", "-f", "Dockerfile.sandbox", "."],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Base image build failed:\n{result.stderr}")
            return 1

    # Build repo images
    results = {}
    for repo in args.repos:
        logger.info(f"Building image for {repo}...")
        ok = build_repo_image(repo, args.base_image)
        results[repo] = ok

    # Summary
    success = sum(1 for v in results.values() if v)
    logger.info(f"\nBuilt {success}/{len(results)} images")
    for repo, ok in results.items():
        status = "OK" if ok else "FAIL"
        logger.info(f"  {repo}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
