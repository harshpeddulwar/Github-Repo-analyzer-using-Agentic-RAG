import os
from git import Repo
from src.logger import get_logger

logger = get_logger("cloning")

REPO_DIR = "repos"

# Directories that should not be scanned
IGNORE_DIRS = {
    ".git",
    "venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__"
}

# File types useful for code/documentation analysis
ALLOWED_EXTENSIONS = {
    ".py",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
}

# Create repository storage directory
os.makedirs(REPO_DIR, exist_ok=True)


def repo_id_from_path(local_path):
    """Stable ID used to tag/filter a repository's vectors in Qdrant."""
    return os.path.basename(os.path.normpath(local_path))


def clone_repository(github_url):
    """
    Clone a GitHub repository into the repos directory.
    """
    logger.info("[INGESTION - STEP 1/4] Resolving repository from URL: %s", github_url)

    # Only plain http(s) URLs: blocks git's "ext::" / "file::" transports.
    if not str(github_url).startswith(("https://", "http://")):
        logger.error("[INGESTION - STEP 1/4] Refusing to clone non-http(s) URL: %r", github_url)
        return None

    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join(REPO_DIR, repo_name)

    # If repository already exists, don't clone again
    if os.path.exists(local_path):
        logger.info("[INGESTION - STEP 1/4] Repository directory already exists at: %s", local_path)
        return local_path

    try:
        logger.info("[INGESTION - STEP 1/4] Cloning '%s' into '%s'...", github_url, local_path)
        Repo.clone_from(github_url, local_path)
        logger.info("[INGESTION - STEP 1/4] Repository cloned successfully: %s", local_path)

    except Exception as e:
        logger.error("[INGESTION - STEP 1/4] Error cloning repository '%s': %s", github_url, e)
        return None

    return local_path


def walk_repository(local_path):
    """
    Walk through the repository and return useful source/documentation files.
    """
    file_paths = []

    if not os.path.isdir(local_path):
        logger.warning("Repository path does not exist: %s", local_path)
        return file_paths

    for root, dirs, files in os.walk(local_path):

        # Prevent os.walk from entering ignored directories
        dirs[:] = [
            d for d in dirs
            if d not in IGNORE_DIRS
        ]

        for file in files:

            # Get file extension
            extension = os.path.splitext(file)[1].lower()

            # Skip unsupported files
            if extension not in ALLOWED_EXTENSIONS:
                continue

            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    logger.debug("Walked '%s': discovered %d candidate source files.", local_path, len(file_paths))
    return file_paths
