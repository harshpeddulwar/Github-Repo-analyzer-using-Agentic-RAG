import os
from git import Repo

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
    ".yml"
}

# Create repository storage directory
os.makedirs(REPO_DIR, exist_ok=True)


def clone_repository(github_url):
    """
    Clone a GitHub repository into the repos directory.
    """

    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join(REPO_DIR, repo_name)

    # If repository already exists, don't clone again
    if os.path.exists(local_path):
        print(f"Repository already exists: {local_path}")
        return local_path

    try:
        Repo.clone_from(github_url, local_path)
        print(f"Repository cloned successfully: {local_path}")

    except Exception as e:
        print(f"Error cloning repository: {e}")
        return None

    return local_path


def walk_repository(local_path):
    """
    Walk through the repository and return useful source/documentation files.
    """

    file_paths = []

    if not os.path.isdir(local_path):
        print(f"Repository path does not exist: {local_path}")
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

    return file_paths
