#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "source" && "$current_branch" != "main" ]]; then
  echo "Publish from the source branch (or the Sites main checkout), not $current_branch." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Commit or stash source changes before publishing." >&2
  exit 1
fi

github_remote="${GITHUB_REMOTE:-origin}"
if ! git remote get-url "$github_remote" >/dev/null 2>&1; then
  echo "Git remote '$github_remote' does not exist." >&2
  exit 1
fi

github_url="$(git remote get-url "$github_remote")"
if [[ "$github_url" != *"github.com/chenhuiyu/chenhuiyu.github.io"* ]]; then
  if git remote get-url github >/dev/null 2>&1; then
    github_remote="github"
  else
    echo "No GitHub remote for chenhuiyu/chenhuiyu.github.io was found." >&2
    exit 1
  fi
fi

npm run export:static

if [[ "${PUBLISH_CONFIRM:-}" != "YES" ]]; then
  read -r -p "Replace the production master branch with this static export? [y/N] " answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Publish cancelled."
    exit 0
  fi
fi

publish_root="$(mktemp -d)"
publish_worktree="$publish_root/master"

cleanup() {
  git worktree remove --force "$publish_worktree" >/dev/null 2>&1 || true
  rm -rf "$publish_root"
}
trap cleanup EXIT

git fetch "$github_remote" master
git worktree add --detach "$publish_worktree" "$github_remote/master"

find "$publish_worktree" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$project_root/static-export/." "$publish_worktree/"
touch "$publish_worktree/.nojekyll"

git -C "$publish_worktree" add -A
if git -C "$publish_worktree" diff --cached --quiet; then
  echo "GitHub Pages is already up to date."
  exit 0
fi

source_sha="$(git rev-parse --short HEAD)"
git -C "$publish_worktree" commit -m "Publish site from source ${source_sha}"
git -C "$publish_worktree" push "$github_remote" HEAD:master

echo "Published https://chenhuiyu.github.io/"
