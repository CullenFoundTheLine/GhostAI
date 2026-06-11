#!/bin/bash
# push_to_github.sh
# Run this from inside the folder where you extracted GhostAI.zip
#
# Usage:
#   bash push_to_github.sh             ← commits and pushes to main
#   bash push_to_github.sh master      ← if your branch is master

BRANCH=${1:-main}
REMOTE="https://github.com/CullenFoundTheLine/GhostAI.git"

echo "[Ghost AI] Pushing restored project to GitHub..."
echo "[Ghost AI] Branch: $BRANCH"
echo ""

# Initialize git if needed
if [ ! -d ".git" ]; then
  git init
  git remote add origin "$REMOTE"
  echo "[Ghost AI] Initialized new git repo."
else
  # Make sure origin is set correctly
  git remote set-url origin "$REMOTE" 2>/dev/null || git remote add origin "$REMOTE"
  echo "[Ghost AI] Using existing git repo."
fi

# Stage all files
git add .

# Commit
git commit -m "Restore original GhostAI — remove Copilot overrides, add session_repository.py"

# Push
echo ""
echo "[Ghost AI] Pushing to $REMOTE ($BRANCH)..."
git push -u origin "$BRANCH"

echo ""
echo "[Ghost AI] Done. Check https://github.com/CullenFoundTheLine/GhostAI"
