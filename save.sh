#!/bin/bash
#
# This script adds all files, commits with a custom or default message,
# and pushes to the default remote/branch.

# Check if an argument ($1) was provided.
# -n "$1" checks if the string $1 is not empty.
if [ -n "$1" ]; then
    # If an argument exists, use it as the commit message.
    COMMIT_MESSAGE="$1"
else
    # If no argument is provided, use the default message.
    COMMIT_MESSAGE="Saved work quickly"
fi

echo "Adding all files..."
git add .

echo "Committing with message: '$COMMIT_MESSAGE'"
git commit -m "$COMMIT_MESSAGE"

echo "Pushing to origin..."
git push
