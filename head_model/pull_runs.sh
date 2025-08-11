#!/bin/bash
#
# Pull runs directory from remote system back to laptop
#
# Usage:
#   ./pull_runs.sh           # Perform actual transfer
#   ./pull_runs.sh --dry-run # Show what would be transferred (no actual copy)
#   ./pull_runs.sh -n        # Same as --dry-run
#

REMOTE="pc"
REMOTE_DIR="/home/eph/beaker"
LOCAL_DIR="/Users/eph/beaker"

set -e  # Exit on any error

# Check for dry-run flag
DRY_RUN=""
if [[ "$1" == "--dry-run" || "$1" == "-n" ]]; then
    DRY_RUN="--dry-run"
    echo "🔍 DRY RUN MODE - No files will be transferred"
    echo ""
elif [[ -n "$1" ]]; then
    echo "❌ Error: Unknown argument '$1'"
    echo ""
    echo "Usage:"
    echo "  $0           # Perform actual transfer"
    echo "  $0 --dry-run # Show what would be transferred (no actual copy)"
    echo "  $0 -n        # Same as --dry-run"
    exit 1
fi

echo "⬇️  Starting runs pull from remote system..."

# Check if local directory exists, create if not
if [[ -z "$DRY_RUN" ]]; then
    echo "📁 Ensuring local directory structure exists..."
    mkdir -p "$LOCAL_DIR/head_model/runs"
else
    echo "📁 [DRY RUN] Would ensure local directory structure exists..."
fi

# Pull runs directory from remote
echo "🏃 Pulling runs directory..."
echo "   Source: $REMOTE:$REMOTE_DIR/head_model/runs"
echo "   Destination: $LOCAL_DIR/head_model/runs"
rsync -avz --progress $DRY_RUN \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    $REMOTE:$REMOTE_DIR/head_model/runs/ $LOCAL_DIR/head_model/runs/

# Verify the transfer
if [[ -z "$DRY_RUN" ]]; then
    echo "✅ Verifying transfer..."
    echo "Local runs directory contents:"
    ls -la "$LOCAL_DIR/head_model/runs/" | head -10

    echo "🎉 Runs pull completed successfully!"
else
    echo ""
    echo "🔍 DRY RUN COMPLETED - No files were actually transferred"
    echo "   Run without --dry-run to perform the actual transfer"
fi
