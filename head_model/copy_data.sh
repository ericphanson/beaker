#!/bin/bash
#
# Copy data and runs directories to remote system
#
# Usage:
#   ./copy_data.sh           # Perform actual transfer (fast mode for data, normal for runs)
#   ./copy_data.sh --dry-run # Show what would be transferred (no actual copy)
#   ./copy_data.sh -n        # Same as --dry-run
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

echo "🚀 Starting data transfer to remote system..."

# Check if remote directory exists, create if not
if [[ -z "$DRY_RUN" ]]; then
    echo "📁 Ensuring remote directory structure exists..."
    ssh $REMOTE "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/head_model/runs"
else
    echo "📁 [DRY RUN] Would ensure remote directory structure exists..."
fi

# Copy data directory
echo "📊 Copying data directory (fast mode for many small files)..."
echo "   Source: $LOCAL_DIR/data"
echo "   Destination: $REMOTE:$REMOTE_DIR/data"

# Fast mode for data: optimized for many small files
rsync -a --info=progress2 $DRY_RUN \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='*_cached_*.torch' \
    --exclude='*_cached_*.pt' \
    --exclude='*.cache' \
    --exclude='*_cache_*' \
    --exclude='.cache/' \
    --exclude='wandb/' \
    --exclude='.neptune/' \
    --exclude='mlruns/' \
    --no-compress \
    --whole-file \
    --inplace \
    --partial \
    --no-perms \
    --no-owner \
    --no-group \
    $LOCAL_DIR/data/ $REMOTE:$REMOTE_DIR/data/

# Copy runs directory
echo "🏃 Copying runs directory (normal mode for fewer, larger files)..."
echo "   Source: $LOCAL_DIR/head_model/runs"
echo "   Destination: $REMOTE:$REMOTE_DIR/head_model/runs"

# Normal mode for runs: balanced compression and speed
rsync -az --info=progress2 $DRY_RUN \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='*_cached_*.torch' \
    --exclude='*_cached_*.pt' \
    --exclude='*.cache' \
    --exclude='*_cache_*' \
    --exclude='.cache/' \
    --exclude='wandb/' \
    --exclude='.neptune/' \
    --exclude='mlruns/' \
    --compress-level=1 \
    --whole-file \
    --inplace \
    --partial \
    $LOCAL_DIR/head_model/runs/ $REMOTE:$REMOTE_DIR/head_model/runs/

# Verify the transfer
if [[ -z "$DRY_RUN" ]]; then
    echo "✅ Verifying transfer..."
    echo "Remote data directory contents:"
    ssh $REMOTE "ls -la $REMOTE_DIR/data/ | head -10"
    echo ""
    echo "Remote runs directory contents:"
    ssh $REMOTE "ls -la $REMOTE_DIR/head_model/runs/ | head -10"

    echo "🎉 Data transfer completed successfully!"
else
    echo ""
    echo "🔍 DRY RUN COMPLETED - No files were actually transferred"
    echo "   Run without --dry-run to perform the actual transfer"
fi
