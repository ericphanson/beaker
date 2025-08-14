#!/bin/bash
#
# Pull a subdirectory from remote system back to laptop
#
# Usage:
#   ./pull_runs.sh [subdir] [--dry-run]
#   ./pull_runs.sh [subdir] [-n]
#
# Examples:
#   ./pull_runs.sh                    # Pull runs directory (default)
#   ./pull_runs.sh data               # Pull data subdirectory
#   ./pull_runs.sh runs --dry-run     # Dry run for runs directory
#   ./pull_runs.sh data -n            # Dry run for data directory
#

REMOTE="pc"
REMOTE_DIR="/home/eph/beaker"
LOCAL_DIR="/Users/eph/beaker"

set -e  # Exit on any error

# Parse arguments
SUBDIR="output"  # Default subdirectory
DRY_RUN=""

# Handle arguments - order matters: subdir first, then flags
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n)
            DRY_RUN="--dry-run"
            ;;
        -*)
            echo "❌ Error: Unknown flag '$arg'"
            echo ""
            echo "Usage:"
            echo "  $0 [subdir] [--dry-run|-n]"
            echo ""
            echo "Examples:"
            echo "  $0                    # Pull runs directory (default)"
            echo "  $0 data               # Pull data subdirectory"
            echo "  $0 runs --dry-run     # Dry run for runs directory"
            echo "  $0 data -n            # Dry run for data directory"
            exit 1
            ;;
        *)
            # If it's not a flag, treat it as the subdirectory
            if [[ "$SUBDIR" == "output" ]]; then
                SUBDIR="$arg"
            else
                echo "❌ Error: Multiple subdirectories specified"
                exit 1
            fi
            ;;
    esac
done

if [[ -n "$DRY_RUN" ]]; then
    echo "🔍 DRY RUN MODE - No files will be transferred"
    echo ""
fi

echo "⬇️  Starting $SUBDIR pull from remote system..."

# Check if local directory exists, create if not
if [[ -z "$DRY_RUN" ]]; then
    echo "📁 Ensuring local directory structure exists..."
    mkdir -p "$LOCAL_DIR/detect_rfdetr/$SUBDIR"
else
    echo "📁 [DRY RUN] Would ensure local directory structure exists..."
fi

# Pull subdirectory from remote
echo "🏃 Pulling $SUBDIR directory..."
echo "   Source: $REMOTE:$REMOTE_DIR/detect_rfdetr/$SUBDIR"
echo "   Destination: $LOCAL_DIR/detect_rfdetr/$SUBDIR"
rsync -avz --progress $DRY_RUN \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    "$REMOTE:$REMOTE_DIR/detect_rfdetr/$SUBDIR/" "$LOCAL_DIR/detect_rfdetr/$SUBDIR/"

# Verify the transfer
if [[ -z "$DRY_RUN" ]]; then
    echo "✅ Verifying transfer..."
    echo "Local $SUBDIR directory contents:"
    find "$LOCAL_DIR/detect_rfdetr/$SUBDIR/" -maxdepth 1 -type f -o -type d | head -10

    echo "🎉 $SUBDIR pull completed successfully!"
else
    echo ""
    echo "🔍 DRY RUN COMPLETED - No files were actually transferred"
    echo "   Run without --dry-run to perform the actual transfer"
fi
