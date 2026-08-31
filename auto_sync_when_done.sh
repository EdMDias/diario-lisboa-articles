#!/bin/bash
# Auto-sync OCR output when processing completes

SSH_HOST="root@193.143.121.70"
SSH_PORT="19971"
REMOTE_LOG="/workspace/batch_ocr_new.log"
REMOTE_DIR="/workspace/ocr_output"
LOCAL_DIR="/home/eduardomdias/Projects/Personal/diario-lisbon/ocr_output"

echo "🔍 Monitoring OCR processing on vast.ai..."
echo "Will auto-sync when complete. You can close this terminal - it runs in background."
echo ""

while true; do
    # Check if process is still running
    PROCESS_RUNNING=$(ssh -p $SSH_PORT $SSH_HOST "pgrep -f 'batch_ocr_vllm.py'" 2>/dev/null)

    if [ -z "$PROCESS_RUNNING" ]; then
        echo ""
        echo "✅ Processing completed! Starting sync..."

        # Run rsync to copy files
        rsync -avz --progress -e "ssh -p $SSH_PORT" \
            $SSH_HOST:$REMOTE_DIR/ $LOCAL_DIR/

        echo ""
        echo "🎉 Sync complete!"
        echo "Total files: $(find $LOCAL_DIR -name '*_ocr.json' | wc -l)"

        # Show final stats
        FINAL_STATS=$(ssh -p $SSH_PORT $SSH_HOST "tail -1 $REMOTE_LOG")
        echo ""
        echo "Final status: $FINAL_STATS"

        exit 0
    fi

    # Show progress every 5 minutes
    PROGRESS=$(ssh -p $SSH_PORT $SSH_HOST "grep 'Processing images:' $REMOTE_LOG | tail -1" 2>/dev/null)
    echo "[$(date +'%H:%M:%S')] $PROGRESS"

    # Check every 5 minutes
    sleep 300
done
