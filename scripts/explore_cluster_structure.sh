#!/bin/bash
# Run from ~/ondemand/upload_me/RegionTokenizer/scripts/RL
# Shows structure of all relevant output directories

BASE="$HOME/ondemand/upload_me/RegionTokenizer/scripts/RL"
cd "$BASE"

echo "=== Working directory: $BASE ==="
echo ""

echo "=== Top-level directories ==="
ls -la | grep "^d"
echo ""

for DIR in outputs_track_a_offline_paper outputs_track_a_offline_paper_wikitext2 \
           outputs_track_a_offline_paper_gpt2_medium scale_200k_seed42; do
    if [ -d "$DIR" ]; then
        echo "========================================"
        echo "DIR: $DIR"
        echo "========================================"
        find "$DIR" -maxdepth 4 \( -name "*.pt" -o -name "*.json" -o -name "*.csv" -o -type d \) \
            | sort \
            | sed 's|[^/]*/|  |g'
        echo ""
    fi
done
