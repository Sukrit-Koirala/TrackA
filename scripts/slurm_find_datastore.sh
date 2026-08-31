#!/bin/bash
#SBATCH --job-name=find_datastore
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/find_datastore-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/find_datastore-%j.err

BASE=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer

echo "=== Searching for datastore.pt ==="
find $BASE -name "datastore.pt" 2>/dev/null

echo ""
echo "=== Top-level of outputs_track_a_offline_paper ==="
ls -la $BASE/outputs_track_a_offline_paper/ 2>/dev/null || echo "Directory does not exist"

echo ""
echo "=== Recursive tree (depth 5) of outputs_track_a_offline_paper ==="
find $BASE/outputs_track_a_offline_paper -maxdepth 5 -type f 2>/dev/null | sort

echo ""
echo "=== All .pt files anywhere under BASE ==="
find $BASE -name "*.pt" 2>/dev/null | sort
