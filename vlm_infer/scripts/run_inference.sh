#!/bin/bash

# Default values
DEFAULT_PARTITION="nlp"
DEFAULT_ACCOUNT="nlp"
DEFAULT_EMAIL=""
DEFAULT_MEM="20000mb"
DEFAULT_TIME="23:59:59"
DEFAULT_NUM_GPUS=1
DEFAULT_DEBUG=false

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --model) MODEL_NAME="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --dataset_path) DATASET_PATH="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        --account) ACCOUNT="$2"; shift ;;
        --email) EMAIL="$2"; shift ;;
        --memory) MEMORY="$2"; shift ;;
        --time) TIME="$2"; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift ;;
        --debug) DEBUG=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check for required arguments
if [ -z "$CONFIG" ] || [ -z "$MODEL_NAME" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 --config <CONFIG_PATH> --model <MODEL_NAME> --dataset <DATASET_NAME> [OPTIONS]"
    echo "Optional arguments:"
    echo "  --dataset_path <PATH>    Path to dataset directory"
    echo "  --output_dir <DIR>       Output directory (default: results)"
    echo "  --partition <PARTITION>   SLURM partition (default: $DEFAULT_PARTITION)"
    echo "  --account <ACCOUNT>      SLURM account (default: $DEFAULT_ACCOUNT)"
    echo "  --email <EMAIL>          Email for notifications"
    echo "  --memory <MEM>           Memory per CPU (default: $DEFAULT_MEM)"
    echo "  --time <TIME>            Wall time limit (default: $DEFAULT_TIME)"
    echo "  --num_gpus <NUM>         Number of GPUs (default: $DEFAULT_NUM_GPUS)"
    echo "  --debug                  Run in debug mode"
    exit 1
fi

# Use defaults if not provided
PARTITION=${PARTITION:-$DEFAULT_PARTITION}
ACCOUNT=${ACCOUNT:-$DEFAULT_ACCOUNT}
EMAIL=${EMAIL:-$DEFAULT_EMAIL}
MEMORY=${MEMORY:-$DEFAULT_MEM}
TIME=${TIME:-$DEFAULT_TIME}
NUM_GPUS=${NUM_GPUS:-$DEFAULT_NUM_GPUS}
DEBUG=${DEBUG:-$DEFAULT_DEBUG}
OUTPUT_DIR=${OUTPUT_DIR:-"results"}

# Create SLURM script
SLURM_SCRIPT="slurm_scripts/run_${MODEL_NAME}_${DATASET}.sh"
mkdir -p slurm_scripts

cat << EOF > $SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=vlm_${MODEL_NAME}_${DATASET}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --gpus=${NUM_GPUS}
#SBATCH --mem-per-cpu=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=slurm_logs/%j.log
EOF

# Add email notification if provided
if [ ! -z "$EMAIL" ]; then
    echo "#SBATCH --mail-user=${EMAIL}" >> $SLURM_SCRIPT
    echo "#SBATCH --mail-type=END,FAIL" >> $SLURM_SCRIPT
fi

# Add job content
cat << EOF >> $SLURM_SCRIPT

echo "Date              = \$(date)"
echo "Hostname          = \$(hostname -s)"
echo "Working Directory = \$(pwd)"
echo ""
echo "Number of Nodes Allocated      = \$SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = \$SLURM_NTASKS"
echo "Number of GPUs Allocated       = \$SLURM_GPUS_PER_TASK"

# Load any necessary modules or activate conda environment here
# module load cuda/11.7
# source activate your_env

# Run the inference
python -m vlm_infer.main \\
    --config ${CONFIG} \\
    --model_name ${MODEL_NAME} \\
    --dataset ${DATASET} \\
EOF

# Add optional arguments
if [ ! -z "$DATASET_PATH" ]; then
    echo "    --dataset_path ${DATASET_PATH} \\" >> $SLURM_SCRIPT
fi

echo "    --output_dir ${OUTPUT_DIR} \\" >> $SLURM_SCRIPT
echo "    --num_gpus ${NUM_GPUS} \\" >> $SLURM_SCRIPT

if [ "$DEBUG" = true ]; then
    echo "    --debug \\" >> $SLURM_SCRIPT
fi

echo "    --device cuda" >> $SLURM_SCRIPT

echo "Job completed at \$(date)" >> $SLURM_SCRIPT

# Create logs directory
mkdir -p slurm_logs

# Submit the job
sbatch $SLURM_SCRIPT 