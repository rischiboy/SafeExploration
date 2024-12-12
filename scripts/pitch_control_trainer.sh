#!/bin/bash

timestamp=$(date +"%Y-%m-%dT%H:%M")
dir_name="Pitch-Control_${timestamp}"
log_dir_path="run/logs/${dir_name}"
mkdir -p "$log_dir_path"

log_file_name="${log_dir_path}/training.log"

# Creating a Slurm batch script dynamically
cat <<EOT >> job_script.sh
#!/bin/bash
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=8:00:00
#SBATCH --output="${log_file_name}"

python ./run/run_trainer.py --config ./run/config/bnn_dynamics_pitch_control.yaml --log_dir "${log_dir_path}/"
EOT

# Make the generated script executable (if needed)
chmod +x job_script.sh

# Submit the job using the generated script
sbatch < job_script.sh

rm job_script.sh