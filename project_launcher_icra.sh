# Function to run a command and wait for a specific phrase
run_and_wait() {
    local cmd="$1"
    local phrase="$2" 
    
    # Wait for the phrase to appear in the log
    echo "Waiting for '$phrase'..."
    local log_file=$(mktemp)
    # Start command in the background
    eval "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    tail -f -n 0 "$log_file" | grep -m 1 "$phrase" >/dev/null
    rm -f "$log_file"
    
    return $pid
}
cleanup() {
    local pids=("$@")
    echo "Stopping all Docker containers..."
    docker stop $(docker ps -q) 2>/dev/null
    
    echo "Sending graceful shutdown signal (SIGINT) to background processes..."
    for pid in "${pids[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null  # Sends SIGINT (equivalent to Ctrl+C)
        fi
    done
    
    # Wait up to 10 seconds for processes to finish cleaning up the camera
    sleep 10
    
    # Force-kill any processes that are still running
    for pid in "${pids[@]}"; do
        while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do
            echo "Process $pid did not exit. Sending again SIGINT..."
            kill -INT "$pid" 2>/dev/null
            sleep 3
        done
    done
    echo "Cleanup complete."
}

# Declare an array to hold PIDs of background processes
declare -a pids

# Register cleanup to run on script exit
trap 'cleanup "${pids[@]}"' EXIT

# Wait for Docker to be ready
while ! docker info >/dev/null 2>&1; do
    sleep 1
done

# Command sequences with their corresponding wait phrases
# Format: (command phrase) pairs
commands=(
    "cd ~/prova/Vision_Docker_Orrbec && docker compose up mqtt" "mosquitto version"
    "cd ~/prova/Vision_Docker_Orrbec&& python -u inference_hand.py" "OBD2CTransform"
    "cd ~/prova/Supreme-Docker-qb-Integration && docker compose up" "Basic environment generated."
)

# Run commands sequentially
for ((i=0; i<${#commands[@]}; i+=2)); do
    run_and_wait "${commands[i]}" "${commands[i+1]}"
    pids[$((i/2))]=$!
done

echo "All services started successfully! Starting task..."

docker exec supreme-docker-qb-integration-pnrr_project_jazzy-1 bash -c 'source home/blah/workspace/.bash_sources && ros2 service call /start_trigger std_srvs/srv/Empty'

