# Wait until cores 0-3 are not oversubscribed
while true; do
    load=$(mpstat -P 0,1,2,3 1 1 | tail -n +4 | awk '{sum+=$3+$5} END {print sum/4}')
    if (( $(echo "$load < 90" | bc -l) )); then
        break
    fi
    sleep 300
done

# Launch new job on cores 0-3
taskset -c 0-3 mpirun -np 4 lmp -in input.lammps > output.log 2>&1 &