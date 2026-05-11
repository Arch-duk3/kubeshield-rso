import csv
import glob
import numpy as np
import os

def aggregate_logs():
    csv_files = glob.glob("controller/experiment_seed_*.csv")
    if not csv_files:
        print("No seed log files found. Ensure train() was run.")
        return

    data_by_seed = []
    headers = None

    for file in csv_files:
        with open(file, "r") as f:
            reader = csv.reader(f)
            h = next(reader)
            if not headers:
                headers = h
            
            seed_data = []
            for row in reader:
                seed_data.append([float(x) for x in row])
            data_by_seed.append(seed_data)
            
    # shape: (seeds, timesteps, metrics)
    data_by_seed = np.array(data_by_seed) 
    
    means = np.mean(data_by_seed, axis=0)
    stds = np.std(data_by_seed, axis=0)
    num_seeds = data_by_seed.shape[0]
    cis = 1.96 * stds / np.sqrt(num_seeds)
    
    with open("aggregated_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header_row = []
        for h in headers:
            header_row.extend([f"{h}_mean", f"{h}_std", f"{h}_ci"])
        writer.writerow(header_row)
        
        for t in range(means.shape[0]):
            row = []
            for i in range(len(headers)):
                row.extend([f"{means[t, i]:.4f}", f"{stds[t, i]:.4f}", f"{cis[t, i]:.4f}"])
            writer.writerow(row)
            
    print(f"Aggregated {len(csv_files)} logs into aggregated_results.csv")

if __name__ == "__main__":
    aggregate_logs()
