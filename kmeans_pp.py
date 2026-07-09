import sys
import numpy as np
import pandas as pd
import mykmeanssp

def is_valid_integer(val_str):
    """Checks if a string represents a valid integer (rejects floats like '3.5')."""
    try:
        val_float = float(val_str)
        return val_float.is_integer()
    except ValueError:
        return False

def parse_arguments():
    """
    Parses and strictly validates CLI arguments.
    """
    args = sys.argv[1:]
    if len(args) < 3 or len(args) > 5:
        print("An Error Has Occurred")
        sys.exit(1)
        
    file2 = args[-1]
    file1 = args[-2]
    
    # Epsilon validation
    try:
        eps = float(args[-3])
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)
        
    if eps < 0:
        print("Incorrect epsilon!")
        sys.exit(1)
        
    K = 3      # Default
    iters = 300 # Default
    
    # K and iter validation based on number of arguments
    if len(args) == 5:
        if not is_valid_integer(args[0]):
            print("Incorrect number of clusters!")
            sys.exit(1)
        K = int(float(args[0]))
        
        if not is_valid_integer(args[1]):
            print("Incorrect maximum iteration!")
            sys.exit(1)
        iters = int(float(args[1]))
        
    elif len(args) == 4:
        if not is_valid_integer(args[0]):
            print("Incorrect number of clusters!")
            sys.exit(1)
        K = int(float(args[0]))
        
    # Boundary validations
    if K <= 1:
        print("Incorrect number of clusters!")
        sys.exit(1)
        
    if iters <= 1 or iters >= 400:
        print("Incorrect maximum iteration!")
        sys.exit(1)
        
    return K, iters, eps, file1, file2

def main():
    K, iters, eps, file1, file2 = parse_arguments()
    
    # 1. Read files and Merge
    try:
        df1 = pd.read_csv(file1, header=None)
        df2 = pd.read_csv(file2, header=None)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
        
    try:
        # Inner join on the first column (index 0)
        df_merged = pd.merge(df1, df2, on=0, how='inner')
        # Sort by the first column ascending
        df_merged = df_merged.sort_values(by=0).reset_index(drop=True)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
    
    N = len(df_merged)
    # Re-validate K against N after knowing N
    if K >= N:
        print("Incorrect number of clusters!")
        sys.exit(1)
        
    # Extract indices and vectors
    indices = df_merged[0].astype(int).tolist()
    data_points = df_merged.drop(columns=[0]).to_numpy()
    
    # 2. K-Means++ Initialization
    np.random.seed(1234)
    
    centroids_indices = []
    centroids_data = []
    
    # Choose first centroid randomly
    first_idx = np.random.choice(N)
    centroids_indices.append(indices[first_idx])
    centroids_data.append(data_points[first_idx])
    
    # D array to keep track of shortest distance from each point to any chosen centroid
    D = np.full(N, np.inf)
    
    for _ in range(1, K):
        last_centroid = centroids_data[-1]
        
        # Calculate Euclidean distance to the last chosen centroid
        distances_to_last = np.linalg.norm(data_points - last_centroid, axis=1)
        
        # Update D(x) if distance to the new centroid is smaller
        D = np.minimum(D, distances_to_last)
        
        # Calculate probabilities P(x) = D(x) / sum(D(x))
        sum_D = np.sum(D)
        if sum_D == 0:
            break
            
        P = D / sum_D
        
        # Choose next centroid
        next_idx = np.random.choice(N, p=P)
        centroids_indices.append(indices[next_idx])
        centroids_data.append(data_points[next_idx])
        
    # 3. Output first line (indices)
    print(",".join(map(str, centroids_indices)))
    
    # 4. C Extension Call
    data_list = data_points.tolist()
    centroids_list = [c.tolist() for c in centroids_data]
    
    try:
        final_centroids = mykmeanssp.fit(K, iters, eps, centroids_list, data_list)
        
        # Output final centroids
        for c in final_centroids:
            print(",".join([f"{x:.4f}" for x in c]))
            
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == '__main__':
    main()