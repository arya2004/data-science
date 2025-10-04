"""
K-Nearest Neighbors Implementation
Simple and human-readable version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def load_data():
    """Load the dataset"""
    data = pd.read_csv('../datasets/knn1_csv.csv')
    return data


def distance(point1, point2):
    """Calculate distance between two points"""
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def nearest_neighbor(data, query_point):
    """Find the closest neighbor"""
    min_dist = float('inf')
    closest_class = None
    
    for _, row in data.iterrows():
        point = [row['x'], row['y']]
        dist = distance(query_point, point)
        
        if dist < min_dist:
            min_dist = dist
            closest_class = row['class']
    
    return closest_class


def k_nearest_neighbors(data, query_point, k=5):
    """Find k nearest neighbors and vote for class"""
    distances = []
    
    # Calculate all distances
    for _, row in data.iterrows():
        point = [row['x'], row['y']]
        dist = distance(query_point, point)
        distances.append((dist, row['class']))
    
    # Sort by distance and take first k
    distances.sort()
    k_nearest = distances[:k]
    
    # Count votes
    votes = {}
    for _, class_label in k_nearest:
        votes[class_label] = votes.get(class_label, 0) + 1
    
    # Return most voted class
    return max(votes, key=votes.get)


def radius_neighbors(data, query_point, radius=1.45):
    """Find all neighbors within radius"""
    neighbors = []
    
    for _, row in data.iterrows():
        point = [row['x'], row['y']]
        dist = distance(query_point, point)
        
        if dist <= radius:
            neighbors.append(row['class'])
    
    if not neighbors:
        return "No neighbors found"
    
    # Count votes
    votes = {}
    for class_label in neighbors:
        votes[class_label] = votes.get(class_label, 0) + 1
    
    return max(votes, key=votes.get)


def weighted_knn(data, query_point, k=5):
    """K-NN with weighted voting (closer points have more weight)"""
    distances = []
    
    # Calculate all distances
    for _, row in data.iterrows():
        point = [row['x'], row['y']]
        dist = distance(query_point, point)
        distances.append((dist, row['class']))
    
    # Sort by distance and take first k
    distances.sort()
    k_nearest = distances[:k]
    
    # Weighted voting
    weights = {}
    for dist, class_label in k_nearest:
        weight = 1 / (dist + 0.001)  # Add small value to avoid division by zero
        weights[class_label] = weights.get(class_label, 0) + weight
    
    return max(weights, key=weights.get)


def plot_data(data, query_point):
    """Plot the data points and query point"""
    plt.figure(figsize=(10, 6))
    
    # Plot each class with different colors
    for class_name in data['class'].unique():
        class_data = data[data['class'] == class_name]
        plt.scatter(class_data['x'], class_data['y'], label=f'Class {class_name}', alpha=0.7)
    
    # Plot query point
    plt.scatter(query_point[0], query_point[1], color='red', marker='x', 
                s=200, label='Query Point')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Nearest Neighbors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    # Load data
    data = load_data()
    query_point = [2, 3]
    
    print("K-Nearest Neighbors Results")
    print(f"Query point: {query_point}")
    print("-" * 30)
    
    # Test different algorithms
    nn_result = nearest_neighbor(data, query_point)
    print(f"Nearest Neighbor: {nn_result}")
    
    knn5_result = k_nearest_neighbors(data, query_point, k=5)
    print(f"KNN (k=5): {knn5_result}")
    
    knn7_result = k_nearest_neighbors(data, query_point, k=7)
    print(f"KNN (k=7): {knn7_result}")
    
    rnn_result = radius_neighbors(data, query_point, radius=1.45)
    print(f"Radius NN (r=1.45): {rnn_result}")
    
    mknn_result = weighted_knn(data, query_point, k=5)
    print(f"Weighted KNN (k=5): {mknn_result}")
    
    # Show plot
    plot_data(data, query_point)


if __name__ == "__main__":
    main()