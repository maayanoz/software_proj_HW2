import matplotlib
matplotlib.use('Agg') # מגדיר ל-matplotlib לשמור קובץ בלי לנסות לפתוח חלון תצוגה
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

def main():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data

    inertias = []
    
    # Calculate inertia for K=1 to K=10 using K-Means++
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plotting the line chart
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertias, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.xlabel('K')
    plt.ylabel('Average Dispersion')
    
    # Annotating the Elbow Point at K=3
    elbow_k = 3
    elbow_inertia = inertias[elbow_k - 1]
    
    plt.annotate('Elbow Point',
                 xy=(elbow_k, elbow_inertia),
                 xytext=(elbow_k + 1.5, elbow_inertia + 150),
                 arrowprops=dict(facecolor='black', shrink=0.05, linestyle='--', width=1.5, headwidth=8))
    
    # Save the plot as requested
    plt.savefig('elbow.png')

if __name__ == "__main__":
    main()