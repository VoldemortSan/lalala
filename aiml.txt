#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <ctime>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Mat;

// Euclidean distance
double distance(const Vec &a, const Vec &b) {
    double sum = 0.0;
    for (int i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

// Sequential K-means clustering
void kmeans_sequential(const Mat &data, int k, int max_iters, double &time) {
    int n = data.size();
    int d = data[0].size();

    // Randomly initialize centroids
    Mat centroids(k, Vec(d));
    srand(std::time(0));
    for (int i = 0; i < k; ++i)
        centroids[i] = data[rand() % n];

    vector<int> labels(n);

    double start = omp_get_wtime();

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assignment step
        for (int i = 0; i < n; ++i) {
            double min_dist = distance(data[i], centroids[0]);
            int label = 0;
            for (int j = 1; j < k; ++j) {
                double dist = distance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    label = j;
                }
            }
            labels[i] = label;
        }

        // Update step
        Mat new_centroids(k, Vec(d, 0.0));
        vector<int> counts(k, 0);

        for (int i = 0; i < n; ++i) {
            int label = labels[i];
            for (int j = 0; j < d; ++j)
                new_centroids[label][j] += data[i][j];
            counts[label]++;
        }

        for (int l = 0; l < k; ++l)
            if (counts[l] > 0)
                for (int j = 0; j < d; ++j)
                    centroids[l][j] = new_centroids[l][j] / counts[l];
    }

    double end = omp_get_wtime();
    time = end - start;

    cout << "\nSequential Final Centroids:\n";
    for (int i = 0; i < k; ++i) {
        cout << "Centroid " << i + 1 << ": ";
        for (int j = 0; j < d; ++j)
            cout << centroids[i][j] << " ";
        cout << endl;
    }
}

// Parallel K-means clustering
void kmeans_parallel(const Mat &data, int k, int max_iters, double &time) {
    int n = data.size();
    int d = data[0].size();

    // Randomly initialize centroids
    Mat centroids(k, Vec(d));
    srand(std::time(0));
    for (int i = 0; i < k; ++i)
        centroids[i] = data[rand() % n];

    vector<int> labels(n);

    double start = omp_get_wtime();

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assignment step
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double min_dist = distance(data[i], centroids[0]);
            int label = 0;
            for (int j = 1; j < k; ++j) {
                double dist = distance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    label = j;
                }
            }
            labels[i] = label;
        }

        // Update step
        Mat new_centroids(k, Vec(d, 0.0));
        vector<int> counts(k, 0);

        #pragma omp parallel
        {
            Mat local_centroids(k, Vec(d, 0.0));
            vector<int> local_counts(k, 0);

            #pragma omp for
            for (int i = 0; i < n; ++i) {
                int label = labels[i];
                for (int j = 0; j < d; ++j)
                    local_centroids[label][j] += data[i][j];
                local_counts[label]++;
            }

            #pragma omp critical
            {
                for (int l = 0; l < k; ++l) {
                    for (int j = 0; j < d; ++j)
                        new_centroids[l][j] += local_centroids[l][j];
                    counts[l] += local_counts[l];
                }
            }
        }

        for (int l = 0; l < k; ++l)
            if (counts[l] > 0)
                for (int j = 0; j < d; ++j)
                    centroids[l][j] = new_centroids[l][j] / counts[l];
    }

    double end = omp_get_wtime();
    time = end - start;

    cout << "\nParallel Final Centroids:\n";
    for (int i = 0; i < k; ++i) {
        cout << "Centroid " << i + 1 << ": ";
        for (int j = 0; j < d; ++j)
            cout << centroids[i][j] << " ";
        cout << endl;
    }
}

// Sequential PCA
void pca_sequential(const Mat &data, int n_components, double &time) {
    int n = data.size();
    int d = data[0].size();

    // Compute mean
    Vec mean(d, 0.0);
    double start = omp_get_wtime();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            mean[j] += data[i][j];
        }
    }

    for (int j = 0; j < d; ++j) {
        mean[j] /= n;
    }

    // Center data
    Mat centered = data;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            centered[i][j] -= mean[j];
        }
    }

    // Compute covariance matrix
    Mat cov(d, Vec(d, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < d; ++k) {
                cov[j][k] += centered[i][j] * centered[i][k];
            }
        }
    }

    for (int j = 0; j < d; ++j) {
        for (int k = 0; k < d; ++k) {
            cov[j][k] /= n;
        }
    }

    double end = omp_get_wtime();
    time = end - start;

    cout << "\nSequential Covariance Matrix (" << d << "x" << d << "):\n";
    for (int j = 0; j < d; ++j) {
        for (int k = 0; k < d; ++k) {
            cout << cov[j][k] << " ";
        }
        cout << "\n";
    }

    cout << "\n(For simplicity, eigen decomposition step is skipped here.)\n";
}

// Parallel PCA
void pca_parallel(const Mat &data, int n_components, double &time) {
    int n = data.size();
    int d = data[0].size();

    // Compute mean
    Vec mean(d, 0.0);
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        Vec local_mean(d, 0.0);
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                local_mean[j] += data[i][j];
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < d; ++j) {
                mean[j] += local_mean[j];
            }
        }
    }

    for (int j = 0; j < d; ++j) {
        mean[j] /= n;
    }

    // Center data
    Mat centered = data;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            centered[i][j] -= mean[j];
        }
    }

    // Compute covariance matrix
    Mat cov(d, Vec(d, 0.0));
    #pragma omp parallel
    {
        Mat local_cov(d, Vec(d, 0.0));
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    local_cov[j][k] += centered[i][j] * centered[i][k];
                }
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    cov[j][k] += local_cov[j][k];
                }
            }
        }
    }

    for (int j = 0; j < d; ++j) {
        for (int k = 0; k < d; ++k) {
            cov[j][k] /= n;
        }
    }

    double end = omp_get_wtime();
    time = end - start;

    cout << "\nParallel Covariance Matrix (" << d << "x" << d << "):\n";
    for (int j = 0; j < d; ++j) {
        for (int k = 0; k < d; ++k) {
            cout << cov[j][k] << " ";
        }
        cout << "\n";
    }

    cout << "\n(For simplicity, eigen decomposition step is skipped here.)\n";
}

int main() {
    int n_samples, n_features, k, max_iters;
    cout << "Enter number of samples: ";
    cin >> n_samples;
    cout << "Enter number of features: ";
    cin >> n_features;
    cout << "Enter number of clusters (k): ";
    cin >> k;
    cout << "Enter max iterations for K-means: ";
    cin >> max_iters;

    // Generate random data
    Mat data(n_samples, Vec(n_features));
    srand(time(0));
    for (int i = 0; i < n_samples; ++i)
        for (int j = 0; j < n_features; ++j)
            data[i][j] = rand() % 100;

    double time_kmeans_seq, time_kmeans_par, time_pca_seq, time_pca_par;

    // Run sequential K-means
    kmeans_sequential(data, k, max_iters, time_kmeans_seq);

    // Run parallel K-means
    kmeans_parallel(data, k, max_iters, time_kmeans_par);

    // Run sequential PCA
    pca_sequential(data, n_features, time_pca_seq);

    // Run parallel PCA
    pca_parallel(data, n_features, time_pca_par);

    // Print Speedup Table
    cout << "\nSpeedup Table:\n";
    cout << "---------------------------------\n";
    cout << "Algorithm\tSequential Time\tParallel Time\tSpeedup\n";
    cout << "K-Means\t\t" << time_kmeans_seq << "\t\t" << time_kmeans_par << "\t\t" 
         << time_kmeans_seq / time_kmeans_par << endl;
    cout << "PCA\t\t" << time_pca_seq << "\t\t" << time_pca_par << "\t\t" 
         << time_pca_seq / time_pca_par << endl;

    return 0;
}
// Exe: g++ -fopenmp ai_ml.cpp -o a 
//./a