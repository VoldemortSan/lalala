#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
using namespace std;

// Bubble Sort Sequential
void bubbleSortSeq(int arr[], int n) {
    for(int i=0; i<n-1; i++) {
        for(int j=0; j<n-i-1; j++) {
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Bubble Sort Parallel
void bubbleSortPar(int arr[], int n) {
    for(int i=0; i<n; i++) {
        #pragma omp parallel for
        for(int j=i%2; j<n-1; j+=2) {
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Merge function
void merge(int arr[], int l, int m, int r) {
    int n1=m-l+1, n2=r-m;
    int* L = new int[n1];
    int* R = new int[n2];

    for(int i=0;i<n1;i++) L[i]=arr[l+i];
    for(int i=0;i<n2;i++) R[i]=arr[m+1+i];

    int i=0, j=0, k=l;
    while(i<n1 && j<n2)
        arr[k++] = (L[i]<R[j]) ? L[i++] : R[j++];

    while(i<n1) arr[k++] = L[i++];
    while(j<n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

// Merge Sort Sequential
void mergeSortSeq(int arr[], int l, int r) {
    if(l<r) {
        int m=(l+r)/2;
        mergeSortSeq(arr, l, m);
        mergeSortSeq(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

// Merge Sort Parallel
void mergeSortPar(int arr[], int l, int r) {
    if(l<r) {
        int m=(l+r)/2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortPar(arr, l, m);

            #pragma omp section
            mergeSortPar(arr, m+1, r);
        }
        merge(arr, l, m, r);
    }
}

// Print array
void printArray(int arr[], int n) {
    for(int i=0; i<n; i++) cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    int* arr1 = new int[n];
    int* arr2 = new int[n];
    int* arr3 = new int[n];
    int* arr4 = new int[n];

    srand(time(0));
    for(int i=0;i<n;i++) {
        int val = rand()%100;
        arr1[i] = arr2[i] = arr3[i] = arr4[i] = val;
    }

    cout << "\nOriginal Array:\n";
    printArray(arr1, n);

    double start, end;

    // Bubble Sort
    start = omp_get_wtime();
    bubbleSortSeq(arr1, n);
    end = omp_get_wtime();
    double bubble_seq_time = end - start;

    cout << "\nSequential Bubble Sort Result:\n";
    printArray(arr1, n);

    start = omp_get_wtime();
    bubbleSortPar(arr2, n);
    end = omp_get_wtime();
    double bubble_par_time = end - start;

    cout << "\nParallel Bubble Sort Result:\n";
    printArray(arr2, n);

    double bubble_speedup = bubble_seq_time / bubble_par_time;

    // Merge Sort
    start = omp_get_wtime();
    mergeSortSeq(arr3, 0, n-1);
    end = omp_get_wtime();
    double merge_seq_time = end - start;

    cout << "\nSequential Merge Sort Result:\n";
    printArray(arr3, n);

    start = omp_get_wtime();
    mergeSortPar(arr4, 0, n-1);
    end = omp_get_wtime();
    double merge_par_time = end - start;

    cout << "\nParallel Merge Sort Result:\n";
    printArray(arr4, n);

    double merge_speedup = merge_seq_time / merge_par_time;

    // Final Comparison Table
    cout << "\n-----------------------------------------\n";
    cout << "Algorithm           Seq Time (s)  Par Time (s)  Speedup\n";
    cout << "-----------------------------------------\n";
    cout << "Bubble Sort         " << bubble_seq_time << "      " 
         << bubble_par_time << "      " << bubble_speedup << "x\n";
    cout << "Merge Sort          " << merge_seq_time << "      " 
         << merge_par_time << "      " << merge_speedup << "x\n";
    cout << "-----------------------------------------\n";

    delete[] arr1;
    delete[] arr2;
    delete[] arr3;
    delete[] arr4;

    return 0;
}
// Exe: g++ -fopenmp sort.cpp -o a 
//./a