using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [StructLayout(LayoutKind.Sequential)]
    public struct ClusteringParameters
    {
        public int Niter;
        public int Nredo;
        public int Verbose;
        public int Spherical;
        public int IntCentroids;
        public int UpdateIndex;
        public int FrozenCentroids;
        public int MinPointsPerCentroid;
        public int MaxPointsPerCentroid;
        public int Seed;
        public UIntPtr DecodeBlockSize;
    }
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_ClusteringParameters_init(ref ClusteringParameters parameters);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_new(out IntPtr pClustering, int d, int k);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_new_with_params(out IntPtr pClustering, int d, int k, in ClusteringParameters cp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_Clustering_free(IntPtr clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_train(FaissClusteringHandle clustering, long n, float* x, FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_Clustering_centroids(FaissClusteringHandle clustering, out IntPtr centroids, out UIntPtr size);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_Clustering_iteration_stats(IntPtr clustering, out IntPtr iterationStats, out UIntPtr size);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_niter(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_nredo(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_verbose(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_spherical(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_int_centroids(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_update_index(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_frozen_centroids(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_min_points_per_centroid(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_max_points_per_centroid(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Clustering_seed(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_Clustering_decode_block_size(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_Clustering_d(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_Clustering_k(FaissClusteringHandle clustering);

    [LibraryImport(LibraryName)]
    internal static partial float faiss_ClusteringIterationStats_obj(IntPtr stats);

    [LibraryImport(LibraryName)]
    internal static partial double faiss_ClusteringIterationStats_time(IntPtr stats);

    [LibraryImport(LibraryName)]
    internal static partial double faiss_ClusteringIterationStats_time_search(IntPtr stats);

    [LibraryImport(LibraryName)]
    internal static partial double faiss_ClusteringIterationStats_imbalance_factor(IntPtr stats);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_ClusteringIterationStats_nsplit(IntPtr stats);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_kmeans_clustering(UIntPtr d, UIntPtr n, UIntPtr k, float* x, float* centroids, out float qError);
}