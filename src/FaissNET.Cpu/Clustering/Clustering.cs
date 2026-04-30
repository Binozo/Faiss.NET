using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Clustering;

public sealed class Clustering : IDisposable
{
    internal FaissClusteringHandle SafeHandle { get; }

    public Clustering(int dimensions, int k)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_Clustering_new(out IntPtr ptr, dimensions, k));

        SafeHandle = new FaissClusteringHandle(ptr);
    }

    public Clustering(int dimensions, int k, ClusteringOptions parameters)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_Clustering_new_with_params(out IntPtr ptr, dimensions, k, parameters.ToNative())
        );

        SafeHandle = new FaissClusteringHandle(ptr);
    }

    public int Niter => Native.faiss_Clustering_niter(SafeHandle);

    public int Nredo => Native.faiss_Clustering_nredo(SafeHandle);

    public bool Verbose => Native.faiss_Clustering_verbose(SafeHandle) != 0;

    public bool Spherical => Native.faiss_Clustering_spherical(SafeHandle) != 0;

    public bool IntCentroids => Native.faiss_Clustering_int_centroids(SafeHandle) != 0;

    public bool UpdateIndex => Native.faiss_Clustering_update_index(SafeHandle) != 0;

    public bool FrozenCentroids => Native.faiss_Clustering_frozen_centroids(SafeHandle) != 0;

    public int MinPointsPerCentroid => Native.faiss_Clustering_min_points_per_centroid(SafeHandle);

    public int MaxPointsPerCentroid => Native.faiss_Clustering_max_points_per_centroid(SafeHandle);

    public int Seed => Native.faiss_Clustering_seed(SafeHandle);

    public int DecodeBlockSize => (int)Native.faiss_Clustering_decode_block_size(SafeHandle);

    public int Dimensions => (int)Native.faiss_Clustering_d(SafeHandle);

    public int K => (int)Native.faiss_Clustering_k(SafeHandle);

    /// <summary>
    /// Train the clustering.
    /// </summary>
    public unsafe void Train(long n, ReadOnlySpan<float> vectors, IFlatIndex index)
    {
        fixed (float* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Clustering_train(SafeHandle, n, pVectors, index.Handle)
            );
        }
    }

    /// <summary>
    /// Get the trained centroids (size = k * d).
    /// </summary>
    public unsafe float[] GetCentroids()
    {
        Native.faiss_Clustering_centroids(SafeHandle, out IntPtr centroidsPtr, out UIntPtr size);
        int count = (int)size;
        float[] result = new float[count];
        fixed (float* pResult = result)
        {
            Buffer.MemoryCopy((void*)centroidsPtr, pResult, count * sizeof(float), count * sizeof(float));
        }

        return result;
    }

    public void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}