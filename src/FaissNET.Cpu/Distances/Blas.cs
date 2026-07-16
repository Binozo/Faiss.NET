using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Distances;

public static class Blas
{
    /// <summary>
    /// Threshold value on nx * d above which we switch to BLAS to compute distances
    /// </summary>
    public static int DistanceComputeBlasThreshold
    {
        get => Native.faiss_get_distance_compute_blas_threshold();
        set => Native.faiss_set_distance_compute_blas_threshold(value);
    }
    
    /// <summary>
    /// Block sizes value for BLAS distance computations
    /// </summary>
    public static int DistanceComputeBlasBlockSizes
    {
        get => Native.faiss_get_distance_compute_blas_query_bs();
        set => Native.faiss_set_distance_compute_blas_query_bs(value);
    }
    
    /// <summary>
    /// Block sizes value for BLAS database distance computations
    /// </summary>
    public static int DistanceComputeBlasDatabaseBlockSizes
    {
        get => Native.faiss_get_distance_compute_blas_database_bs();
        set => Native.faiss_set_distance_compute_blas_database_bs(value);
    }
    
    /// <summary>
    /// Number of results we switch to a reservoir to collect results rather than a heap
    /// </summary>
    public static int DistanceComputeMinKReservoir
    {
        get => Native.faiss_get_distance_compute_blas_database_bs();
        set => Native.faiss_set_distance_compute_blas_database_bs(value);
    }
}