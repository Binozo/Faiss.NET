namespace Faiss.Models;

/// <summary>
/// Distance metrics supported by Faiss.
/// </summary>
public enum MetricType
{
    /// <summary>Maximum inner product search. The higher, the better</summary>
    InnerProduct = 0,
    
    /// <summary>Squared L2 search.</summary>
    L2 = 1,
    
    /// <summary>L1 (aka cityblock) distance.</summary>
    L1 = 2,
    
    /// <summary>Infinity distance.</summary>
    Linf = 3,
    
    /// <summary>L_p distance, p is given by metric_arg.</summary>
    Lp = 4,

    /// <summary>Canberra distance.</summary>
    Canberra = 20,

    /// <summary>Bray-Curtis distance.</summary>
    BrayCurtis = 21,

    /// <summary>Jensen-Shannon divergence.</summary>
    JensenShannon = 22,
    
    /// <summary>sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i)) where a_i, b_i > 0. The higher, the better.</summary>
    Jaccard = 23,
    
    /// <summary>Squared Euclidean distance, ignoring NaNs</summary>
    NaNEuclidean = 24,
    
    /// <summary>Gower's distance</summary>
    Gower = 25
}