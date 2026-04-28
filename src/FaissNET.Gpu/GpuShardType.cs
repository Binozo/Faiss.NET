namespace Faiss.Gpu;

public enum GpuShardType
{
    /// <summary>
    /// Assigns vectors to shards using ID modulo distribution.
    /// </summary>
    IdModulo = 1,
    /// <summary>
    /// Splits vectors into contiguous ID ranges across shards.
    /// </summary>
    IdRange = 2,
    /// <summary>
    /// Assigns entire inverted lists (coarse quantizer buckets) to shards.
    /// </summary>
    InvertedList = 4,
}