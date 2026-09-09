namespace Faiss.Models;

/// <summary>
/// Flags accepted by the index read functions.
/// </summary>
[Flags]
public enum IoFlags
{
    None = 0,

    /// <summary>Do not load the nested storage of graph indexes (HNSW). <c>IO_FLAG_SKIP_STORAGE</c>.</summary>
    SkipStorage = 1,

    /// <summary>Open the index read-only. <c>IO_FLAG_READ_ONLY</c>.</summary>
    ReadOnly = 2,

    /// <summary>Strip the directory from an on-disk inverted-list filename and resolve it next to the index. <c>IO_FLAG_ONDISK_SAME_DIR</c>.</summary>
    OnDiskSameDir = 4,

    /// <summary>Load inverted-list sizes only, leaving the IVF data on disk. <c>IO_FLAG_SKIP_IVF_DATA</c>.</summary>
    SkipIvfData = 8,

    /// <summary>Skip building the precomputed table after loading. <c>IO_FLAG_SKIP_PRECOMPUTE_TABLE</c>.</summary>
    SkipPrecomputeTable = 16,

    /// <summary>Skip the SDC table for PQ-based indexes. <c>IO_FLAG_PQ_SKIP_SDC_TABLE</c>.</summary>
    PqSkipSdcTable = 32,

    /// <summary>
    /// Memory-map codes of <c>IndexFlatCodes</c>-derived indexes and HNSW. <c>IO_FLAG_MMAP_IFC</c>.
    /// Gives a zero-copy view of flat vector storage.
    /// </summary>
    MmapIfc = 512,

    /// <summary>
    /// Memory-map inverted lists, loading an <c>ArrayInvertedLists</c> as an <c>OnDiskInvertedLists</c>.
    /// <c>IO_FLAG_MMAP</c>, which is <see cref="SkipIvfData"/> combined with a magic marker.
    /// </summary>
    Mmap = SkipIvfData | 0x646f0000,
}
