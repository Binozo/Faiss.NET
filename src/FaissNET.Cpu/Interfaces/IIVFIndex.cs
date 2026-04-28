using Faiss.Cpu.Indexes.IVF;

namespace Faiss.Cpu.Interfaces;

public interface IIVFIndex
{
    /// <summary>Number of inverted lists (coarse clusters).</summary>
    int Nlist { get; }

    /// <summary>Number of lists to probe during search.</summary>
    int Nprobe { get; set; }

    /// <summary>
    /// Build or clear the direct map, enabling operations like
    /// <see cref="IndexIVFFlat.UpdateVectors"/> and reconstruction by ID.
    /// </summary>
    void MakeDirectMap(bool maintainDirectMap);
    
    /// <summary>
    /// Measure of list balance. 1.0 = perfectly balanced, higher = more imbalanced.
    /// </summary>
    double ImbalanceFactor { get; }
}