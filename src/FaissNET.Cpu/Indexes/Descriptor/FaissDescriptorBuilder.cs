namespace Faiss.Cpu.Indexes.Descriptor;

public sealed class FaissDescriptorBuilder
{
    private readonly List<string> _components = new();

    /// <summary>
    /// Starts a new index descriptor pipeline.
    /// </summary>
    public static FaissDescriptorBuilder Create() => new();

    public FaissDescriptorBuilder TransformWithPCA(int dimensions)
    {
        _components.Add($"PCA{dimensions}");
        return this;
    }

    public FaissDescriptorBuilder TransformWithOPQ(int bytes)
    {
        _components.Add($"OPQ{bytes}");
        return this;
    }

    public FaissDescriptorBuilder NormalizeL2()
    {
        _components.Add("L2Norm");
        return this;
    }

    public FaissDescriptorBuilder WithIVF(int centroids)
    {
        _components.Add($"IVF{centroids}");
        return this;
    }

    public FaissDescriptorBuilder WithHNSW(int neighbors)
    {
        _components.Add($"HNSW{neighbors}");
        return this;
    }

    public FaissDescriptorBuilder EncodeWithFlat()
    {
        _components.Add("Flat");
        return this;
    }

    public FaissDescriptorBuilder EncodeWithPQ(int bytes)
    {
        _components.Add($"PQ{bytes}");
        return this;
    }

    public FaissDescriptorBuilder EncodeWithSQ8()
    {
        _components.Add("SQ8");
        return this;
    }

    public string Build()
    {
        if (_components.Count == 0)
        {
            throw new InvalidOperationException("No components added");
        }
        
        return string.Join(",", _components);
    }
}