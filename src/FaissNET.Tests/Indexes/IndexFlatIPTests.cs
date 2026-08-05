using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Cpu.Serializer;
using Xunit;

namespace Faiss.Tests.Indexes;

/// <summary>
/// Tests the core Add and Search functionality for dot product calculations.
/// </summary>
public class IndexFlatIPTests() : IndexTest<IndexFlatIP>(() => new IndexFlatIP(2))
{
    protected override int Dimensions => 2;

    [Fact]
    public void AddAndSearch_CalculatesCorrectInnerProduct()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);

        Assert.Equal(dimensions, index.Dimensions);
        Assert.Equal(0, index.TotalCount);

        float[] vectors = [2.0f, 3.0f];

        index.Add(1, vectors);
        Assert.Equal(1, index.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];

        index.Search(1, vectors, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(13.0f, distances[0]);
    }
    
    [Fact]
    public void AddAndSearch_CalculatesCorrectInnerProduct_Extensions()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);

        float[] vectors = [2.0f, 3.0f];

        index.Add(vectors);
        Assert.Equal(1, index.TotalCount);

        using var searchResultSpan = index.Search(vectors, 1);

        Assert.Equal(0, searchResultSpan.Labels[0]);
        Assert.Equal(13.0f, searchResultSpan.Distances[0]);
    }
    
    [Fact]
    public void AddAndRemove()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);

        index.Add([2.0f, 3.0f]);
        index.Add([2.0f, 3.1f]);
        index.Add([2.0f, 3.2f]);
        index.Add([2.0f, 3.3f]);
        
        Assert.Equal(4, index.TotalCount);

        index.RemoveIds(new IDSelectorBatch([1, 2]));
        Assert.Equal(2, index.TotalCount);
    }
    
    [Fact]
    public void AddAndReconstruct()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);
        float[] vector = [2.0f, 3.0f];
        index.Add(vector);

        float[] reconstructedVector = index.Reconstruct(0);
        Assert.Equal(vector, reconstructedVector);
    }
    
    [Fact]
    public void AddAndReconstructBatch()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);
        float[] vectors = [2.0f, 3.0f, 4.0f, 5.0f,  6.0f, 7.0f];
        index.Add(3, vectors);

        float[] reconstructedVector = index.ReconstructBatch(0, 3);
        Assert.Equal(vectors, reconstructedVector);
    }
    
    [Fact]
    public void Add_ComputeResidual()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);
        float[] vector = [2.0f, 3.0f];
        index.Add(vector);

        float[] residualVector = new float[dimensions];

        index.ComputeResidual(vector, residualVector, 0);
        Assert.Equal([0f, 0f], residualVector);
    }
    
    [Fact]
    public void Add_ComputeResidualBatch()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);
        float[] vectors = [2.0f, 3.0f, 4.0f, 5.0f,  6.0f, 7.0f];
        index.Add(3, vectors);

        float[] residualVectors = new float[dimensions * 3];

        index.ComputeResidualBatch(vectors, residualVectors, [0, 1, 2]);
        Assert.Equal([0f, 0f, 0f, 0f, 0f, 0f], residualVectors);
    }
    
    [Fact]
    public void Add_GetStandaloneCodeSize()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);
        float[] vector = [2.0f, 3.0f];
        index.Add(vector);

        long size = index.GetStandaloneCodeSize();
        long expectedSize = sizeof(float) * 2;
        Assert.Equal(expectedSize, size);
    }
    
    [Fact]
    public void Add_EncodeVector()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);
        float[] vector = [2.0f, 3.0f];

        long codeSize = index.GetStandaloneCodeSize();
        byte[] encodedVectors = new byte[codeSize];
        index.Encode(1, vector, encodedVectors);

        Assert.Equal([0, 0, 0, 64, 0, 0, 64, 64], encodedVectors);
    }
    
    [Fact]
    public void Add_DecodeVector()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);

        long codeSize = index.GetStandaloneCodeSize();
        byte[] encodedVectors = [0, 0, 0, 64, 0, 0, 64, 64];
        float[] decodedVectors = new float[codeSize / sizeof(float)];
        index.Decode(1, encodedVectors, decodedVectors);

        Assert.Equal([2f, 3f], decodedVectors);
    }
    
    [Fact]
    public void SerializationAndDeserialization()
    {
        using var index = new IndexFlatIP(2);
        index.Add([2.0f, 3.0f]);
        
        using var memoryStream = new MemoryStream();
        IndexSerializer.Write(index, memoryStream);
        memoryStream.Position = 0;
        
        using var deserializedIndex = IndexDeserializer.Read<IndexFlatIP>(memoryStream);
        Assert.Equal(2, deserializedIndex.Dimensions);
        Assert.Equal(1, deserializedIndex.TotalCount);
    }
    
    [Fact]
    public void GPUTransferrable()
    {
        IGpuClonableIndex _ = default(IndexFlatIP);
    }
}
