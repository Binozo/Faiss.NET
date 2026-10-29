using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Exceptions;
using Faiss.Models;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Serializer;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class SerializeContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.SerializableFloat), MemberType = typeof(IndexCatalog))]
    public void StreamRoundtrip_RestoresSearch(IndexCase c)
    {
        Assert.NotNull(c.Deserialize);

        using var original = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);
        Sequential(original).Add(4, vectors);

        using var stream = new MemoryStream();
        IndexSerializer.Write(Assert.IsAssignableFrom<ISerializableFloatIndex>(original), stream);
        stream.Position = 0;

        using var restored = c.Deserialize(stream);

        Assert.Equal(original.Dimensions, restored.Dimensions);
        Assert.Equal(original.Metric, restored.Metric);
        Assert.Equal(4, restored.TotalCount);

        var query = vectors.AsSpan(0, c.Dimensions);
        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];
        restored.Search(1, query, k: 1, distances, labels);
        Assert.Equal(0, labels[0]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SerializableBinary), MemberType = typeof(IndexCatalog))]
    public void StreamRoundtrip_RestoresSearch_Binary(BinaryIndexCase c)
    {
        Assert.NotNull(c.Deserialize);

        using var original = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 1, count: 4, bitDimensions: c.Dimensions);
        Sequential(original).Add(4, vectors);

        using var stream = new MemoryStream();
        BinaryIndexSerializer.Write(Assert.IsAssignableFrom<ISerializableBinaryIndex>(original), stream);
        stream.Position = 0;

        using var restored = c.Deserialize(stream);

        Assert.Equal(original.Dimensions, restored.Dimensions);
        Assert.Equal(4, restored.TotalCount);

        var query = vectors.AsSpan(0, c.CodeSize);
        Span<int> distances = stackalloc int[1];
        Span<long> labels = stackalloc long[1];
        restored.Search(1, query, k: 1, distances, labels);
        Assert.Equal(0, labels[0]);
        Assert.Equal(0, distances[0]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SerializableFloat), MemberType = typeof(IndexCatalog))]
    public void FileRoundtrip_RestoresSearch(IndexCase c)
    {
        Assert.NotNull(c.DeserializeFile);

        var path = Path.Combine(Path.GetTempPath(), $"faissnet-{Guid.NewGuid():N}.index");
        try
        {
            using var original = c.CreateIndex();
            var vectors = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);
            Sequential(original).Add(4, vectors);

            IndexSerializer.Write(Assert.IsAssignableFrom<ISerializableFloatIndex>(original), path);
            Assert.True(File.Exists(path));

            using var restored = c.DeserializeFile(path);

            Assert.Equal(original.Dimensions, restored.Dimensions);
            Assert.Equal(original.Metric, restored.Metric);
            Assert.Equal(4, restored.TotalCount);

            AssertSameNeighbours(original, restored, vectors.AsSpan(0, c.Dimensions), k: 4);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SerializableBinary), MemberType = typeof(IndexCatalog))]
    public void FileRoundtrip_RestoresSearch_Binary(BinaryIndexCase c)
    {
        Assert.NotNull(c.DeserializeFile);

        var path = Path.Combine(Path.GetTempPath(), $"faissnet-{Guid.NewGuid():N}.index");
        try
        {
            using var original = c.CreateIndex();
            var vectors = Vectors.RandomBinary(seed: 1, count: 4, bitDimensions: c.Dimensions);
            Sequential(original).Add(4, vectors);

            BinaryIndexSerializer.Write(Assert.IsAssignableFrom<ISerializableBinaryIndex>(original), path);

            using var restored = c.DeserializeFile(path);

            Assert.Equal(original.Dimensions, restored.Dimensions);
            Assert.Equal(4, restored.TotalCount);

            var distances = new int[1];
            var labels = new long[1];
            restored.Search(1, vectors.AsSpan(0, c.CodeSize), k: 1, distances, labels);

            Assert.Equal(0, labels[0]);
            Assert.Equal(0, distances[0]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SerializableFloat), MemberType = typeof(IndexCatalog))]
    public void StreamRoundtrip_ShortReads_RestoreIdenticalIndex(IndexCase c)
    {
        Assert.NotNull(c.Deserialize);

        using var original = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions);
        Sequential(original).Add(8, vectors);

        using var buffer = new MemoryStream();
        IndexSerializer.Write(Assert.IsAssignableFrom<ISerializableFloatIndex>(original), buffer);

        // Check if the chunked stream reading works as Stream.Read could legally return fewer bytes than requested
        using var chunked = new ChunkedStream(buffer.ToArray(), chunk: 7);
        using var restored = c.Deserialize(chunked);

        Assert.Equal(original.TotalCount, restored.TotalCount);
        AssertSameNeighbours(original, restored, vectors.AsSpan(0, c.Dimensions), k: 4);
    }

    [Fact]
    public async Task StreamRoundtrip_TrainedIndex_PreservesTrainingAndResults()
    {
        const int dimensions = 8;
        const int count = 32;

        using var quantizer = new IndexFlatL2(dimensions);
        using var original = new IndexIVFFlat<IndexFlatL2>(quantizer, dimensions, nlist: 4);
        await original.TrainAsync(256, Vectors.Random(seed: 80, count: 256, dimensions: dimensions));

        var vectors = Vectors.Random(seed: 81, count: count, dimensions: dimensions);
        original.Add(count, vectors);
        original.Nprobe = original.Nlist;

        using var stream = new MemoryStream();
        IndexSerializer.Write(original, stream);
        stream.Position = 0;

        using var restored = IndexDeserializer.Read<IndexIVFFlat<IndexFlatL2>>(stream);

        // Coarse centroids and list assignments must survive, not just the vector payload.
        Assert.True(restored.IsTrained);
        Assert.Equal(original.Nlist, restored.Nlist);
        Assert.Equal(count, restored.TotalCount);

        restored.Nprobe = restored.Nlist;
        AssertSameNeighbours(original, restored, vectors.AsSpan(0, dimensions), k: 5);
    }

    [Fact]
    public void Read_MissingFile_Throws()
    {
        var path = Path.Combine(Path.GetTempPath(), $"faissnet-missing-{Guid.NewGuid():N}.index");

        Assert.Throws<FaissNativeException>(() => IndexDeserializer.Read<IndexFlatL2>(path));
    }

    [Fact]
    public void Read_TruncatedStream_Throws()
    {
        using var original = new IndexFlatL2(4);
        original.Add(4, Vectors.Random(seed: 82, count: 4, dimensions: 4));

        using var buffer = new MemoryStream();
        IndexSerializer.Write(original, buffer);

        var truncated = buffer.ToArray()[..(buffer.ToArray().Length / 2)];
        using var stream = new MemoryStream(truncated);

        Assert.ThrowsAny<FaissException>(() => IndexDeserializer.Read<IndexFlatL2>(stream));
    }

    [Theory]
    [InlineData(IoFlags.None)]
    [InlineData(IoFlags.ReadOnly)]
    [InlineData(IoFlags.MmapIfc)]
    public void Read_WithIoFlags_RestoresSearchableIndex(IoFlags flags)
    {
        const int dimensions = 8;
        const int count = 16;

        var path = Path.Combine(Path.GetTempPath(), $"faissnet-{Guid.NewGuid():N}.index");
        try
        {
            using var original = new IndexFlatL2(dimensions);
            var vectors = Vectors.Random(seed: 83, count: count, dimensions: dimensions);
            original.Add(count, vectors);

            IndexSerializer.Write(original, path);

            // MmapIfc gives a zero-copy view of flat codes; ReadOnly forbids mutation. Both must still search.
            using var restored = IndexDeserializer.Read<IndexFlatL2>(path, flags);

            Assert.Equal(count, restored.TotalCount);
            AssertSameNeighbours(original, restored, vectors.AsSpan(0, dimensions), k: 4);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SkipStorage_IsIgnoredByFlatIndexes()
    {
        const int dimensions = 8;
        const int count = 4;

        var path = Path.Combine(Path.GetTempPath(), $"faissnet-{Guid.NewGuid():N}.index");
        try
        {
            using var original = new IndexFlatL2(dimensions);
            var vectors = Vectors.Random(seed: 84, count: count, dimensions: dimensions);
            original.Add(count, vectors);
            IndexSerializer.Write(original, path);

            // Value 1 is IO_FLAG_SKIP_STORAGE, not memory mapping: the C header's FAISS_IO_FLAG_MMAP macro
            // is misleading. Read flags are implemented per index type, and IndexFlat ignores this one,
            // so the read succeeds and returns a fully populated index rather than a storage-less shell.
            using var restored = IndexDeserializer.Read<IndexFlatL2>(path, IoFlags.SkipStorage);
            Assert.Equal(count, restored.TotalCount);
            AssertSameNeighbours(original, restored, vectors.AsSpan(0, dimensions), k: 2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void AssertSameNeighbours(FloatIndex expected, FloatIndex actual, ReadOnlySpan<float> query, int k)
    {
        var expectedDistances = new float[k];
        var expectedLabels = new long[k];
        expected.Search(1, query, k, expectedDistances, expectedLabels);

        var actualDistances = new float[k];
        var actualLabels = new long[k];
        actual.Search(1, query, k, actualDistances, actualLabels);

        Assert.Equal(expectedLabels, actualLabels);
        FloatAssert.Equal(expectedDistances, actualDistances);
    }

    /// <summary>A stream that never returns more than <c>chunk</c> bytes per read, like a network or compression stream.</summary>
    private sealed class ChunkedStream(byte[] data, int chunk) : MemoryStream(data, writable: false)
    {
        public override int Read(Span<byte> buffer) => base.Read(buffer[..Math.Min(chunk, buffer.Length)]);

        public override int Read(byte[] buffer, int offset, int count) => base.Read(buffer, offset, Math.Min(chunk, count));
    }

    private static IIDSequentialFloatIndex Sequential(FloatIndex index) => Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index);

    private static IIDSequentialBinaryIndex Sequential(BinaryIndex index) => Assert.IsAssignableFrom<IIDSequentialBinaryIndex>(index);
}
