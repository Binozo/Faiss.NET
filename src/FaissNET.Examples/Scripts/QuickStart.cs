using Faiss.Cpu.Distances;
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Search;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Cpu.Serializer;
using Faiss.Models;


using var oldIndex = new IndexBinaryFlat(4);

oldIndex.Add([16, 32, 64, 128]);
BinaryIndexSerializer.Write(oldIndex, "index.faiss");

using var newIndex = BinaryIndexDeserializer.Read<IndexBinaryFlat>("index.faiss");

var index = new IndexHNSW(dimensions: 4, metricType: MetricType.InnerProduct); // We don't need using here because takeOwnership below
using var mappedIndex = new IndexIDMap<IndexHNSW>(index, takeOwnership: true); // disposes index as soon as mappedIndex is disposed

mappedIndex.Add([new[] { 1f, 2f, 3f, 4f }, new[] { 2f, 3f, 4f, 1f }, new[] { 3f, 4f, 1f, 2f }, new[] { 4f, 1f, 2f, 3f }], [4, 3, 2, 1]);

var queryNeighborsCount = 2; // K
var searchResult = mappedIndex.SearchWithParams(new [] {1f, 1f, 2f, 3f}, queryNeighborsCount, new SearchParameters(new IDSelectorRange(2, 4)));
QueryResults queryResult = searchResult.GetQueryResults(0);

for (int i = 0; i < searchResult.K; i++)
{
    float distance = queryResult.Distances[i];
    long label     = queryResult.Labels[i];

    Console.WriteLine($"Rank {i}: label={label}, dist={distance}");
}