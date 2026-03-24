namespace Faiss.Cpu.Search;

public readonly record struct SearchResult
{
    public long Label { get; init; }
    public float Distance { get; init; }

    public SearchResult(long label, float distance)
    {
        Label = label;
        Distance = distance;
    }
}