using Faiss.Cpu.Selectors;
using Xunit;

namespace Faiss.Tests.Contract;

public class SelectorContractTests
{
    private const int Ids = 16;

    [Fact]
    public void Range_IncludesLowerBoundExcludesUpper()
    {
        using var selector = IDSelector.Range(4, 9);

        AssertMembership(selector, id => id is >= 4 and < 9);
    }

    [Fact]
    public void Range_Empty_MatchesNothing()
    {
        using var selector = IDSelector.Range(5, 5);

        AssertMembership(selector, _ => false);
    }

    [Fact]
    public void Batch_IncludesListedIdsOnly()
    {
        long[] allowed = [1, 3, 12];
        using var selector = IDSelector.Batch(allowed);

        AssertMembership(selector, allowed.Contains);
    }

    [Fact]
    public void Bitmap_IncludesIdsWithSetBit()
    {
        // Bit i of byte i/8 selects id i, least significant bit first.
        byte[] bitmap = [0b0000_1010, 0b1000_0001];
        using var selector = IDSelector.Bitmap(bitmap);

        AssertMembership(selector, id => (bitmap[id / 8] & (1 << (int)(id % 8))) != 0);
    }

    [Fact]
    public void Not_NegatesMembership()
    {
        using var selector = IDSelector.Batch(2, 7).Not();

        AssertMembership(selector, id => id is not (2 or 7));
    }

    [Fact]
    public void And_IntersectsOperands()
    {
        using var left = IDSelector.Range(0, 8);
        using var right = IDSelector.Batch(3, 9);
        using var selector = left.And(right);

        AssertMembership(selector, id => id == 3);
    }

    [Fact]
    public void Or_UnionsOperands()
    {
        using var left = IDSelector.Range(0, 3);
        using var right = IDSelector.Batch(10);
        using var selector = left.Or(right);

        AssertMembership(selector, id => id is < 3 or 10);
    }

    [Fact]
    public void XOr_TakesSymmetricDifference()
    {
        using var left = IDSelector.Range(0, 6);
        using var right = IDSelector.Range(4, 10);
        using var selector = left.XOr(right);

        AssertMembership(selector, id => id is < 4 or >= 6 and < 10);
    }

    [Fact]
    public void Composition_RootsBorrowedOperandAcrossCollection()
    {
        using var selector = IDSelector.Batch(1, 2).Not();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        AssertMembership(selector, id => id is not (1 or 2));
    }

    [Fact]
    public void Bitmap_ReportsByteLength()
    {
        using var selector = new IDSelectorBitmap([0xFF, 0x00, 0x0F]);

        Assert.Equal(3, selector.N);
    }

    [Fact]
    public void UseAfterDispose_Throws()
    {
        var selector = IDSelector.Range(0, 4);
        selector.Dispose();

        Assert.Throws<ObjectDisposedException>(() => selector.IsMember(1));
    }

    private static void AssertMembership(IDSelector selector, Func<long, bool> expected)
    {
        for (long id = 0; id < Ids; id++)
        {
            Assert.Equal(expected(id), selector.IsMember(id));
        }
    }
}
