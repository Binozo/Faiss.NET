using System.Runtime.InteropServices;
using Faiss.Cpu.Selectors;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelector_is_member(FaissIDSelectorHandle sel, long id);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_IDSelector_free(IntPtr sel);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorBatch_new(out IntPtr pSel, UIntPtr n, long* indices);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorRange_new(out IntPtr pSel, long imin, long imax);
    
    [LibraryImport(LibraryName)]
    internal static partial long faiss_IDSelectorRange_imin(FaissIDSelectorHandle sel);
    
    [LibraryImport(LibraryName)]
    internal static partial long faiss_IDSelectorRange_imax(FaissIDSelectorHandle sel);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorBatch_nbits(FaissIDSelectorHandle sel);
    
    [LibraryImport(LibraryName)]
    internal static partial long faiss_IDSelectorBatch_mask(FaissIDSelectorHandle sel);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorBitmap_new(out IntPtr pSel, UIntPtr n, byte* bitmap);
    
    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IDSelectorBitmap_n(FaissIDSelectorHandle sel);
    
    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IDSelectorBitmap_bitmap(FaissIDSelectorHandle sel);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorNot_new(out IntPtr pSel, FaissIDSelectorHandle sel);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorAnd_new(out IntPtr pSel, FaissIDSelectorHandle lhsSel, FaissIDSelectorHandle rhsSel);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorOr_new(out IntPtr pSel, FaissIDSelectorHandle lhsSel, FaissIDSelectorHandle rhsSel);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IDSelectorXOr_new(out IntPtr pSel, FaissIDSelectorHandle lhsSel, FaissIDSelectorHandle rhsSel);
}