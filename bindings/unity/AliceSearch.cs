// ALICE-Search — Unity C# Bindings
// License: AGPL-3.0
// Author: Moroya Sakamoto
//
// 10 DllImport + IDisposable RAII wrapper

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Alice.Search
{
    internal static class Native
    {
#if UNITY_IOS && !UNITY_EDITOR
        private const string Lib = "__Internal";
#else
        private const string Lib = "alice_search";
#endif

        [DllImport(Lib)] public static extern IntPtr alice_index_build(
            byte[] text, uint textLen, uint sampleStep);
        [DllImport(Lib)] public static extern uint alice_index_count(
            IntPtr index, byte[] pattern, uint patternLen);
        [DllImport(Lib)] public static extern IntPtr alice_index_locate(
            IntPtr index, byte[] pattern, uint patternLen, out uint outLen);
        [DllImport(Lib)] public static extern void alice_index_locate_free(IntPtr ptr, uint len);
        [DllImport(Lib)] public static extern byte alice_index_contains(
            IntPtr index, byte[] pattern, uint patternLen);
        [DllImport(Lib)] public static extern uint alice_index_size_bytes(IntPtr index);
        [DllImport(Lib)] public static extern uint alice_index_text_len(IntPtr index);
        [DllImport(Lib)] public static extern double alice_index_compression_ratio(IntPtr index);
        [DllImport(Lib)] public static extern uint alice_index_sample_step(IntPtr index);
        [DllImport(Lib)] public static extern void alice_index_destroy(IntPtr index);
    }

    public sealed class AliceIndex : IDisposable
    {
        private IntPtr _ptr;

        public AliceIndex(byte[] text, uint sampleStep = 4)
        {
            _ptr = Native.alice_index_build(text, (uint)text.Length, sampleStep);
        }

        public AliceIndex(string text, uint sampleStep = 4)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            _ptr = Native.alice_index_build(bytes, (uint)bytes.Length, sampleStep);
        }

        public uint Count(byte[] pattern)
            => Native.alice_index_count(_ptr, pattern, (uint)pattern.Length);

        public uint Count(string pattern)
            => Count(Encoding.UTF8.GetBytes(pattern));

        public uint[] Locate(byte[] pattern)
        {
            var ptr = Native.alice_index_locate(_ptr, pattern, (uint)pattern.Length, out uint len);
            if (ptr == IntPtr.Zero || len == 0) return Array.Empty<uint>();
            var result = new uint[len];
            Marshal.Copy(ptr, (int[])(object)result, 0, (int)len);
            Native.alice_index_locate_free(ptr, len);
            return result;
        }

        public uint[] Locate(string pattern)
            => Locate(Encoding.UTF8.GetBytes(pattern));

        public bool Contains(byte[] pattern)
            => Native.alice_index_contains(_ptr, pattern, (uint)pattern.Length) != 0;

        public bool Contains(string pattern)
            => Contains(Encoding.UTF8.GetBytes(pattern));

        public uint SizeBytes => Native.alice_index_size_bytes(_ptr);
        public uint TextLen => Native.alice_index_text_len(_ptr);
        public double CompressionRatio => Native.alice_index_compression_ratio(_ptr);
        public uint SampleStep => Native.alice_index_sample_step(_ptr);

        public void Dispose()
        {
            if (_ptr != IntPtr.Zero)
            {
                Native.alice_index_destroy(_ptr);
                _ptr = IntPtr.Zero;
            }
        }

        ~AliceIndex() => Dispose();
    }
}
