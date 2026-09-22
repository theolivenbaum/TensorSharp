// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace TensorSharp.Structured.Decisions
{
    /// <summary>
    /// CPython's <c>random.Random</c>, as far as djev uses it: seeding from a string and
    /// <c>randrange(n)</c>.
    ///
    /// djev fills the answer slots of its seed canvas with <c>random.Random(f"djev-canvas-v1:{seed}")
    /// .randrange(vocab)</c>. Reproducing the generator exactly means a request with the same seed starts
    /// from the same canvas here as it does there, so a difference in the answer is the engine's and not
    /// the noise's. The generator is MT19937, seeded the way CPython seeds from a <c>str</c> (version 2:
    /// the UTF-8 bytes followed by their SHA-512, read as one big-endian integer, fed to
    /// <c>init_by_array</c> as little-endian 32-bit words), and <c>randrange</c> is
    /// <c>_randbelow_with_getrandbits</c>: draw <c>n.bit_length()</c> bits and reject until below
    /// <c>n</c>.
    /// </summary>
    public sealed class PythonRandom
    {
        private const int N = 624;
        private const int M = 397;
        private const uint MatrixA = 0x9908b0dfu;
        private const uint UpperMask = 0x80000000u;
        private const uint LowerMask = 0x7fffffffu;

        private readonly uint[] _mt = new uint[N];
        private int _mti = N + 1;

        /// <summary>Seed exactly as <c>random.Random(text)</c> does.</summary>
        public PythonRandom(string seed)
        {
            ArgumentNullException.ThrowIfNull(seed);
            byte[] utf8 = Encoding.UTF8.GetBytes(seed);
            byte[] digest = SHA512.HashData(utf8);
            var material = new byte[utf8.Length + digest.Length];
            utf8.CopyTo(material, 0);
            digest.CopyTo(material, utf8.Length);
            InitByArray(Words(new BigInteger(material, isUnsigned: true, isBigEndian: true)));
        }

        /// <summary>Seed exactly as <c>random.Random(n)</c> does for an integer: its absolute value.</summary>
        public PythonRandom(BigInteger seed) => InitByArray(Words(BigInteger.Abs(seed)));

        /// <summary><c>getrandbits(k)</c> for <c>0 &lt; k &lt;= 32</c>.</summary>
        public uint GetRandBits(int k)
        {
            if (k is < 1 or > 32) throw new ArgumentOutOfRangeException(nameof(k));
            return NextUInt32() >> (32 - k);
        }

        /// <summary><c>randrange(n)</c> for <c>0 &lt; n &lt;= 2^32</c>.</summary>
        public long RandRange(long n)
        {
            if (n <= 0 || n > 1L << 32) throw new ArgumentOutOfRangeException(nameof(n));
            int k = 64 - BitOperations.LeadingZeroCount((ulong)n);
            long r = GetRandBits(k);
            while (r >= n) r = GetRandBits(k);
            return r;
        }

        /// <summary>The seed as 32-bit words, least significant first; zero is one zero word.</summary>
        private static uint[] Words(BigInteger value)
        {
            byte[] bytes = value.ToByteArray(isUnsigned: true, isBigEndian: false);
            int count = Math.Max(1, (bytes.Length + 3) / 4);
            var words = new uint[count];
            for (int i = 0; i < bytes.Length; i++) words[i / 4] |= (uint)bytes[i] << (8 * (i % 4));
            return words;
        }

        private void InitGenrand(uint s)
        {
            _mt[0] = s;
            for (_mti = 1; _mti < N; _mti++)
                _mt[_mti] = 1812433253u * (_mt[_mti - 1] ^ (_mt[_mti - 1] >> 30)) + (uint)_mti;
        }

        private void InitByArray(uint[] key)
        {
            InitGenrand(19650218u);
            int i = 1, j = 0;
            for (int k = Math.Max(N, key.Length); k > 0; k--)
            {
                _mt[i] = (_mt[i] ^ ((_mt[i - 1] ^ (_mt[i - 1] >> 30)) * 1664525u)) + key[j] + (uint)j;
                i++; j++;
                if (i >= N) { _mt[0] = _mt[N - 1]; i = 1; }
                if (j >= key.Length) j = 0;
            }
            for (int k = N - 1; k > 0; k--)
            {
                _mt[i] = (_mt[i] ^ ((_mt[i - 1] ^ (_mt[i - 1] >> 30)) * 1566083941u)) - (uint)i;
                i++;
                if (i >= N) { _mt[0] = _mt[N - 1]; i = 1; }
            }
            _mt[0] = 0x80000000u;
        }

        private uint NextUInt32()
        {
            uint y;
            if (_mti >= N)
            {
                int kk;
                for (kk = 0; kk < N - M; kk++)
                {
                    y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                    _mt[kk] = _mt[kk + M] ^ (y >> 1) ^ ((y & 1u) != 0 ? MatrixA : 0u);
                }
                for (; kk < N - 1; kk++)
                {
                    y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                    _mt[kk] = _mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1u) != 0 ? MatrixA : 0u);
                }
                y = (_mt[N - 1] & UpperMask) | (_mt[0] & LowerMask);
                _mt[N - 1] = _mt[M - 1] ^ (y >> 1) ^ ((y & 1u) != 0 ? MatrixA : 0u);
                _mti = 0;
            }

            y = _mt[_mti++];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            return y;
        }
    }
}
