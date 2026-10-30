/* HMAC-SHA256 (RFC 2104 over the SHA-256 of FIPS 180-4) in portable C without the crypto of an OS: the lowercase hex of
   the digest of a message signed with a secret, both texts taken as their UTF-8 bytes. */
#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "strict.h"

typedef struct
{
	uint32_t state[8];
	uint64_t bytes;
	unsigned char block[64];
	size_t filled;
} sha256;

static const uint32_t rounds[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static uint32_t rotated(uint32_t value, int bits)
{
	return (value >> bits) | (value << (32 - bits));
}

static void compressed(sha256* hash, const unsigned char* block)
{
	uint32_t words[64], work[8];
	for (int index = 0; index < 16; index++)
		words[index] = (uint32_t)block[index * 4] << 24 | (uint32_t)block[index * 4 + 1] << 16 |
			(uint32_t)block[index * 4 + 2] << 8 | block[index * 4 + 3];
	for (int index = 16; index < 64; index++)
	{
		uint32_t early = words[index - 15], late = words[index - 2];
		words[index] = words[index - 16] + (rotated(early, 7) ^ rotated(early, 18) ^ (early >> 3)) + words[index - 7] +
			(rotated(late, 17) ^ rotated(late, 19) ^ (late >> 10));
	}
	memcpy(work, hash->state, sizeof(work));
	for (int index = 0; index < 64; index++)
	{
		uint32_t choice = (work[4] & work[5]) ^ (~work[4] & work[6]);
		uint32_t mixed = work[7] + (rotated(work[4], 6) ^ rotated(work[4], 11) ^ rotated(work[4], 25)) + choice +
			rounds[index] + words[index];
		uint32_t majority = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]);
		uint32_t spread = (rotated(work[0], 2) ^ rotated(work[0], 13) ^ rotated(work[0], 22)) + majority;
		memmove(work + 1, work, 7 * sizeof(uint32_t));
		work[4] += mixed;
		work[0] = mixed + spread;
	}
	for (int index = 0; index < 8; index++)
		hash->state[index] += work[index];
}

static void started(sha256* hash)
{
	static const uint32_t initial[8] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
	memcpy(hash->state, initial, sizeof(initial));
	hash->bytes = 0;
	hash->filled = 0;
}

static void added(sha256* hash, const unsigned char* data, size_t size)
{
	hash->bytes += size;
	while (size > 0)
	{
		size_t taken = 64 - hash->filled < size ? 64 - hash->filled : size;
		memcpy(hash->block + hash->filled, data, taken);
		hash->filled += taken;
		data += taken;
		size -= taken;
		if (hash->filled == 64)
		{
			compressed(hash, hash->block);
			hash->filled = 0;
		}
	}
}

static void finished(sha256* hash, unsigned char* digest)
{
	uint64_t bits = hash->bytes * 8;
	unsigned char padding[64] = { 0x80 }, length[8];
	added(hash, padding, (hash->filled < 56 ? 56 : 120) - hash->filled);
	for (int index = 0; index < 8; index++)
		length[index] = (unsigned char)(bits >> (56 - index * 8));
	added(hash, length, 8);
	for (int index = 0; index < 32; index++)
		digest[index] = (unsigned char)(hash->state[index / 4] >> (24 - index % 4 * 8));
}

static void digested(const unsigned char* first, size_t firstSize, const unsigned char* second, size_t secondSize,
	unsigned char* digest)
{
	sha256 hash;
	started(&hash);
	added(&hash, first, firstSize);
	added(&hash, second, secondSize);
	finished(&hash, digest);
}

STRICT_EXPORT double Hmac_Sha256(double self, double secret, double message)
{
	static _Thread_local char hex[65];
	const char* key = strict_text(secret);
	const char* text = strict_text(message);
	key = key ? key : "";
	text = text ? text : "";
	unsigned char block[64] = { 0 }, inner[64], outer[64], innerDigest[32], digest[32];
	size_t keySize = strlen(key);
	if (keySize > 64)
		digested((const unsigned char*)key, keySize, 0, 0, block);
	else
		memcpy(block, key, keySize);
	for (int index = 0; index < 64; index++)
	{
		inner[index] = block[index] ^ 0x36;
		outer[index] = block[index] ^ 0x5c;
	}
	digested(inner, 64, (const unsigned char*)text, strlen(text), innerDigest);
	digested(outer, 64, innerDigest, 32, digest);
	for (int index = 0; index < 32; index++)
		snprintf(hex + index * 2, 3, "%02x", digest[index]);
	return strict_text_result(hex);
}
