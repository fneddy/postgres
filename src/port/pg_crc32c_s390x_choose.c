/*-------------------------------------------------------------------------
 *
 * pg_crc32c_s390x_choose.c
 *	  Choose between S390X vectorized CRC-32C and software CRC-32C
 *	  implementation.
 *
 * On first call, checks if the CPU we're running on supports the
 * S390X_VX Extension. If it does, use the special instructions for
 * CRC-32C computation. Otherwise, fall back to the pure software
 * implementation (slicing-by-8).
 *
 * Copyright (c) 2025, International Business Machines (IBM)
 *
 * IDENTIFICATION
 *	  src/port/pg_crc32c_s390x_choose.c
 *
 *-------------------------------------------------------------------------
 */

#include "c.h"
#include "port/pg_crc32c.h"

#include <sys/auxv.h>

/*
 * This gets called on the first call. It replaces the function pointer
 * so that subsequent calls are routed directly to the chosen implementation.
 */
static pg_crc32c
pg_comp_crc32c_choose(pg_crc32c crc, const void *data, size_t len)
{
	/* default call sb8 */
	pg_comp_crc32c = pg_comp_crc32c_sb8;

	if (getauxval(AT_HWCAP) & HWCAP_S390_VX)
	{
		pg_comp_crc32c = pg_comp_crc32c_s390x;
	}

	return pg_comp_crc32c(crc, data, len);
}

pg_crc32c	(*pg_comp_crc32c) (pg_crc32c crc, const void *data, size_t len) = pg_comp_crc32c_choose;
