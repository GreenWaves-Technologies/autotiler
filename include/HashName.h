/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the Apache License.  See the LICENSE file for details.
 *
 */

#ifndef __HASH_NAME_H__
#define __HASH_NAME_H__

typedef struct ANameT NameT;
typedef struct ANameT {
        char *Name;
	size_t Len;
        NameT *Next;
} NameT;

typedef struct {
	unsigned int Size;
	NameT **Table;
} HashTableT;

NameT *HashInsertName(char *Name);
HashTableT *CreateHashTable(int Size);
void SetActiveHashTable(HashTableT *HashTable);
void FreeHashTable(HashTableT *HashTable);
#endif
