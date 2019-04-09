/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the Apache License.  See the LICENSE file for details.
 *
 */


#include <stdio.h>
#include "CNN_BasicKernels.h"

// #define PROFILING_BK
#define VOL volatile
#define ASM_OPT volatile

#define Min(a, b)		(((a)<(b))?(a):(b))
#define Minu(a, b)		(( ((unsigned int)a)<((unsigned int)b) )?((unsigned int)a):((unsigned int)b) )
#define Max(a, b)		(((a)>(b))?(a):(b))
#define Maxu(a, b)		(( ((unsigned int)a)>((unsigned int)b) )?((unsigned int)a):((unsigned int)b) )

#ifdef OLD_RUNTIME
#define wait_synch_barrier()          eu_bar_trig_wait_clr(eu_bar_addr(0))
#else
#define wait_synch_barrier()          rt_team_barrier()
#endif

static int CoreCountDynamic = 1;
static int ActiveCore = gap8_ncore();

static rt_perf_t perf[8];

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
        unsigned int NCore;
        unsigned int Log2Core;
        unsigned int Chunk;

        if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap8_ncore();
        Log2Core = gap8_fl1(NCore);
        Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
        return Chunk;
}


static inline unsigned int __attribute__((always_inline)) ChunkSizeEven(unsigned int X) {

	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap8_ncore();
	Log2Core = gap8_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return (Chunk+(Chunk&0x1));
}


/* Padding Functions */
void KerExpandTileInitH_fp(ArgExpandTile_fp_T *Arg) {

	short int * __restrict__ In = Arg->In;
	unsigned int Win = Arg->Win;
	unsigned int Hin = Arg->Hin;
	unsigned int Pad = Arg->Pad;

	unsigned int CoreId = gap8_coreid();
	unsigned int N = (Hin-Pad);
	unsigned int Chunk = ChunkSize(N);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, N);

	In += Win*Pad;
	for(unsigned int i=First; i<Last; i++) {
		for(unsigned j=0; j<Pad; j++) {
			In[i*Win+j] = 0; 
			In[(i+1)*Win-j-1] = 0;
		}
	}
	rt_team_barrier();
}

void KerPadFirstTileH_fp(ArgExpandTile_fp_T *Arg) {

	short int * __restrict__ In = Arg->In;
	unsigned int Win = Arg->Win;
	unsigned int Pad = Arg->Pad;
	
	unsigned int CoreId = gap8_coreid();
	unsigned int N = (Win*Pad)/4;
	unsigned int Chunk = ChunkSize(N);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, N);

	int *LineIn = (int *) (In);
	for(unsigned int i=First; i<Last; i++) {
		LineIn[2*i] = 0; 
		LineIn[2*i+1] = 0;
	}
	for(unsigned int i=4*N; i<(Win*Pad); i++) In[i] = 0;
	rt_team_barrier();
}

void KerExpandTileH_fp(ArgExpandTile_fp_T *Arg) {

	short int * __restrict__ In = Arg->In;
	unsigned int Win = Arg->Win;
	unsigned int Hin = Arg->Hin;
	short int * __restrict__ Out = Arg->Out;
	unsigned int Pad = Arg->Pad;
	unsigned char TileIndex = Arg->TileIndex;
	unsigned char NTile = Arg->NTile;
	unsigned int FirstTile = (TileIndex==0);
	unsigned int LastTile = (TileIndex==(NTile-1));

	unsigned int tmp_Hin = Hin;
	Hin = Hin - Pad*(FirstTile+LastTile);

	unsigned int CoreId = gap8_coreid();
	unsigned int ChunkRow = ChunkSize(Hin);
	unsigned int FirstRow = ChunkRow*CoreId;
	unsigned int LastRow = Min(FirstRow+ChunkRow, Hin);
	unsigned int Wout = Win + 2*Pad;
	unsigned int Row, Col;

	// if(CoreId==0) printf("ExpandInit Core%d [%d-%d] %d %d %d/%d\n", CoreId, FirstRow, LastRow, Win, Hin, TileIndex, NTile);

	Out += FirstTile*(Wout*Pad);
	for (Row=FirstRow; Row<LastRow; Row++) {
		int *LineOut = (int *) (&Out[Row*Wout + Pad]);
		int *LineIn = (int *) (&In[Row*Win]);
		for (Col=0; Col<(Win/4); Col++) {
			int V0 = LineIn[2*Col], V1 = LineIn[2*Col+1];
			LineOut[2*Col] = V0;
			LineOut[2*Col+1] = V1;
		}
		for (Col=(Win/4)*4; Col<Win; Col++) Out[Row*Wout+Pad+Col] = In[Row*Win + Col];
	}
	if (LastTile) {
		unsigned int N = (Wout*Pad)/2;
		unsigned int Chunk = ChunkSize(N);
		unsigned int First = Chunk*CoreId;
		unsigned int Last = Min(First+Chunk, N);

		int *LineOut = (int *) (Out+Hin*Wout);
		for (unsigned int i=First; i<Last; i++) LineOut[i] = 0;
		Out[(Hin+Pad)*Wout-1] = 0;
	}
	rt_team_barrier();
}


//==============================================================================

static void __attribute__ ((noinline)) BorderConv2x3Stride1_V_fp(
        short int * __restrict__ In,
        int W,
        int Wo, int Ho,
        unsigned int Norm,
        short int * __restrict__ Out,
	const v2s C0,
	const v2s C1,
	const v2s C2,
	int Top,
	int Bottom
	)

{
#ifdef ASM_OPT
	v2s V0, V1, V2;
	v2s * __restrict__ Pt = (v2s *) In;
	short int * __restrict__ PtO = Out;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	int Off0 = 2*W;
	int Off1 = 2*Wo;

	if (Top) V0 = (v2s){0,0}; else asm VOL("p.lw %0,%2(%1!)" : "=r" (V0), "+r" (Pt) : "r" (Off0) : "memory");
	asm VOL("p.lw %0,%2(%1!)" : "=r" (V1), "+r" (Pt) : "r" (Off0) : "memory");
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		asm VOL("p.lw %0,%2(%1!)" : "=r" (V2), "+r" (Pt) : "r" (Off0) : "memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "r" (Off1) : "memory");
		V0 = V1; V1 = V2;
	}
	if (Bottom) {
		int Acc = *PtO<<Norm;
		Pt = (v2s *) (((short int *) Pt)-2*W);
		asm VOL("p.lw %0,%2(%1!)" : "=r" (V0), "+r"  (Pt) : "r" (Off0) : "memory");
		asm VOL("p.lw %0,%2(%1!)" : "=r" (V1), "+r"  (Pt) : "r" (Off0) : "memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		*PtO = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	}
#else
	v2s V0, V1, V2;
	v2s * __restrict__ Pt = (v2s *) In;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	short int * __restrict__ PtO = Out;
	if (Top) {
		V0 = (v2s){0,0}; V1 = *((v2s *)(In));
	} else {
		V0 = *((v2s *)(In)); V1 = *((v2s *)(In+W));
	}
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		V2 = *((v2s *)(In+(i+2-Top)*W));
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15); PtO+=Wo;
		V0 = V1; V1 = V2;
	}
	if (Bottom) {
		int Acc = *PtO<<Norm;
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	}
#endif
}

static void __attribute__ ((noinline)) BorderConv2x3Stride2_V_fp(
        short int * __restrict__ In,
        int W,
        int Wo, int Ho,
        unsigned int Norm,
        short int * __restrict__ Out,
	const v2s C0,
	const v2s C1,
	const v2s C2,
	int Top,
	int Bottom
	)

{
#ifdef ASM_OPT
	v2s V0, V1, V2;
	v2s * __restrict__ Pt = (v2s *) In;
	short int * __restrict__ PtO = Out;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	int Off0 = 2*W;
	int Off1 = 2*Wo;

	if (Top) V0 = (v2s){0,0}; else asm VOL("p.lw %0,%2(%1!)" : "=r" (V0), "+r" (Pt) : "r" (Off0) : "memory");
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		asm VOL("p.lw %0,%2(%1!)" : "=r" (V1), "+r" (Pt) : "r" (Off0) : "memory");
		asm VOL("p.lw %0,%2(%1!)" : "=r" (V2), "+r" (Pt) : "r" (Off0) : "memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "r" (Off1) : "memory");
		V0 = V2;
	}
	if (Bottom) {
		int Acc = *PtO<<Norm;
		Pt = (v2s *) (((short int *) Pt)-1*W);
		asm VOL("p.lw %0,%2(%1!)" : "=r" (V0), "+r"  (Pt) : "r" (Off0) : "memory");
		asm VOL("p.lw %0,%2(%1!)" : "=r" (V1), "+r"  (Pt) : "r" (Off0) : "memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		*PtO = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	}
#else
	v2s V0, V1, V2;
	v2s * __restrict__ Pt = (v2s *) In;
	short int * __restrict__ PtO = Out;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	if (Top) {
		V0 = (v2s){0,0};
	} else {
		V0 = *((v2s *)(In));
	}
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		V1 = *((v2s *)(In+(2*i+1-Top)*W));
		V2 = *((v2s *)(In+(2*i+2-Top)*W));
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15); PtO+=Wo;
		V0 = V2;
	}
	if (Bottom) {
		unsigned int i=Ho-(Bottom);
		int Acc = *PtO<<Norm;
		V1 = *((v2s *)(In+(2*i+1-Top)*W));
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	}
#endif
}

static void __attribute__ ((noinline)) BorderConv3x2Stride1_H_fp(
        short int * __restrict__ In,
        int W, 
        int Wo,
        unsigned int Norm,
        short int * __restrict__ Out,
	const v2s C0,
	const v2s C1,
	const v2s C2
	)

{
#ifdef ASM_OPT
	v2s X, Y, V0, V1, V2;
	v2s *Pt = (v2s *) In;
	short int * __restrict__ PtO = Out;
	int Off0 = 2*W;

	asm VOL("p.lw %0,%2(%1)" : "=r" (Y) : "r" (Pt), "r" (Off0) : "memory"); asm VOL("p.lw %0,%2(%1!)" : "=r" (X), "+r" (Pt) : "i" (4) : "memory");
	V0 = __builtin_shuffle(X,Y,(v2s){0,2}); V1 = __builtin_shuffle(X,Y,(v2s){1,3});
	for (unsigned int i=0; i<Wo; i++) {
		int x0, x1, Acc = *PtO<<Norm;
		asm VOL("p.lh %0,%2(%1)"  : "=r" (x1) : "r" (Pt), "r" (Off0) : "memory"); asm VOL("p.lh %0,%2(%1!)" : "=r" (x0), "+r" (Pt) : "i"   (2) : "memory");
		V2 = gap8_pack2(x0, x1);
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		Acc =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "i" (2) : "memory");
		V0 = V1; V1 = V2;
	}
#else
	v2s X, Y, V0, V1, V2;

	X = *((v2s *) &In[0]); Y = *((v2s *) &In[W]);
	V1 = __builtin_shuffle(X,Y,(v2s){0,2}); V2 = __builtin_shuffle(X,Y,(v2s){1,3});
	for (unsigned int i=0; i<Wo; i++) {
		int x0, x1, Acc = Out[i]<<Norm;
		V0 = V1; V1 = V2;
		x0 = In[i+2]; x1 = In[i+W+2]; V2 = gap8_pack2(x0, x1);
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		Out[i] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	}
#endif
}

static void __attribute__ ((noinline)) BorderConv3x2Stride2_H_fp(
        short int * __restrict__ In,
        int W,
        int Wo,
        unsigned int Norm,
        short int * __restrict__ Out,
	const v2s C0,
	const v2s C1,
	const v2s C2
	)

{
#ifdef ASM_OPT
	v2s X, Y, V0, V1, V2;
	v2s *Pt = (v2s *) In;
	short int * __restrict__ PtO = Out;
	int Off0 = 2*W;
	int x0,x1;

	asm VOL("p.lh %0,%2(%1)"  : "=r" (x1) : "r" (Pt), "r" (Off0) : "memory"); asm VOL("p.lh %0,%2(%1!)" : "=r" (x0), "+r" (Pt) : "i"   (2) : "memory"); V0 = gap8_pack2(x0, x1);
	for (unsigned int i=0; i<Wo; i++) {
		int Acc = *PtO<<Norm;
		asm("p.lw %0,%2(%1)" : "=r" (Y) : "r" (Pt), "r" (Off0) : ); asm("p.lw %0,%2(%1!)" : "=r" (X), "+r" (Pt) : "i" (4) : );
		V1 = __builtin_shuffle(X,Y,(v2s){0,2}); V2 = __builtin_shuffle(X,Y,(v2s){1,3});
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		Acc =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "i" (2) : "memory");
		V0 = V2;
	}
#else
	v2s X, Y, V0, V1, V2;

	V0 = gap8_pack2(In[0], In[W]);
	for (unsigned int i=0; i<Wo; i++) {
		int Acc = Out[i]<<Norm;
		X = *((v2s *) &In[2*i+1]); Y = *((v2s *) &In[2*i+1+W]); V1 = __builtin_shuffle(X,Y,(v2s){0,2}); V2 = __builtin_shuffle(X,Y,(v2s){1,3});
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc);
		Out[i] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		V0 = V2;
	}
#endif
}

static void __attribute__ ((noinline)) BorderConv5x4Stride1_H_fp(
        short int * __restrict__ In,
        int W,
        int Wo,
        unsigned int Norm,
        short int * __restrict__ Out,
        v2s * __restrict__ Filter
	)

{
#ifdef ASM_OPT
	const v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	          C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];
	v2s X, Y, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;
	v2s *Pt = (v2s *) In;

	int Off0 = 2*W, Off1 = 4*W-4;

 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"i"(4):"memory");    V0 = __builtin_shuffle(X,Y,(v2s){0,2}); V1 = __builtin_shuffle(X,Y,(v2s){1,3});
 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"r"(Off1):"memory"); V2 = __builtin_shuffle(X,Y,(v2s){0,2}); V3 = __builtin_shuffle(X,Y,(v2s){1,3});
 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"i"(4):"memory");    V5 = __builtin_shuffle(X,Y,(v2s){0,2}); V6 = __builtin_shuffle(X,Y,(v2s){1,3});
	Off1 = -Off1;
 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"r"(Off1):"memory"); V7 = __builtin_shuffle(X,Y,(v2s){0,2}); V8 = __builtin_shuffle(X,Y,(v2s){1,3});
	Off1=-6*W+2;

	for (unsigned int i=0; i<Wo; i++) {
		int x0, x1, x2, x3, Acc = Out[i]<<Norm;
		asm VOL("p.lh %0,%2(%1!)":"=r"(x0),"+r"(Pt):"r"(Off0):"memory"); asm VOL("p.lh %0,%2(%1!)":"=r"(x1),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lh %0,%2(%1!)":"=r"(x2),"+r"(Pt):"r"(Off0):"memory"); asm VOL("p.lh %0,%2(%1!)":"=r"(x3),"+r"(Pt):"r"(Off1):"memory");
		V4 = gap8_pack2(x0,x1); V9 = gap8_pack2(x2,x3);
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc); Acc = gap8_sumdotp2(V4, C4, Acc);
		Acc = gap8_sumdotp2(V5, C5, Acc); Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc); Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		V0=V1; V1=V2; V2=V3; V3=V4; V5=V6; V6=V7; V7=V8; V8=V9;
	}
#else
	const v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	          C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];

	v2s X, Y, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;

	X = *((v2s *) &In[0*W+0]); Y = *((v2s *) &In[1*W+0]); V0 = __builtin_shuffle(X,Y,(v2s){0,2}); V1 = __builtin_shuffle(X,Y,(v2s){1,3});
	X = *((v2s *) &In[0*W+2]); Y = *((v2s *) &In[1*W+2]); V2 = __builtin_shuffle(X,Y,(v2s){0,2}); V3 = __builtin_shuffle(X,Y,(v2s){1,3});
	X = *((v2s *) &In[2*W+0]); Y = *((v2s *) &In[3*W+0]); V5 = __builtin_shuffle(X,Y,(v2s){0,2}); V6 = __builtin_shuffle(X,Y,(v2s){1,3});
	X = *((v2s *) &In[2*W+2]); Y = *((v2s *) &In[3*W+2]); V7 = __builtin_shuffle(X,Y,(v2s){0,2}); V8 = __builtin_shuffle(X,Y,(v2s){1,3});
	for (unsigned int i=0; i<Wo; i++) {
		int x0, x1, x2, x3, Acc = Out[i]<<Norm;
		x0 = In[0*W+i+4]; x1 = In[1*W+i+4]; x2 = In[2*W+i+4]; x3 = In[3*W+i+4];
		V4 = gap8_pack2(x0,x1); V9 = gap8_pack2(x2,x3);
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc); Acc = gap8_sumdotp2(V4, C4, Acc);
		Acc = gap8_sumdotp2(V5, C5, Acc); Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc); Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		V0=V1; V1=V2; V2=V3; V3=V4; V5=V6; V6=V7; V7=V8; V8=V9;
	}
#endif
}

static void __attribute__ ((noinline)) BorderConv5x4Stride2_H_fp(
        short int * __restrict__ In,
        int W,
        int Wo,
        unsigned int Norm,
        short int * __restrict__ Out,
        v2s * __restrict__ Filter
	)

{
#ifdef ASM_OPT
	const v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	          C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];
	v2s X, Y, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;
	v2s *Pt = (v2s *) In;

	int Off0 = 2*W, Off1 = 4*W-4;

 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"i"(4):"memory");    V0 = __builtin_shuffle(X,Y,(v2s){0,2}); V1 = __builtin_shuffle(X,Y,(v2s){1,3});
 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"r"(Off1):"memory"); V2 = __builtin_shuffle(X,Y,(v2s){0,2}); V3 = __builtin_shuffle(X,Y,(v2s){1,3});
 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"i"(4):"memory");    V5 = __builtin_shuffle(X,Y,(v2s){0,2}); V6 = __builtin_shuffle(X,Y,(v2s){1,3});
	Off1 = -Off1-2;
 	asm VOL("p.lw %0,%2(%1)":"=r"(Y):"r"(Pt), "r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"r"(Off1):"memory"); V7 = __builtin_shuffle(X,Y,(v2s){0,2}); V8 = __builtin_shuffle(X,Y,(v2s){1,3});
	Off1=-6*W+4;
	for (unsigned int i=0; i<Wo; i++) {
		int Acc = Out[i]<<Norm;

 		asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(Y),"+r"(Pt):"r"(Off0):"memory"); V3 = __builtin_shuffle(X,Y,(v2s){0,2}); V4 = __builtin_shuffle(X,Y,(v2s){1,3});
 		asm VOL("p.lw %0,%2(%1!)":"=r"(X),"+r"(Pt):"r"(Off0):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(Y),"+r"(Pt):"r"(Off1):"memory"); V8 = __builtin_shuffle(X,Y,(v2s){0,2}); V9 = __builtin_shuffle(X,Y,(v2s){1,3});
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc); Acc = gap8_sumdotp2(V4, C4, Acc);
		Acc = gap8_sumdotp2(V5, C5, Acc); Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc); Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		V0=V2; V1=V3; V2=V4; V5=V7; V6=V8; V7=V9;
	}
#else
	const v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	          C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];

	v2s X, Y, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;

	X = *((v2s *) &In[0*W+0]); Y = *((v2s *) &In[1*W+0]); V0 = __builtin_shuffle(X,Y,(v2s){0,2}); V1 = __builtin_shuffle(X,Y,(v2s){1,3});
	X = *((v2s *) &In[0*W+2]); Y = *((v2s *) &In[1*W+2]); V2 = __builtin_shuffle(X,Y,(v2s){0,2});
	X = *((v2s *) &In[2*W+0]); Y = *((v2s *) &In[3*W+0]); V5 = __builtin_shuffle(X,Y,(v2s){0,2}); V6 = __builtin_shuffle(X,Y,(v2s){1,3});
	X = *((v2s *) &In[2*W+2]); Y = *((v2s *) &In[3*W+2]); V7 = __builtin_shuffle(X,Y,(v2s){0,2});
	for (unsigned int i=0; i<Wo; i++) {
		int Acc = Out[i]<<Norm;
		X = *((v2s *) &In[0*W+2*i+3]); Y = *((v2s *) &In[1*W+2*i+3]); V3 = __builtin_shuffle(X,Y,(v2s){0,2}); V4 = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &In[2*W+2*i+3]); Y = *((v2s *) &In[3*W+2*i+3]); V8 = __builtin_shuffle(X,Y,(v2s){0,2}); V9 = __builtin_shuffle(X,Y,(v2s){1,3});
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc); Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc); Acc = gap8_sumdotp2(V4, C4, Acc);
		Acc = gap8_sumdotp2(V5, C5, Acc); Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc); Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		V0=V2; V1=V3; V2=V4; V5=V7; V6=V8; V7=V9;
	}
#endif
}

static void __attribute__ ((noinline)) BorderConv4x5Stride1_V_fp(
        short int * __restrict__ In,
        int W,
        int Wo, int Ho,
        unsigned int Norm,
        short int * __restrict__ Out,
        v2s * __restrict__ Filter,
	int Top,
	int Bottom
	)

{
#ifdef ASM_OPT
	v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	    C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];

	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;
	short int * __restrict__ PtO = Out;
	v2s * __restrict__ Pt = (v2s *) In;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	int Off0 = 2*W-4;
	int Off1 = 2*Wo;

	if (Ho<=1) {
		C6 = (v2s){0,0}; C7 = (v2s){0,0};
	}
	if (Top) {
		V0 = (v2s){0,0}; V1 = (v2s){0,0};
		V2 = (v2s){0,0}; V3 = (v2s){0,0};
		asm VOL("p.lw %0,%2(%1!)":"=r"(V4),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V5),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V6),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V7),"+r"(Pt):"r"(Off0):"memory");
	} else {
		asm VOL("p.lw %0,%2(%1!)":"=r"(V0),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V1),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V2),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V3),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V4),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V5),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V6),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V7),"+r"(Pt):"r"(Off0):"memory");
	}
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		asm VOL("p.lw %0,%2(%1!)":"=r"(V8),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V9),"+r"(Pt):"r"(Off0):"memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
		Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "r" (Off1) : "memory");
		V0 = V2; V1 = V3;
		V2 = V4; V3 = V5;
		V4 = V6; V5 = V7;
		V6 = V8; V7 = V9;
	}
	if (Bottom) {
		int Acc = *PtO<<Norm;
		Pt = (v2s *) (((short int *) Pt)-4*W);
		asm VOL("p.lw %0,%2(%1!)":"=r"(V0),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V1),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V2),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V3),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V4),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V5),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V6),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V7),"+r"(Pt):"r"(Off0):"memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
		Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "r" (Off1) : "memory");
		if (Ho>1 && Bottom>1) {
			Acc = *PtO<<Norm;
			V0 = V2; V1 = V3;
			V2 = V4; V3 = V5;
			V4 = V6; V5 = V7;
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
			Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
			asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "r" (Off1) : "memory");
		}
	}
#else
	v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	    C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];

	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	short int * __restrict__ PtO = Out;

	if (Ho<=1) {
		C6 = (v2s){0,0}; C7 = (v2s){0,0};
	}
	if (Top) {
		V0 = (v2s){0,0}; V1 = (v2s){0,0};
		V2 = (v2s){0,0}; V3 = (v2s){0,0};
		V4 = *((v2s *)(In+0*W+0)); V5 = *((v2s *)(In+0*W+2));
		V6 = *((v2s *)(In+1*W+0)); V7 = *((v2s *)(In+1*W+2));
	} else {
		V0 = *((v2s *)(In+0*W+0)); V1 = *((v2s *)(In+0*W+2));
		V2 = *((v2s *)(In+1*W+0)); V3 = *((v2s *)(In+1*W+2));
		V4 = *((v2s *)(In+2*W+0)); V5 = *((v2s *)(In+2*W+2));
		V6 = *((v2s *)(In+3*W+0)); V7 = *((v2s *)(In+3*W+2));
	}
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		V8 = *((v2s *)(In+(i+4-Top)*W+0)); V9 = *((v2s *)(In+(i+4-Top)*W+2));
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
		Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15); PtO+=Wo;
		V0 = V2; V1 = V3;
		V2 = V4; V3 = V5;
		V4 = V6; V5 = V7;
		V6 = V8; V7 = V9;
	}
	if (Bottom) {
		int Acc = *PtO<<Norm;
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
		Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15); PtO+=Wo;
		if (Ho>1 && Bottom>1) {
			Acc = *PtO<<Norm;
			V0 = V2; V1 = V3;
			V2 = V4; V3 = V5;
			V4 = V6; V5 = V7;
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
			*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		}
	}
#endif
}

static void __attribute__ ((noinline)) BorderConv4x5Stride2_V_fp(
		short int * __restrict__ In,
		int W, int H,
		int Wo, int Ho,
		unsigned int Norm,
		short int * __restrict__ Out,
		v2s * __restrict__ Filter,
		int Top,
		int Bottom) {

#ifdef ASM_OPT
	const v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	          C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];

	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;
	short int * __restrict__ PtO = Out;
	v2s * __restrict__ Pt = (v2s *) In;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	int Off0 = 2*W-4;
	int Off1 = 2*Wo;
	if (Top) {
		V0 = (v2s){0,0}; V1 = (v2s){0,0};
		V2 = (v2s){0,0}; V3 = (v2s){0,0};
	} else {
		asm VOL("p.lw %0,%2(%1!)":"=r"(V0),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V1),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V2),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V3),"+r"(Pt):"r"(Off0):"memory");
	}
	asm VOL("p.lw %0,%2(%1!)":"=r"(V4),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V5),"+r"(Pt):"r"(Off0):"memory");
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		asm VOL("p.lw %0,%2(%1!)":"=r"(V6),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V7),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V8),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V9),"+r"(Pt):"r"(Off0):"memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
		Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "r" (Off1) : "memory");
		V0 = V4; V1 = V5;
		V2 = V6; V3 = V7;
		V4 = V8; V5 = V9;
	}
	if (Bottom) {
		int Acc = *PtO<<Norm;
		Pt = (v2s *) (((short int *) Pt)-(3)*W);
		asm VOL("p.lw %0,%2(%1!)":"=r"(V0),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V1),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V2),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V3),"+r"(Pt):"r"(Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r"(V4),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V5),"+r"(Pt):"r"(Off0):"memory");
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
		if ((H&0x1)==0) {
			asm VOL("p.lw %0,%2(%1!)":"=r"(V6),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r"(V7),"+r"(Pt):"r"(Off0):"memory");
			Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		}
		Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		asm VOL("p.sh %1,%2(%0!)"  : "+r" (PtO) : "r" (Acc), "r" (Off1) : "memory");
	}

#else

	const v2s C0 = Filter[0], C1 = Filter[1], C2 = Filter[2], C3 = Filter[3], C4 = Filter[4],
	          C5 = Filter[5], C6 = Filter[6], C7 = Filter[7], C8 = Filter[8], C9 = Filter[9];

	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9;
	int Ho_Iter = Max(0, (int)(Ho-Bottom));
	short int * __restrict__ PtO = Out;

	if (Top) {
		Top <<= 1;
		V0 = (v2s){0,0}; V1 = (v2s){0,0};
		V2 = (v2s){0,0}; V3 = (v2s){0,0};
		V4 = *((v2s *)(In+0*W+0)); V5 = *((v2s *)(In+0*W+2));
	} else {
		V0 = *((v2s *)(In+0*W+0)); V1 = *((v2s *)(In+0*W+2));
		V2 = *((v2s *)(In+1*W+0)); V3 = *((v2s *)(In+1*W+2));
		V4 = *((v2s *)(In+2*W+0)); V5 = *((v2s *)(In+2*W+2));
	}
	for (unsigned int i=0; i<Ho_Iter; i++) {
		int Acc = *PtO<<Norm;
		V6 = *((v2s *)(In+(2*i+3-Top)*W+0)); V7 = *((v2s *)(In+(2*i+3-Top)*W+2));
		V8 = *((v2s *)(In+(2*i+4-Top)*W+0)); V9 = *((v2s *)(In+(2*i+4-Top)*W+2));
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
		Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		Acc = gap8_sumdotp2(V8, C8, Acc); Acc = gap8_sumdotp2(V9, C9, Acc);
		Acc = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15); 
		*PtO = Acc; PtO+=Wo;
		V0 = V4; V1 = V5;
		V2 = V6; V3 = V7;
		V4 = V8; V5 = V9;
	}
	if (Bottom) {
		int Acc = *PtO<<Norm;
		Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
		Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
		Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);

		if ((H&0x1)==0) {
			// V6 = *((v2s *)(In+(2*(Ho-Top)+3-Top)*W+0)); V7 = *((v2s *)(In+(2*(Ho-Top)+3-Top)*W+2));
			V6 = *((v2s *)(In+(2*(Ho_Iter)+3-Top)*W+0)); V7 = *((v2s *)(In+(2*(Ho_Iter)+3-Top)*W+2));
			Acc = gap8_sumdotp2(V6, C6, Acc); Acc = gap8_sumdotp2(V7, C7, Acc);
		}

		*PtO =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	}
#endif
}

void __attribute__ ((noinline)) KerDoPaddingConv3x3Stride1_fp(
        short int * __restrict__ In,
        int W, int H,
        int Wo, int Ho,
        short int * __restrict__ Filter,
        unsigned int Norm,
        short int * __restrict__ Out,
        v4s Pad
        )
{
	int Left = Pad[0], Right = Pad[1], Top = Pad[2], Bottom = Pad[3];

	if (Left)    BorderConv2x3Stride1_V_fp(In,         W, Wo,          Ho, Norm, Out,                *((v2s *) &Filter[1]), *((v2s *) &Filter[4]), *((v2s *) &Filter[7]), Top, Bottom);
	if (Right)   BorderConv2x3Stride1_V_fp(In+W-2,     W, Wo,          Ho, Norm, Out+Wo-1,           *((v2s *) &Filter[0]), *((v2s *) &Filter[3]), *((v2s *) &Filter[6]), Top, Bottom);
	if (Top )    BorderConv3x2Stride1_H_fp(In,         W, Wo-(Left+Right), Norm, Out+Left,           gap8_pack2(Filter[3], Filter[6]), gap8_pack2(Filter[4], Filter[7]), gap8_pack2(Filter[5], Filter[8]));
	if (Bottom)  BorderConv3x2Stride1_H_fp(In+(H-2)*W, W, Wo-(Left+Right), Norm, Out+(Ho-1)*Wo+Left, gap8_pack2(Filter[0], Filter[3]), gap8_pack2(Filter[1], Filter[4]), gap8_pack2(Filter[2], Filter[5]));
}

void __attribute__ ((noinline)) KerDoPaddingConv3x3Stride2_fp(
        short int * __restrict__ In,
        int W, int H,
        int Wo, int Ho,
        short int * __restrict__ Filter,
        unsigned int Norm,
        short int * __restrict__ Out,
        v4s Pad
        )
{
        int Left = Pad[0], Right  = Pad[1], Top  = Pad[2], Bottom = Pad[3];

	if (Left)    BorderConv2x3Stride2_V_fp(In,              W, Wo,          Ho, Norm, Out,                *((v2s *) &Filter[1]), *((v2s *) &Filter[4]), *((v2s *) &Filter[7]), Top, Bottom);
	if (Right)   BorderConv2x3Stride2_V_fp(In+W-2,          W, Wo,          Ho, Norm, Out+Wo-1,           *((v2s *) &Filter[0]), *((v2s *) &Filter[3]), *((v2s *) &Filter[6]), Top, Bottom);
	if (Top )    BorderConv3x2Stride2_H_fp(In+Left,         W, Wo-(Left+Right), Norm, Out+Left,           gap8_pack2(Filter[3], Filter[6]), gap8_pack2(Filter[4], Filter[7]), gap8_pack2(Filter[5], Filter[8]));
	if (Bottom)  BorderConv3x2Stride2_H_fp(In+(H-2)*W+Left, W, Wo-(Left+Right), Norm, Out+(Ho-1)*Wo+Left, gap8_pack2(Filter[0], Filter[3]), gap8_pack2(Filter[1], Filter[4]), gap8_pack2(Filter[2], Filter[5]));
}

void __attribute__ ((noinline)) KerDoPaddingConv5x5Stride1_fp(
        short int * __restrict__ In,
        int W, int H,
        int Wo, int Ho,
        short int * __restrict__ Filter,
        unsigned int Norm,
        short int * __restrict__ Out,
        v4s Pad
        )
{
	v2s X, Y, F[4*5];
	int Left = Pad[0], Right = Pad[1], Top = Pad[2], Bottom = Pad[3];

	if (Left) {
		F[0] = *((v2s *) &Filter[ 1]); F[1] = *((v2s *) &Filter[ 3]);
		F[2] = *((v2s *) &Filter[ 6]); F[3] = *((v2s *) &Filter[ 8]);
		F[4] = *((v2s *) &Filter[11]); F[5] = *((v2s *) &Filter[13]);
		F[6] = *((v2s *) &Filter[16]); F[7] = *((v2s *) &Filter[18]);
		F[8] = *((v2s *) &Filter[21]); F[9] = *((v2s *) &Filter[23]);
		BorderConv4x5Stride1_V_fp(In,         W, Wo, Ho, Norm, Out+1,            F, Top, Bottom);

		F[0] = *((v2s *) &Filter[ 2]); F[1] = ((v2s) ((int) (unsigned short int) Filter[ 4]));
		F[2] = *((v2s *) &Filter[ 7]); F[3] = ((v2s) ((int) (unsigned short int) Filter[ 9]));
		F[4] = *((v2s *) &Filter[12]); F[5] = ((v2s) ((int) (unsigned short int) Filter[14]));
		F[6] = *((v2s *) &Filter[17]); F[7] = ((v2s) ((int) (unsigned short int) Filter[19]));
		F[8] = *((v2s *) &Filter[22]); F[9] = ((v2s) ((int) (unsigned short int) Filter[24]));
		BorderConv4x5Stride1_V_fp(In,         W, Wo, Ho, Norm, Out,              F, Top, Bottom);
	}
	if (Right) {
		F[0] = *((v2s *) &Filter[ 0]); F[1] = *((v2s *) &Filter[ 2]);
		F[2] = *((v2s *) &Filter[ 5]); F[3] = *((v2s *) &Filter[ 7]);
		F[4] = *((v2s *) &Filter[10]); F[5] = *((v2s *) &Filter[12]);
		F[6] = *((v2s *) &Filter[15]); F[7] = *((v2s *) &Filter[17]);
		F[8] = *((v2s *) &Filter[20]); F[9] = *((v2s *) &Filter[22]);
		if (Right>1) BorderConv4x5Stride1_V_fp(In+W-4,     W, Wo, Ho, Norm, Out+Wo-2,          F, Top, Bottom);
		F[1][1] = 0; F[3][1] = 0; F[5][1] = 0; F[7][1] = 0; F[9][1] = 0;
		BorderConv4x5Stride1_V_fp(In+W-3,     W, Wo, Ho, Norm, Out+Wo-1,          F, Top, Bottom);
	}
	if (Top ) {
		X = *((v2s *) &Filter[ 5]); Y = *((v2s *) &Filter[10]); F[0] = __builtin_shuffle(X,Y,(v2s){0,2}); F[1] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[ 7]); Y = *((v2s *) &Filter[12]); F[2] = __builtin_shuffle(X,Y,(v2s){0,2}); F[3] = __builtin_shuffle(X,Y,(v2s){1,3}); F[4] = gap8_pack2(Filter[ 9], Filter[14]);
		X = *((v2s *) &Filter[15]); Y = *((v2s *) &Filter[20]); F[5] = __builtin_shuffle(X,Y,(v2s){0,2}); F[6] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[17]); Y = *((v2s *) &Filter[22]); F[7] = __builtin_shuffle(X,Y,(v2s){0,2}); F[8] = __builtin_shuffle(X,Y,(v2s){1,3}); F[9] = gap8_pack2(Filter[19], Filter[24]);
		BorderConv5x4Stride1_H_fp(In,         W, Wo-(Left+Right), Norm, Out+Left+Wo,      F);

		X = *((v2s *) &Filter[10]); Y = *((v2s *) &Filter[15]); F[0] = __builtin_shuffle(X,Y,(v2s){0,2}); F[1] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[12]); Y = *((v2s *) &Filter[17]); F[2] = __builtin_shuffle(X,Y,(v2s){0,2}); F[3] = __builtin_shuffle(X,Y,(v2s){1,3}); F[4] = gap8_pack2(Filter[14], Filter[19]);
		F[5] = ((v2s) ((int) (unsigned short int) Filter[20])); F[6] = ((v2s) ((int) (unsigned short int) Filter[21]));
		F[7] = ((v2s) ((int) (unsigned short int) Filter[22])); F[8] = ((v2s) ((int) (unsigned short int) Filter[23]));
		F[9] = ((v2s) ((int) (unsigned short int) Filter[24]));
		BorderConv5x4Stride1_H_fp(In,         W, Wo-(Left+Right), Norm, Out+Left,         F);
	}
	if (Bottom) {
		X = *((v2s *) &Filter[ 0]); Y = *((v2s *) &Filter[ 5]); F[0] = __builtin_shuffle(X,Y,(v2s){0,2}); F[1] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[ 2]); Y = *((v2s *) &Filter[ 7]); F[2] = __builtin_shuffle(X,Y,(v2s){0,2}); F[3] = __builtin_shuffle(X,Y,(v2s){1,3}); F[4] = gap8_pack2(Filter[ 4], Filter[9]);
		X = *((v2s *) &Filter[10]); Y = *((v2s *) &Filter[15]); F[5] = __builtin_shuffle(X,Y,(v2s){0,2}); F[6] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[12]); Y = *((v2s *) &Filter[17]); F[7] = __builtin_shuffle(X,Y,(v2s){0,2}); F[8] = __builtin_shuffle(X,Y,(v2s){1,3}); F[9] = gap8_pack2(Filter[14], Filter[19]);
		if (Bottom==1) {
			BorderConv5x4Stride1_H_fp(In+(H-4)*W, W, Wo-(Left+Right), Norm, Out+(Ho-1)*Wo+Left, F);
		} else {
			if (Ho>1) BorderConv5x4Stride1_H_fp(In+(H-4)*W, W, Wo-(Left+Right), Norm, Out+(Ho-2)*Wo+Left, F);
			F[5][1] = 0; F[6][1] = 0; F[7][1] = 0; F[8][1] = 0; F[9][1] = 0;
			BorderConv5x4Stride1_H_fp(In+(H-3)*W, W, Wo-(Left+Right), Norm, Out+(Ho-1)*Wo+Left, F);
		}
	}
}

void __attribute__ ((noinline)) KerDoPaddingConv5x5Stride2_fp(
        short int * __restrict__ In,
        int W, int H,
        int Wo, int Ho,
        short int * __restrict__ Filter,
        unsigned int Norm,
        short int * __restrict__ Out,
        v4s Pad
        )
{
	v2s X, Y, F[4*5];
    // int Left = Pad[0]>>1, Right  = Pad[1]>>1, Top  = Pad[2]>>1, Bottom = Pad[3]>>1;
        int Left = (Pad[0]!=0), Right  = (Pad[1]!=0), Top  = (Pad[2]!=0), Bottom = (Pad[3]!=0);
/*	Instead use ? To cover the case where the tiler has bumped into a case where padding overlaps 2 adjacent tiles 
        int Left = (Pad[0]!=0), Right  = (Pad[1]!=0), Top  = (Pad[2]!=0), Bottom = (Pad[3]!=0);
*/

	if (Left) {
		F[0] = *((v2s *) &Filter[ 2]); F[1] = *((v2s *) &Filter[ 4]);
		F[2] = *((v2s *) &Filter[ 7]); F[3] = *((v2s *) &Filter[ 9]);
		F[4] = *((v2s *) &Filter[12]); F[5] = *((v2s *) &Filter[14]);
		F[6] = *((v2s *) &Filter[17]); F[7] = *((v2s *) &Filter[19]);
		F[8] = *((v2s *) &Filter[22]); F[9] = *((v2s *) &Filter[24]);
		F[1][1] = 0; F[3][1] = 0; F[5][1] = 0; F[7][1] = 0; F[9][1] = 0;
		BorderConv4x5Stride2_V_fp(In,         W, H, Wo, Ho, Norm, Out,                F, Top, Bottom); // Pad[2], Pad[3]);
	}
	if (Right) {
		F[0] = *((v2s *) &Filter[ 0]); F[1] = *((v2s *) &Filter[ 2]);
		F[2] = *((v2s *) &Filter[ 5]); F[3] = *((v2s *) &Filter[ 7]);
		F[4] = *((v2s *) &Filter[10]); F[5] = *((v2s *) &Filter[12]);
		F[6] = *((v2s *) &Filter[15]); F[7] = *((v2s *) &Filter[17]);
		F[8] = *((v2s *) &Filter[20]); F[9] = *((v2s *) &Filter[22]);
		if ((W&0x1)!=0) {
			F[1][1] = 0; F[3][1] = 0; F[5][1] = 0; F[7][1] = 0; F[9][1] = 0;
		}
		BorderConv4x5Stride2_V_fp(In+W-4+((W&0x1)!=0),     W, H, Wo, Ho, Norm, Out+Wo-1,          F, Top, Bottom); // Pad[2], Pad[3]);
	}
	if (Top ) {
		X = *((v2s *) &Filter[10]); Y = *((v2s *) &Filter[15]); F[0] = __builtin_shuffle(X,Y,(v2s){0,2}); F[1] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[12]); Y = *((v2s *) &Filter[17]); F[2] = __builtin_shuffle(X,Y,(v2s){0,2}); F[3] = __builtin_shuffle(X,Y,(v2s){1,3}); F[4] = gap8_pack2(Filter[14], Filter[19]);
		F[5] = gap8_pack2(Filter[20], 0); F[6] = gap8_pack2(Filter[21], 0); F[7] = gap8_pack2(Filter[22], 0); F[8] = gap8_pack2(Filter[23], 0); F[9] = gap8_pack2(Filter[24], 0);
		BorderConv5x4Stride2_H_fp(In,         W, Wo-(Left+Right), Norm, Out+Left,           F);
	}
	if (Bottom) {
		X = *((v2s *) &Filter[ 0]); Y = *((v2s *) &Filter[ 5]); F[0] = __builtin_shuffle(X,Y,(v2s){0,2}); F[1] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[ 2]); Y = *((v2s *) &Filter[ 7]); F[2] = __builtin_shuffle(X,Y,(v2s){0,2}); F[3] = __builtin_shuffle(X,Y,(v2s){1,3}); F[4] = gap8_pack2(Filter[ 4], Filter[9]);
		X = *((v2s *) &Filter[10]); Y = *((v2s *) &Filter[15]); F[5] = __builtin_shuffle(X,Y,(v2s){0,2}); F[6] = __builtin_shuffle(X,Y,(v2s){1,3});
		X = *((v2s *) &Filter[12]); Y = *((v2s *) &Filter[17]); F[7] = __builtin_shuffle(X,Y,(v2s){0,2}); F[8] = __builtin_shuffle(X,Y,(v2s){1,3}); F[9] = gap8_pack2(Filter[14], Filter[19]);
		if ((H&0x1)!=0) {
			F[5][1] = 0; F[6][1] = 0; F[7][1] = 0; F[8][1] = 0; F[9][1] = 0;
		}
		BorderConv5x4Stride2_H_fp(In+(H-4+((H&0x1)!=0))*W, W, Wo-(Left+Right), Norm, Out+(Ho-1)*Wo+Left, F);
	}
}

void __attribute__ ((noinline)) KerConv1x1Stride1_fp(
	short int * __restrict__ In,
	unsigned int W,
	unsigned int H,
	int F,
	unsigned int Norm,
	short int * __restrict__ Out
	)

{
	unsigned int Wo=W;
	unsigned int Ho=H;

	if (Wo&0x1) {
		for (unsigned int i=0; i<Ho; i++) {
			v2s *LineOut = (v2s *) (&Out[Wo*i]);
			for (unsigned int j=0; j<(Wo/2); j++) {
				v2s O = LineOut[j];
				int Acc0 = O[0]<<Norm, Acc1 = O[1]<<Norm;
				Acc0 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc0, In[i*W + 2*j  ], F), Norm), 15);
				Acc1 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc1, In[i*W + 2*j+1], F), Norm), 15);
				LineOut[j] =  gap8_pack2(Acc0, Acc1);
			}
			Out[Wo*i+Wo-1] = gap8_clip(gap8_roundnorm_reg(gap8_macs((Out[Wo*i+Wo-1]<<Norm), In[W*i+Wo-1], F), Norm), 15);
		}
	} else {
		for (unsigned int i=0; i<Ho; i++) {
			v2s *LineOut = (v2s *) (&Out[Wo*i]);
			v2s *LineIn = (v2s *) (&In[W*i]);
			for (unsigned int j=0; j<(Wo/2); j++) {
				v2s O = LineOut[j];
				int Acc0 = O[0]<<Norm, Acc1 = O[1]<<Norm;
				Acc0 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc0, In[i*W + 2*j  ], F), Norm), 15);
				Acc1 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc1, In[i*W + 2*j+1], F), Norm), 15);
				LineOut[j] =  gap8_pack2(Acc0, Acc1);
			}
		}
	}
}

void __attribute__ ((noinline)) KerConv1x1Stride2_fp(
	short int * __restrict__ In,
	unsigned int W,
	unsigned int H,
	int F,
	unsigned int Norm,
	short int * __restrict__ Out
	)

{
	unsigned int Wo = (W-1)/2+1, Ho = (H-1)/2+1;

	if (Wo&0x1) {
		for (unsigned int i=0; i<Ho; i++) {
			v2s *LineOut = (v2s *) (&Out[Wo*i]);
			for (unsigned int j=0; j<(Wo/2); j++) {
				v2s O = LineOut[j];
				int Acc0 = O[0]<<Norm, Acc1 = O[1]<<Norm;
				Acc0 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc0, In[2*i*W + 4*j    ], F), Norm), 15);
				Acc1 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc1, In[2*i*W + 4*j + 2], F), Norm), 15);
				LineOut[j] =  gap8_pack2(Acc0, Acc1);
			}
			Out[Wo*i+Wo-1] = gap8_clip(gap8_roundnorm_reg(gap8_macs((Out[Wo*i+Wo-1]<<Norm), In[W*2*i+2*(Wo-1)], F), Norm), 15);
		}
	} else {
		for (unsigned int i=0; i<Ho; i++) {
			v2s *LineOut = (v2s *) (&Out[Wo*i]);
			for (unsigned int j=0; j<(Wo/2); j++) {
				v2s O = LineOut[j];
				int Acc0 = O[0]<<Norm, Acc1 = O[1]<<Norm;
				Acc0 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc0, In[2*i*W + 4*j    ], F), Norm), 15);
				Acc1 = gap8_clip(gap8_roundnorm_reg(gap8_macs(Acc1, In[2*i*W + 4*j + 2], F), Norm), 15);
				LineOut[j] =  gap8_pack2(Acc0, Acc1);
			}
		}
	}
}

void __attribute__ ((noinline)) KerConv3x3Stride1_fp(KerPaddedConv_fpT *Arg)

{
	short int * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	short int *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	short int * __restrict__ Out = Arg->Out;
	v2s C0 = *((v2s *) &Filter[0]),	           C1 = *((v2s *) &Filter[3]),
	    C2 = gap8_pack2(Filter[2], Filter[5]), C3 = *((v2s *) &Filter[6]),
	    C4 = gap8_pack2(Filter[8], 0);
	v2s V0, V1, V2, V3, V4;
	unsigned int i, j;
	unsigned int Wo, Ho;
	int In_Off, Out_Off, Right, Bottom, Wo_Iter, Ho_Iter;
	v2s *Pt;

	if ((int) Arg->Pad) {
		Wo = (W-3+Arg->Pad[0]+Arg->Pad[1]) + 1; 		Ho = (H-3+Arg->Pad[2]+Arg->Pad[3]) + 1;
		// In_Off = If (Pad): -Pad+Stride Else 0. Pad=1,Stride=1 => 0
		In_Off = 0;                             		Out_Off = Arg->Pad[0]+Arg->Pad[2]*Wo;
		Wo_Iter = Max(0, (int)(Wo-Arg->Pad[0]-Arg->Pad[1]));   	Ho_Iter = Max(0, (int)(Ho-Arg->Pad[2]-Arg->Pad[3]));
	} else {
		Wo = (W-3)+1; Ho = (H-3)+1;
		In_Off = 0; Out_Off = 0;
		Wo_Iter = Wo; Ho_Iter = Ho;
	}

/****************************** PROFILING KERNEL ******************************/
#ifdef PROFILING_BK
	unsigned int CoreId = gap8_coreid();
	// initialize the performance clock
	if(CoreId==0) {
		rt_perf_init(&perf[CoreId]);
		// Configure performance counters for counting the cycles
		rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES));
		rt_perf_reset(&perf[CoreId]);
		rt_perf_start(&perf[CoreId]);
	}
#endif
/******************************************************************************/

#ifdef ASM_OPT
	int Off0 = 2*W-4;
	for (j=0;j<Wo_Iter; j++) {
		Pt = (v2s *) (In + j);
		asm VOL("p.lw %0,%2(%1!)":"=r"(V0),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lh %0,%2(%1!)":"=r"(V2),"+r"(Pt):"r"(Off0):"memory"); 
		asm VOL("p.lw %0,%2(%1!)":"=r"(V1),"+r"(Pt):"i"(4):"memory"); asm VOL("p.lh %0,%2(%1!)":"=r"(V3),"+r"(Pt):"r"(Off0):"memory"); V2 = __builtin_shuffle(V2, V3, (v2s){0,2});
		for (i=0; i<Ho_Iter; i++)  {
			int Acc = Out[i*Wo+j+Out_Off]<<Norm;
			asm VOL ("p.lw %0,%2(%1!)" : "=r" (V3), "+r" (Pt) : "i" (4) : "memory"); asm VOL ("p.lh %0,%2(%1!)" : "=r" (V4), "+r" (Pt) : "r" (Off0) : "memory");
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
			V0 = V1; V1 = V3;
			V2 = __builtin_shuffle(V2, V4, (v2s) {1, 2});
		}
	}
#else
	for (j=0;j<Wo_Iter; j++) {
		Pt = (v2s *) (&In[j]);	 V0 = *Pt++; V2 = *Pt; 
		Pt = (v2s *) (&In[W+j]); V1 = *Pt++; V3 = *Pt; V2 = __builtin_shuffle(V2, V3, (v2s){0,2});
		for (i=0; i<Ho_Iter; i++)  {
			int Acc = Out[i*Wo+j+Out_Off]<<Norm;
			Pt = (v2s *) (&In[(i+2)*W+j]); V3 = *Pt++; V4 = *Pt;
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
			V0 = V1; V1 = V3;
			V2 = __builtin_shuffle(V2, V4, (v2s) {1, 2});
		}
	}
#endif
	if ((int) Arg->Pad) KerDoPaddingConv3x3Stride1_fp(In, W, H, Wo, Ho, Filter, Norm, Out, Arg->Pad);

/****************************** PROFILING KERNEL ******************************/
#ifdef PROFILING_BK
	if(CoreId==0) {
		rt_perf_stop(&perf[CoreId]);
		rt_perf_save(&perf[CoreId]);

		printf("Cycles_%d: %d\n", CoreId, rt_perf_get(&perf[CoreId], RT_PERF_CYCLES));
	}
#endif
/******************************************************************************/
}

void __attribute__ ((noinline)) KerConv3x3Stride2_fp(KerPaddedConv_fpT *Arg)

{
	short int * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	short int *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	short int * __restrict__ Out = Arg->Out;
	v2s C0 = *((v2s *) &Filter[0]), C1 = gap8_pack2(Filter[2], 0),
	    C2 = *((v2s *) &Filter[3]), C3 = gap8_pack2(Filter[5], 0),
	    C4 = *((v2s *) &Filter[6]), C5 = gap8_pack2(Filter[8], 0);
	v2s V0, V1, V2, V3, V4, V5;
	unsigned int i, j;
	unsigned int Wo, Ho;
	int In_Off, Out_Off, Right, Bottom, Wo_Iter, Ho_Iter;
	v2s *Pt;

	if ((int) Arg->Pad) {
		Wo = (W-3+Arg->Pad[0]+Arg->Pad[1])/2 + 1;        Ho = (H-3+Arg->Pad[2]+Arg->Pad[3])/2 + 1;
		In_Off = Arg->Pad[0] + Arg->Pad[2]*W;            Out_Off = Arg->Pad[0]+Arg->Pad[2]*Wo;
		// Process Right: Pad Right and Padded W is Odd. Process Bottom: Pad Bottom and Padded H is Odd
		Right = Arg->Pad[1]&(W+Arg->Pad[0]+Arg->Pad[1]); Bottom = Arg->Pad[3]&(H+Arg->Pad[2]+Arg->Pad[3]);
		// Right and Bottom padding processing update
		Arg->Pad[1] = Right;                             Arg->Pad[3] = Bottom;
		Wo_Iter = Wo-Arg->Pad[0]-Right;                  Ho_Iter = Ho-Arg->Pad[2]-Bottom;
	} else {
		Wo = (W-3)/2+1; Ho = (H-3)/2+1;
		In_Off = 0; Out_Off = 0;
		Wo_Iter = Wo; Ho_Iter = Ho;
		Right = 0; Bottom = 0; // No need
	}

#ifdef ASM_OPT
	int Off0 = 2*W-4;
	for (j=0; j<Wo_Iter; j++) {
		Pt = (v2s *) (In+2*j+In_Off);
		asm("p.lw %0,%2(%1!)" : "=r" (V0), "+r" (Pt) : "i" (4) : ); asm("p.lh %0,%2(%1!)" : "=r" (V1), "+r" (Pt) : "r" (Off0) : );
		for (i=0; i<Ho_Iter; i++) {
			int Acc = Out[i*Wo+j+Out_Off]<<Norm;
			asm VOL("p.lw %0,%2(%1!)" : "=r" (V2), "+r" (Pt) : "i" (4) : "memory"); asm VOL("p.lh %0,%2(%1!)" : "=r" (V3), "+r" (Pt) : "r" (Off0) : "memory");
			asm VOL("p.lw %0,%2(%1!)" : "=r" (V4), "+r" (Pt) : "i" (4) : "memory"); asm VOL("p.lh %0,%2(%1!)" : "=r" (V5), "+r" (Pt) : "r" (Off0) : "memory");
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
			V0 = V4; V1 = V5;
		}
	}
#else
	for (j=0; j<Wo_Iter; j++) {
		Pt = (v2s *) (&In[2*j+In_Off]); V0 = Pt[0]; V1 = Pt[1];
		for (i=0; i<Ho_Iter; i++) {
			int Acc = Out[i*Wo+j+Out_Off]<<Norm;
			Pt = (v2s *) (&In[(2*i+1)*W+2*j+In_Off]); V2 = Pt[0]; V3 = Pt[1];
			Pt = (v2s *) (&In[(2*i+2)*W+2*j+In_Off]); V4 = Pt[0]; V5 = Pt[1];
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
			V0 = V4; V1 = V5;
		}
	}
#endif
	if ((int) Arg->Pad) KerDoPaddingConv3x3Stride2_fp(In, W, H, Wo, Ho, Filter, Norm, Out, Arg->Pad);
}


void __attribute__ ((noinline)) KerConv5x5Stride1_fp(KerPaddedConv_fpT *Arg)

{
	short int * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	short int *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	short int * __restrict__ Out = Arg->Out;
	v2s C0  = *((v2s *) &Filter[0]),            C1  = *((v2s *) &Filter[2]),
	    C2  = *((v2s *) &Filter[5]),            C3  = *((v2s *) &Filter[7]),
	    C4  = *((v2s *) &Filter[10]),           C5  = *((v2s *) &Filter[12]),
	    C6  = *((v2s *) &Filter[15]),           C7  = *((v2s *) &Filter[17]),
	    C8  = *((v2s *) &Filter[20]),           C9  = *((v2s *) &Filter[22]),
	    C10 = gap8_pack2(Filter[4], Filter[9]), C11 = gap8_pack2(Filter[14], Filter[19]),
	    C12 = gap8_pack2(Filter[24], 0);

	unsigned int i, j;
	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12;
	v2s Mask  = {1,2};
	unsigned int Wo, Ho;
	int In_Off, Out_Off, Wo_Iter, Ho_Iter;
	v2s *Pt;

	if ((int) Arg->Pad) {
		Wo = (W-5+Arg->Pad[0]+Arg->Pad[1]) + 1;    		   Ho = (H-5+Arg->Pad[2]+Arg->Pad[3]) + 1;
		In_Off = 0;                                		   Out_Off = (Arg->Pad[0]>>0)+(Arg->Pad[2]>>0)*Wo;
		Wo_Iter = Max(0, (int) (Wo-(Arg->Pad[0]+Arg->Pad[1])>>0)); Ho_Iter = Max(0, (int) (Ho-(Arg->Pad[2]+Arg->Pad[3])>>0));
	} else {
		Wo = (W-5)+1; Ho = (H-5)+1;
		In_Off = 0; Out_Off = 0;
		Wo_Iter = Wo; Ho_Iter = Ho;
	}


#ifdef ASM_OPT
	int Off0 = 2*W - 8;

	for (j=0; j<Wo_Iter; j++) {
		Pt = (v2s *) (In+j+In_Off);
		int X, Y;
		asm("p.lw %0,%2(%1!)" : "=r" (V0), "+r" (Pt) : "i" (4) : ); asm("p.lw %0,%2(%1!)" : "=r" (V1), "+r" (Pt) : "i" (4) : ); asm("p.lh %0,%2(%1!)" : "=r" (X), "+r" (Pt) : "r" (Off0) : );
		asm("p.lw %0,%2(%1!)" : "=r" (V2), "+r" (Pt) : "i" (4) : ); asm("p.lw %0,%2(%1!)" : "=r" (V3), "+r" (Pt) : "i" (4) : ); asm("p.lh %0,%2(%1!)" : "=r" (Y), "+r" (Pt) : "r" (Off0) : ); V10 = gap8_pack2(X, Y);
		asm("p.lw %0,%2(%1!)" : "=r" (V4), "+r" (Pt) : "i" (4) : ); asm("p.lw %0,%2(%1!)" : "=r" (V5), "+r" (Pt) : "i" (4) : ); asm("p.lh %0,%2(%1!)" : "=r" (X), "+r" (Pt) : "r" (Off0) : );
		asm("p.lw %0,%2(%1!)" : "=r" (V6), "+r" (Pt) : "i" (4) : ); asm("p.lw %0,%2(%1!)" : "=r" (V7), "+r" (Pt) : "i" (4) : ); asm("p.lh %0,%2(%1!)" : "=r" (Y), "+r" (Pt) : "r" (Off0) : ); V11 = gap8_pack2(X, Y);
		for (i=0; i<Ho_Iter; i++) {
			int S = Out[i*Wo+j+Out_Off]<<Norm;
			asm VOL("p.lw %0,%2(%1!)":"=r" (V8),"+r" (Pt):"i" (4):"memory"); asm VOL("p.lw %0,%2(%1!)":"=r" (V9),"+r" (Pt):"i" (4):"memory"); asm VOL("p.lh %0,%2(%1!)":"=r" (V12),"+r" (Pt):"r" (Off0):"memory");
			S = gap8_sumdotp2(V0,  C0,  S); S = gap8_sumdotp2(V1,  C1,  S); S = gap8_sumdotp2(V10, C10, S);
			S = gap8_sumdotp2(V2,  C2,  S); S = gap8_sumdotp2(V3,  C3,  S);
			S = gap8_sumdotp2(V4,  C4,  S); S = gap8_sumdotp2(V5,  C5,  S); S = gap8_sumdotp2(V11, C11, S);
			S = gap8_sumdotp2(V6,  C6,  S); S = gap8_sumdotp2(V7,  C7,  S);
			S = gap8_sumdotp2(V8,  C8,  S); S = gap8_sumdotp2(V9,  C9,  S); S = gap8_sumdotp2(V12, C12, S);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
			V0 = V2; V1 = V3; V2 = V4; V3 = V5; V4 = V6; V5 = V7; V6 = V8; V7 = V9;
			V10 = __builtin_shuffle(V10, V11, Mask); V11 = __builtin_shuffle(V11, V12, Mask);
		}
	}
#else
	for (j=0; j<Wo_Iter; j++) {
		Pt = (v2s *) (In+0*W+j+In_Off); V0 = Pt[0]; V1 = Pt[1]; V10 = (v2s) {In[j+0*W+4], In[j+1*W+4]};
		Pt = (v2s *) (In+1*W+j+In_Off); V2 = Pt[0]; V3 = Pt[1];
		Pt = (v2s *) (In+2*W+j+In_Off); V4 = Pt[0]; V5 = Pt[1]; V11 = (v2s) {In[j+2*W+4], In[j+3*W+4]};
		Pt = (v2s *) (In+3*W+j+In_Off); V6 = Pt[0]; V7 = Pt[1];
		for (i=0; i<Ho_Iter; i++) {
			int S = Out[i*Wo+j+Out_Off]<<Norm;
			Pt = (v2s *) (&In[(i+4)*W+j+In_Off]); V8 = Pt[0]; V9 = Pt[1]; V12 = Pt[2];
			S = gap8_sumdotp2(V0,  C0,  S); S = gap8_sumdotp2(V1,  C1,  S); S = gap8_sumdotp2(V10, C10, S);
			S = gap8_sumdotp2(V2,  C2,  S); S = gap8_sumdotp2(V3,  C3,  S);
			S = gap8_sumdotp2(V4,  C4,  S); S = gap8_sumdotp2(V5,  C5,  S); S = gap8_sumdotp2(V11, C11, S);
			S = gap8_sumdotp2(V6,  C6,  S); S = gap8_sumdotp2(V7,  C7,  S);
			S = gap8_sumdotp2(V8,  C8,  S); S = gap8_sumdotp2(V9,  C9,  S); S = gap8_sumdotp2(V12, C12, S);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
			V0 = V2; V1 = V3; V2 = V4; V3 = V5; V4 = V6; V5 = V7; V6 = V8; V7 = V9;
			V10 = __builtin_shuffle(V10, V11, Mask); V11 = __builtin_shuffle(V11, V12, Mask);
		}
	} 
#endif
	if ((int) Arg->Pad) KerDoPaddingConv5x5Stride1_fp(In, W, H, Wo, Ho, Filter, Norm, Out, Arg->Pad);
}

void __attribute__ ((noinline)) KerConv5x5Stride2_fp(KerPaddedConv_fpT *Arg) {

	short int * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	short int *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	short int * __restrict__ Out = Arg->Out;
	v2s C0  = *((v2s *) &Filter[0]),            C1  = *((v2s *) &Filter[2]),
	    C2  = *((v2s *) &Filter[5]),            C3  = *((v2s *) &Filter[7]),
	    C4  = *((v2s *) &Filter[10]),           C5  = *((v2s *) &Filter[12]),
	    C6  = *((v2s *) &Filter[15]),           C7  = *((v2s *) &Filter[17]),
	    C8  = *((v2s *) &Filter[20]),           C9  = *((v2s *) &Filter[22]),
	    C10 = gap8_pack2(Filter[4], Filter[9]), C11 = gap8_pack2(Filter[14], Filter[19]),
	    C12 = gap8_pack2(Filter[24], 0);
	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12;
	unsigned int i, j;
	unsigned int Wo, Ho;
	int In_Off, Out_Off, Right, Bottom, Wo_Iter, Ho_Iter;
	v2s *Pt;

	if ((int) Arg->Pad) {
		int Fwd, Lwd, Fhd, Lhd;
		Fwd = (Arg->Pad[0]+1)/2; 		     
		Lwd = (W-3+Arg->Pad[0])/2;
		Fhd = (Arg->Pad[2]+1)/2; 		     
		Lhd = (H-3+Arg->Pad[2])/2;
		Wo = (W-5+Arg->Pad[0]+Arg->Pad[1])/2 + 1;    
		Ho = (H-5+Arg->Pad[2]+Arg->Pad[3])/2 + 1;
		In_Off = 0;                                  
		Out_Off = Fhd*Wo+Fwd;
		Wo_Iter = Lwd-Fwd;			     
		Ho_Iter = Lhd-Fhd;
	} else {
		Wo = (W-5)/2+1; 
		Ho = (H-5)/2+1;
		In_Off = 0; Out_Off = 0;
		Wo_Iter = Wo; 
		Ho_Iter = Ho;
	}

#ifdef ASM_OPT
	int Off0 = 2*W-8;
	for(j=0; j<Wo_Iter; j++) {
		Pt = (v2s *) (In+2*j+In_Off);
		int X, Y;
		asm VOL("p.lw %0,%2(%1!)":"=r" (V0),"+r" (Pt):"i" (4):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r" (V1),"+r" (Pt):"i" (4):"memory");
		asm VOL("p.lh %0,%2(%1!)" :"=r" (X),  "+r" (Pt):"r" (Off0):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r" (V2),"+r" (Pt):"i" (4):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r" (V3),"+r" (Pt):"i" (4):"memory");
		asm VOL("p.lh %0,%2(%1!)" :"=r" (Y),  "+r" (Pt):"r" (Off0):"memory");
		V10 = gap8_pack2(X, Y);
		asm VOL("p.lw %0,%2(%1!)":"=r" (V4),"+r" (Pt):"i" (4):"memory");
		asm VOL("p.lw %0,%2(%1!)":"=r" (V5),"+r" (Pt):"i" (4):"memory");
		asm VOL("p.lhu %0,%2(%1!)":"=r" (V11),"+r" (Pt):"r" (Off0):"memory");
		for(i=0; i<Ho_Iter; i++) {
			int S = Out[i*Wo+j+Out_Off] << Norm;
			asm VOL("p.lw %0,%2(%1!)":"=r" (V6),"+r" (Pt):"i" (4):"memory");
			asm VOL("p.lw %0,%2(%1!)":"=r" (V7), "+r" (Pt):"i" (4):"memory");
			asm VOL("p.lh %0,%2(%1!)":"=r" (X),  "+r" (Pt):"r" (Off0):"memory");
			V11 = gap8_pack2((int)V11, X);
			asm VOL("p.lw %0,%2(%1!)":"=r" (V8),"+r" (Pt):"i" (4):"memory");
			asm VOL("p.lw %0,%2(%1!)":"=r" (V9), "+r" (Pt):"i" (4):"memory");
			asm VOL("p.lh %0,%2(%1!)":"=r" (V12),"+r" (Pt):"r" (Off0):"memory");
			S = gap8_sumdotp2(V0, C0, S);
			S = gap8_sumdotp2(V1, C1, S);
			S = gap8_sumdotp2(V10, C10, S);
			S = gap8_sumdotp2(V2, C2, S);
			S = gap8_sumdotp2(V3, C3, S);
			S = gap8_sumdotp2(V4, C4, S);
			S = gap8_sumdotp2(V5, C5, S);
			S = gap8_sumdotp2(V11, C11, S);
			S = gap8_sumdotp2(V6, C6, S);
			S = gap8_sumdotp2(V7, C7, S);
			S = gap8_sumdotp2(V8, C8, S);
			S = gap8_sumdotp2(V9, C9, S);
			S = gap8_sumdotp2(V12, C12, S);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
			V10 = V11; V11 = V12; V0 = V4; V1 = V5; V2 = V6; V3 = V7; V4 = V8; V5 = V9;
		}
	}

#else

	for(j=0; j<Wo_Iter; j++) {
		Pt = (v2s *) (&In[0*W+2*j+In_Off]);
		V0 = Pt[0];
		V1 = Pt[1];
		Pt = (v2s *) (&In[1*W+2*j+In_Off]);
		V2 = Pt[0];
		V3 = Pt[1];
		V10 = (v2s) {In[0*W+2*j+4], In[1*W+2*j+4]};
		Pt = (v2s *) (&In[2*W+2*j+In_Off]);
		V4 = Pt[0];
		V5 = Pt[1];
		V11 = (v2s) {In[2*W+2*j+4], 0};

		for(i=0; i<Ho_Iter; i++) {
			int S = Out[i*Wo+j+Out_Off]<<Norm;
			Pt = (v2s *) (&In[(2*i+3)*W+2*j+In_Off]);
			V6 = Pt[0];
			V7 = Pt[1];
			V11 = (v2s) {(int) V11, (int) Pt[2]};
			Pt = (v2s *) (&In[(2*i+4)*W+2*j+In_Off]);
			V8 = Pt[0];
			V9 = Pt[1];
			V12 = Pt[2];
			S = gap8_sumdotp2(V0, C0, S);
			S = gap8_sumdotp2(V1, C1, S);
			S = gap8_sumdotp2(V10, C10, S);
			S = gap8_sumdotp2(V2, C2, S);
			S = gap8_sumdotp2(V3, C3, S);
			S = gap8_sumdotp2(V4, C4, S);
			S = gap8_sumdotp2(V5, C5, S);
			S = gap8_sumdotp2(V11, C11, S);
			S = gap8_sumdotp2(V6, C6, S);
			S = gap8_sumdotp2(V7, C7, S);
			S = gap8_sumdotp2(V8, C8, S);
			S = gap8_sumdotp2(V9, C9, S);
			S = gap8_sumdotp2(V12, C12, S);
			Out[i*Wo+j+Out_Off] =  gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
			V10 = V11; V11 = V12; V0 = V4; V1 = V5; V2 = V6; V3 = V7; V4 = V8; V5 = V9;
		}
	}

#endif

	if((int) Arg->Pad) {
		KerDoPaddingConv5x5Stride2_fp(In, W, H, Wo, Ho, Filter, Norm, Out, Arg->Pad);
	}
}


void KerParConv1x1Stride1_fp(KerParConv_fp_T *Arg)

{
	unsigned int FS=1, S=1;
        short int * __restrict__ In = Arg->In;
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
        unsigned int InFeatures = Arg->InFeatures;
        unsigned int TotalInFeatures = Arg->InFeatures;
        unsigned int OutFeatures = Arg->OutFeatures;
        unsigned int BaseOutFeature = Arg->BaseOutFeature;
        unsigned int BaseInFeature = 0;
        short int * __restrict__ Filter = Arg->Filter;
        short int * __restrict__ Out = Arg->Out;
        unsigned int Norm = Arg->Norm;

	unsigned int CoreId = gap8_coreid();
	unsigned int Chunk = ChunkSize(OutFeatures);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, OutFeatures);
	v4s PadIn = (v4s){0,0,0,0};

	if (Arg->NTile==0) {
		BaseInFeature = BaseOutFeature; BaseOutFeature = 0;
		TotalInFeatures = Arg->TileIndex;
	}

	KerPaddedConv_fpT ArgConv = {0, W, H, 0, 0, Norm, FS, S, PadIn};
        unsigned int Wo = (W-FS)/S + 1;
        unsigned int Ho = (H-FS)/S + 1;

	First += BaseOutFeature; Last += BaseOutFeature;
	for (unsigned int of=First; of<Last; of++) {
		for (unsigned int If=0; If<InFeatures; If++) {
			KerConv1x1Stride1_fp(In+W*H*If, W, H, Filter[FS*FS*(TotalInFeatures*(of-BaseOutFeature) + If + BaseInFeature)], Norm, Out+Wo*Ho*of);
		}
	}
	gap8_waitbarrier(0);
}

void KerParConv1x1Stride2_fp(KerParConv_fp_T *Arg)

{
	unsigned int FS=1, S=2;
        short int * __restrict__ In = Arg->In;
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
        unsigned int InFeatures = Arg->InFeatures;
        unsigned int TotalInFeatures = Arg->InFeatures;
        unsigned int OutFeatures = Arg->OutFeatures;
        unsigned int BaseOutFeature = Arg->BaseOutFeature;
        unsigned int BaseInFeature = 0;
        short int * __restrict__ Filter = Arg->Filter;
        short int * __restrict__ Out = Arg->Out;
        unsigned int Norm = Arg->Norm;

	unsigned int CoreId = gap8_coreid();
	unsigned int Chunk = ChunkSize(OutFeatures);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, OutFeatures);
	v4s PadIn = (v4s){0,0,0,0};

	if (Arg->NTile==0) {
		BaseInFeature = BaseOutFeature; BaseOutFeature = 0;
		TotalInFeatures = Arg->TileIndex;
	}

	KerPaddedConv_fpT ArgConv = {0, W, H, 0, 0, Norm, FS, S, PadIn};
        unsigned int Wo = (W-FS)/S + 1;
        unsigned int Ho = (H-FS)/S + 1;

	First += BaseOutFeature; Last += BaseOutFeature;
	for (unsigned int of=First; of<Last; of++) {
		for (unsigned int If=0; If<InFeatures; If++) {
    // printf("Out Feat %d, if: %d, Base if: %d\n", of, If, BaseInFeature);
    // DumpPlane("OutIn", Out+Wo*Ho*of, Wo, 10, Ho, 10);
			KerConv1x1Stride2_fp(In+W*H*If, W, H, Filter[FS*FS*(TotalInFeatures*(of-BaseOutFeature) + If + BaseInFeature)], Norm, Out+Wo*Ho*of);
    // DumpPlane("In", In+W*H*If, W, 10, H, 10);
    // DumpPlane("Filter", Filter+ FS*FS*(TotalInFeatures*(of-BaseOutFeature) + If + BaseInFeature), FS, 10, FS, 10);
    // DumpPlane("Out", Out+Wo*Ho*of, Wo, 10, Ho, 10);
		}
	}
	gap8_waitbarrier(0);
}

void KerParConv3x3Stride1_fp(KerParConv_fp_T *Arg)

{
	unsigned int FS=3, S=1;
        short int * __restrict__ In = Arg->In;
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
        unsigned int InFeatures = Arg->InFeatures;
        unsigned int TotalInFeatures = Arg->InFeatures;
        unsigned int OutFeatures = Arg->OutFeatures;
        unsigned int BaseOutFeature = Arg->BaseOutFeature;
        unsigned int BaseInFeature = 0;
        short int * __restrict__ Filter = Arg->Filter;
        short int * __restrict__ Out = Arg->Out;
        unsigned int Norm = Arg->Norm;

	unsigned int CoreId = gap8_coreid();
	unsigned int Chunk = ChunkSize(OutFeatures);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, OutFeatures);

	v4s PadIn = Arg->Pad;
	if (((int)Arg->Pad) && Arg->NTile) {
		PadIn[0] = (FS-1)/2; PadIn[1] = FS/2;
		PadIn[2] *= (Arg->TileIndex==0); PadIn[3] *= (Arg->TileIndex==(Arg->NTile-1));
		H -= (PadIn[2]+PadIn[3]);
	} else if (Arg->NTile==0) {
		BaseInFeature = BaseOutFeature; BaseOutFeature = 0;
		TotalInFeatures = Arg->TileIndex;
	}

	KerPaddedConv_fpT ArgConv = {0, W, H, 0, 0, Norm, FS, S, PadIn};
        unsigned int Wo = (W-FS+PadIn[0]+PadIn[1])/S + 1;
        unsigned int Ho = (H-FS+PadIn[2]+PadIn[3])/S + 1;

	First += BaseOutFeature; Last += BaseOutFeature;
	for (unsigned int of=First; of<Last; of++) {
		for (unsigned int If=0; If<InFeatures; If++) {
			ArgConv.In = In+W*H*If; ArgConv.Filter = Filter+FS*FS*(TotalInFeatures*(of-BaseOutFeature) + If + BaseInFeature); ArgConv.Out = Out+Wo*Ho*of;
//    printf("Out Feat %d, if: %d, Filter: %X\n", of, If, (ArgConv.Filter));
//    DumpPlane("OutIn", ArgConv.Out, Wo, 10, Ho, 10);
			KerConv3x3Stride1_fp(&ArgConv);
//    DumpPlane("In", ArgConv.In, ArgConv.W, 10, ArgConv.H, 10);
//    DumpPlane("Filter", ArgConv.Filter, FS, 10, FS, 10);
//    DumpPlane("Out", ArgConv.Out, Wo, 10, Ho, 10);
		}
	}
	gap8_waitbarrier(0);
}

void KerParConv3x3Stride2_fp(KerParConv_fp_T *Arg)

{
	unsigned int FS=3, S=2;
        short int * __restrict__ In = Arg->In;
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
        unsigned int InFeatures = Arg->InFeatures;
        unsigned int TotalInFeatures = Arg->InFeatures;
        unsigned int OutFeatures = Arg->OutFeatures;
        unsigned int BaseOutFeature = Arg->BaseOutFeature;
        unsigned int BaseInFeature = 0;
        short int * __restrict__ Filter = Arg->Filter;
        short int * __restrict__ Out = Arg->Out;
        unsigned int Norm = Arg->Norm;

	unsigned int CoreId = gap8_coreid();
	unsigned int Chunk = ChunkSize(OutFeatures);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, OutFeatures);

	v4s PadIn = Arg->Pad;
	if (((int)Arg->Pad) && Arg->NTile) {
		PadIn[0] = (FS-1)/2; PadIn[1] = FS/2;
		PadIn[2] *= (Arg->TileIndex==0); PadIn[3] *= (Arg->TileIndex==(Arg->NTile-1));
		H -= (PadIn[2]+PadIn[3]);
	} else if (Arg->NTile==0) {
		BaseInFeature = BaseOutFeature; BaseOutFeature = 0;
		TotalInFeatures = Arg->TileIndex;
	}
	KerPaddedConv_fpT ArgConv = {0, W, H, 0, 0, Norm, FS, S, PadIn};
        unsigned int Wo = (W-FS+PadIn[0]+PadIn[1])/S + 1;
        unsigned int Ho = (H-FS+PadIn[2]+PadIn[3])/S + 1;

	First += BaseOutFeature; Last += BaseOutFeature;
// printf("Core: %d, W: %d, H: %d. Wo: %d, Ho: %d, In Feat: %d, Out Feat: %d .. %d, Pad: %d, %d, %d, %d\n", CoreId, W, H, Wo, Ho, InFeatures, First, Last, PadIn[0], PadIn[1], PadIn[2], PadIn[3]);
	for (unsigned int of=First; of<Last; of++) {
		for (unsigned int If=0; If<InFeatures; If++) {
			ArgConv.In = In+W*H*If; ArgConv.Filter = Filter+FS*FS*(TotalInFeatures*(of-BaseOutFeature) + If + BaseInFeature); ArgConv.Out = Out+Wo*Ho*of;
// printf("Out Feat %d (Base: %d), if: %d (Total: %d, Base: %d)\n", of, BaseOutFeature, If, TotalInFeatures, BaseInFeature);
// DumpPlane("OutIn", ArgConv.Out, Wo, 10, Ho, -1);
			KerConv3x3Stride2_fp(&ArgConv);
// DumpPlane("In", ArgConv.In, ArgConv.W, 10, ArgConv.H, -1);
// DumpPlane("Filter", ArgConv.Filter, FS, 10, FS, 10);
// DumpPlane("Out", ArgConv.Out, Wo, 10, Ho, -1);
		}
	}
	gap8_waitbarrier(0);
}

void KerParConv5x5Stride1_fp(KerParConv_fp_T *Arg)

{
	unsigned int FS=5, S=1;
        short int * __restrict__ In = Arg->In;
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
        unsigned int InFeatures = Arg->InFeatures;
        unsigned int TotalInFeatures = Arg->InFeatures;
        unsigned int OutFeatures = Arg->OutFeatures;
        unsigned int BaseOutFeature = Arg->BaseOutFeature;
        unsigned int BaseInFeature = 0;
        short int * __restrict__ Filter = Arg->Filter;
        short int * __restrict__ Out = Arg->Out;
        unsigned int Norm = Arg->Norm;

	unsigned int CoreId = gap8_coreid();
	unsigned int Chunk = ChunkSize(OutFeatures);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, OutFeatures);

	v4s PadIn = Arg->Pad;
	if (((int)Arg->Pad) && Arg->NTile) {
		PadIn[0] = (FS-1)/2; PadIn[1] = FS/2;
		unsigned int PadContrib = PadIn[2]+PadIn[3];
		unsigned int RemainH = (Arg->TotalSize-PadContrib) - (Arg->TileSize-PadContrib)*Arg->TileIndex-(H-PadContrib);
 		// printf("Tile: %d/%d, TotalH: %d, TileH: %d, CurTileH: %d, Remain: %d\n", Arg->TileIndex, Arg->NTile, Arg->TotalSize, Arg->TileSize, H, RemainH);
		PadIn[2] *= (Arg->TileIndex==0); PadIn[3] *= (Arg->TileIndex==(Arg->NTile-1));
		if (RemainH==1 && Arg->Pad[3]) PadIn[3]=1;
		H -= (PadIn[2]+PadIn[3]);
	} else if (Arg->NTile==0) {
		BaseInFeature = BaseOutFeature; BaseOutFeature = 0;
		TotalInFeatures = Arg->TileIndex;
	}
	KerPaddedConv_fpT ArgConv = {0, W, H, 0, 0, Norm, FS, S, PadIn};
        unsigned int Wo = (W-FS+PadIn[0]+PadIn[1])/S + 1;
        unsigned int Ho = (H-FS+PadIn[2]+PadIn[3])/S + 1;

	First += BaseOutFeature; Last += BaseOutFeature;
   // printf("Core: %d, Wo: %d, Ho: %d, Out Feat: %d .. %d, Pad: %d, %d, %d, %d\n", CoreId, Wo, Ho, First, Last, PadIn[0], PadIn[1], PadIn[2], PadIn[3]);
	for (unsigned int of=First; of<Last; of++) {
		for (unsigned int If=0; If<InFeatures; If++) {
			ArgConv.In = In+W*H*If; ArgConv.Filter = Filter+FS*FS*(TotalInFeatures*(of-BaseOutFeature) + If + BaseInFeature); ArgConv.Out = Out+Wo*Ho*of;
   // printf("Out Feat %d, if: %d, Filter: %X\n", of, If, (ArgConv.Filter));
   // DumpPlane("OutIn", ArgConv.Out, Wo, 10, Ho, -1);
			KerConv5x5Stride1_fp(&ArgConv);
   // DumpPlane("In", ArgConv.In, ArgConv.W, 10, ArgConv.H, -1);
   // DumpPlane("Filter", ArgConv.Filter, FS, 10, FS, 10);
   // DumpPlane("Out", ArgConv.Out, Wo, 10, Ho, -1);
		}
	}
	gap8_waitbarrier(0);
}

/*
	Performs 5x5 convolution with stride 2
	NTile != 0
		At each call fully evaluate a tile (WoxHo) of a group of output features from BaseOutFeature to BaseOutFeature+OutputFeatures-1
		the tile can be horizontal or vertical
		In: A tile of H line from the original input features, each line contains W*InFeatures input data
		Filter: A tile of OutFeatures line of InFeatures*F*F coefficient, this tile starts at BaseOutFeature in the original feature space
		Out: A tile of Ho line from the original output features, each line contains Wo*OutFeatures, in this call [BaseOutFeature..BaseOutFeature+OutFeatures[ are produced
		A call fully evaluates Ho lines out output features from [BaseOutFeature..BaseOutFeature+OutFeatures[
		Padding, if any:
			Horizontal Tile: Left and Right are always padded, Top is padded only for first tile, Bottom is padded only for last tile
			Vertical Tile: Top and Bottom are always padded, Left is padded only for first tile, Right is padded only for last tile
	NTile == 0
		At each call partially evaluate InFeatures out of TotalInputFeatures contributions of OutFeatures full output feature map.
		In: A tile of InFeatures*W*H input features, Tile starts at BaseOutFeature in the original feature space which total size is given by TileIndex
		Filter: OutFeatures lines of TotalInFeatures (passed in TileIndex) FSxFS coefficients
		Out: OutputFeatures lines of WoxHo output features
		Padding: one call evaluate the entire dimension of a feature map so Pad is simply given by the Pad argument
*/
void KerParConv5x5Stride2_fp(KerParConv_fp_T *Arg) {

	unsigned int 				FS				= 5;
	unsigned int				S				= 2;
	short int * __restrict__ 	In				= Arg->In;
	unsigned int 				W				= Arg->W;
	unsigned int 				H				= Arg->H;
	unsigned int 				InFeatures		= Arg->InFeatures;
	unsigned int 				TotalInFeatures = Arg->InFeatures;
	unsigned int 				OutFeatures 	= Arg->OutFeatures;
	unsigned int 				BaseOutFeature 	= Arg->BaseOutFeature;
	unsigned int 				BaseInFeature	= 0;
	short int * __restrict__ 	Filter			= Arg->Filter;
	short int * __restrict__ 	Out				= Arg->Out;
	unsigned int 				Norm			= Arg->Norm;
	unsigned int 				CoreId			= gap8_coreid();
	unsigned int 				Chunk			= ChunkSize(OutFeatures);
	unsigned int 				First			= Chunk*CoreId;
	unsigned int 				Last			= Min(First+Chunk, OutFeatures);
	v4s 						PadIn			= Arg->Pad;

	if (((int)Arg->Pad) && Arg->NTile) {
		if (Arg->Orientation) { // Horizontal
			PadIn[0] = (FS-1)/2; 
			PadIn[1] = FS/2;
			PadIn[2] *= (Arg->TileIndex==0); 
			PadIn[3] *= (Arg->TileIndex==(Arg->NTile-1));
			H -= (PadIn[2]+PadIn[3]);
		} else {
			PadIn[2] = (FS-1)/2; 
			PadIn[3] = FS/2;
			PadIn[0] *= (Arg->TileIndex==0); 
			PadIn[1] *= (Arg->TileIndex==(Arg->NTile-1));
			W -= (PadIn[0]+PadIn[1]);
		}
	} else if (Arg->NTile==0) {
		BaseInFeature = BaseOutFeature; 
		BaseOutFeature = 0;
		TotalInFeatures = Arg->TileIndex;
	}
	KerPaddedConv_fpT ArgConv = {0, W, H, 0, 0, Norm, FS, S, PadIn};
	unsigned int Wo = (W-FS+PadIn[0]+PadIn[1])/S + 1;
	unsigned int Ho = (H-FS+PadIn[2]+PadIn[3])/S + 1;

  	// printf("Core: %d, W: %d, H: %d. Wo: %d, Ho: %d, In Feat: %d, Out Feat: %d .. %d, Pad: %d, %d, %d, %d\n", CoreId, W, H, Wo, Ho, InFeatures, First, Last, PadIn[0], PadIn[1], PadIn[2], PadIn[3]);
	for(unsigned int of=First; of<Last; of++) {
		for(unsigned int If=0; If<InFeatures; If++) {
			ArgConv.In = In+W*H*If;
			ArgConv.Filter = Filter+FS*FS*(TotalInFeatures*of + BaseInFeature + If);
			ArgConv.Out = Out+Wo*Ho*(BaseOutFeature + of);

// 	printf("Core_%d: of: %d, if: %d, Filter: %X %d\n", CoreId, of, If, (ArgConv.Filter), ArgConv.Filter[0]);
//	printf("Ho: %d, Top: %d, Bottom: %d\n", Ho, PadIn[2], PadIn[3]);
//	DumpPlane("OutIn", ArgConv.Out, Wo, 10, Ho, -1);
			KerConv5x5Stride2_fp(&ArgConv);
//   DumpPlane("In", ArgConv.In, ArgConv.W, 10, ArgConv.H, -1);
//   DumpPlane("Filter", ArgConv.Filter, FS, 10, FS, 10);
//   DumpPlane("Out", ArgConv.Out, Wo, 10, Ho, -1);
		}
	}
	gap8_waitbarrier(0);
}

void KerParSetBias_fp(KerParSetBias_fp_T *Arg)

{
        short int * __restrict__ Out = Arg->Out;
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
        unsigned int OutFeatures = Arg->OutFeatures;
        short int * __restrict__ Bias = Arg->Bias;

        unsigned int CoreId = gap8_coreid();
        unsigned int Chunk = ChunkSize(OutFeatures);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, OutFeatures);

        for (unsigned int of=First; of<Last; of++) {
                v2s *LineOut = (v2s *) (Out+W*H*of);
                v2s B = (v2s) {Bias[of], Bias[of]};
                for (unsigned int i=0; i<((W*H)/4); i++) {
                        LineOut[2*i] = B; LineOut[2*i+1] = B;
                }
                for (unsigned int i=(4*((W*H)/4)); i<(W*H); i++) Out[W*H*of+i] = Bias[of];
        }
        gap8_waitbarrier(0);
}

static void KerParDoMaxPool2x2Stride2_fp(
	short int * __restrict__ In,
	unsigned short int W,
	unsigned short int H,
	short int * __restrict__ Out,
	unsigned short int Wo,
	unsigned short int Ho,
	unsigned short int PadR,
	unsigned short int PadB
	)

{

	for (unsigned int i=0; i<(Ho-PadB); i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			v2s M = gap8_max2(Line1[j], Line2[j]);
			Out[Wo*i+j] = Max(M[0], M[1]);
		}
	}
	if (PadR) for (unsigned int i=0; i<(Ho-PadB); i++) Out[Wo*i+Wo-1] = Max(Max(0, In[(2*i)*W+W-1]), In[(2*i+1)*W+W-1]);
	if (PadB) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(H-1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			v2s M = gap8_max2(Line1[j], ((v2s){0,0}));
			Out[Wo*(Ho-1)+j] = Max(M[0], M[1]);
		}
		if (PadR) Out[Wo*Ho-1] = Max(In[W*H-1], 0);
	}
}


static void KerParDoMaxPool2x2Stride2ReLU_fp(
	short int * __restrict__ In,
	unsigned short int W,
	unsigned short int H,
	short int * __restrict__ Out,
	unsigned short int Wo,
	unsigned short int Ho,
	unsigned short int PadR,
	unsigned short int PadB
	)

{

	for (unsigned int i=0; i<(Ho-PadB); i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			v2s M = gap8_max2(Line1[j], Line2[j]);
			Out[Wo*i+j] = Max(0, Max(M[0], M[1]));
		}
	}
	if (PadR) for (unsigned int i=0; i<(Ho-PadB); i++) Out[Wo*i+Wo-1] = Max(Max(0, In[(2*i)*W+W-1]), In[(2*i+1)*W+W-1]);
	if (PadB) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(H-1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			v2s M = gap8_max2(Line1[j], ((v2s){0,0}));
			Out[Wo*(Ho-1)+j] = Max(M[0], M[1]);
		}
		if (PadR) Out[Wo*Ho-1] = Max(In[W*H-1], 0);
	}
}

static void KerParDoAvgPool2x2Stride2_fp(
	short int * __restrict__ In,
	unsigned short int W,
	unsigned short int H,
	short int * __restrict__ Out,
	unsigned short int Wo,
	unsigned short int Ho,
	unsigned short int PadR,
	unsigned short int PadB
	)

{

	for (unsigned int i=0; i<(Ho-PadB); i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			int S = gap8_dotp2(Line1[j], ((v2s) {1,1}));
                        Out[Wo*i+j] = gap8_sumdotp2(Line2[j], ((v2s) {1,1}), S)>>2;

		}
	}
	if (PadR) for (unsigned int i=0; i<(Ho-PadB); i++) Out[Wo*i+Wo-1] = (In[(2*i)*W+W-1]+In[(2*i+1)*W+W-1])>>2;
	if (PadB) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(H-1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			int S = gap8_dotp2(Line1[j], ((v2s) {1,1}));
			Out[Wo*(Ho-1)+j] = S>>2;
		}
		if (PadR) Out[Wo*Ho-1] = In[W*H-1]>>2;
	}
}

static void KerParDoAvgPool2x2Stride2ReLU_fp(
	short int * __restrict__ In,
	unsigned short int W,
	unsigned short int H,
	short int * __restrict__ Out,
	unsigned short int Wo,
	unsigned short int Ho,
	unsigned short int PadR,
	unsigned short int PadB
	)

{

	for (unsigned int i=0; i<(Ho-PadB); i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			int S = gap8_dotp2(Line1[j], ((v2s) {1,1}));
                        Out[Wo*i+j] = Max(0, gap8_sumdotp2(Line2[j], ((v2s) {1,1}), S)>>2);

		}
	}
	if (PadR) for (unsigned int i=0; i<(Ho-PadB); i++) Out[Wo*i+Wo-1] = Max(0, (In[(2*i)*W+W-1]+In[(2*i+1)*W+W-1])>>2);
	if (PadB) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(H-1)*W]);
		for (unsigned int j=0; j<(Wo-PadR); j++) {
			int S = gap8_dotp2(Line1[j], ((v2s) {1,1}));
			Out[Wo*(Ho-1)+j] = Max(0, S>>2);
		}
		if (PadR) Out[Wo*Ho-1] = Max(0, In[W*H-1]>>2);
	}
}

void KerParAvgPool2x2Stride2_fp(KerParReLUMaxPool_fp_T *Arg)

{
        short int * __restrict__ In = Arg->In;
	unsigned int PadR = Arg->Pad[1];
	unsigned int PadB = Arg->Pad[3];
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
	unsigned int Wo = (W-2+PadR)/2+1;
	unsigned int Ho = (H-2+PadB)/2+1;
        unsigned int OutFeatures = Arg->OutFeatures;
        short int * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int Chunk = ChunkSize(OutFeatures);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, OutFeatures);

	PadR = PadR & (W & 0x1);
	PadB = PadB & (H & 0x1);

	if (Arg->DoReLU)
        	for (unsigned int of=First; of<Last; of++) KerParDoAvgPool2x2Stride2ReLU_fp(In+of*W*H, W, H, Out+of*Wo*Ho, Wo, Ho, PadR, PadB);
	else
        	for (unsigned int of=First; of<Last; of++) KerParDoAvgPool2x2Stride2_fp(In+of*W*H, W, H, Out+of*Wo*Ho, Wo, Ho, PadR, PadB);

        gap8_waitbarrier(0);
}

void KerParMaxPool2x2Stride2_fp(KerParReLUMaxPool_fp_T *Arg)

{
        short int * __restrict__ In = Arg->In;
	unsigned int PadR = Arg->Pad[1];
	unsigned int PadB = Arg->Pad[3];
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
	unsigned int Wo = (W-2+PadR)/2+1;
	unsigned int Ho = (H-2+PadB)/2+1;
        unsigned int OutFeatures = Arg->OutFeatures;
        short int * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int Chunk = ChunkSize(OutFeatures);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, OutFeatures);

	PadR = PadR & (W & 0x1);
	PadB = PadB & (H & 0x1);

	if (Arg->DoReLU)
		for (unsigned int of=First; of<Last; of++) KerParDoMaxPool2x2Stride2ReLU_fp(In+of*W*H, W, H, Out+of*Wo*Ho, Wo, Ho, PadR, PadB);
	else
		for (unsigned int of=First; of<Last; of++) KerParDoMaxPool2x2Stride2_fp(In+of*W*H, W, H, Out+of*Wo*Ho, Wo, Ho, PadR, PadB);

        gap8_waitbarrier(0);
}

static void KerParDoReLU_fp(
	short int * __restrict__ In,
	unsigned int W,
	unsigned int H,
	short int * __restrict__ Out
	)

{
        v2s * VectIn  = (v2s *) In;
        v2s * VectOut = (v2s *) Out;

	for (unsigned int i=0; i<((W*H)/4); i++) {
		v2s X = gap8_max2(VectIn[2*i], ((v2s) {0, 0}));
		v2s Y = gap8_max2(VectIn[2*i+1], ((v2s) {0, 0}));
		VectOut[2*i] = X; VectOut[2*i+1] = Y;
	}
	for (unsigned int i=4*((W*H)/4); i<(W*H); i++) Out[i] = Max(In[i], 0);
}

void KerParReLU_fp(KerParReLUMaxPool_fp_T *Arg)

{
        short int * __restrict__ In = Arg->In;
        unsigned int W = Arg->W;
        unsigned int H = Arg->H;
	unsigned int Wo = W;
	unsigned int Ho = H;
        unsigned int OutFeatures = Arg->OutFeatures;
        short int * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int Chunk = ChunkSize(OutFeatures);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, OutFeatures);

        for (unsigned int of=First; of<Last; of++) KerParDoReLU_fp(In+of*W*H, W, H, Out+of*Wo*Ho);

        gap8_waitbarrier(0);
}

//==============================================================================


/*
	Convolution related function. Add two Feature Maps.
	In:	short int * pointer
	Out:	short int * pointer
	W, H:	Size of the output
*/
void __attribute__ ((noinline)) KerAddFM_fp(KerAddFM_fpT *Arg) {

unsigned int CoreId = gap8_coreid();

/****************************** PROFILING KERNEL ******************************/
#ifdef PROFILING_BK
	// // initialize the performance clock
	// if(CoreId==0) {
	// 	rt_perf_init(&perf[CoreId]);
	// 	// Configure performance counters for counting the cycles
	// 	rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	// 	rt_perf_reset(&perf[CoreId]);
	// 	rt_perf_start(&perf[CoreId]);
	// }
#endif
/******************************************************************************/

	Word16 * __restrict__ In 	= Arg->In;
	Word16 * __restrict__ Out 	= Arg->Out;
	int W 						= Arg->W;
	int H 						= Arg->H;

/********************************** CLIPPING **********************************/
	unsigned int ChunkCell 		= ChunkSize(W*H);
	unsigned int First 			= CoreId*ChunkCell;
	unsigned int Last 			= Minu(First+ChunkCell, W*H);
	int i;

	for(i=First; i<Last; i++) {
		Out[i] = gap8_clip(In[i]+Out[i], 15);
	}
/******************************************************************************/

/********************************* VECTOR ADD *********************************/
	// unsigned int CoreId 		= gap8_coreid();
	// unsigned int ChunkCell 		= ChunkSize((W*H)/2);
	// unsigned int First			= CoreId*ChunkCell;
	// unsigned int Last  			= Minu(First+ChunkCell, (W*H)/2);
	// v2s * VectIn  				= (v2s *) In;
	// v2s * VectOut				= (v2s *) Out;
	// int i;

	// for(i=First; i<Last; i++) {
	// 	VectOut[i] = gap8_add2(VectIn[i],VectOut[i]);
	// }

	// if((W*H)&0x1 && Last==(W*H)/2) {
	// 	Out[W*H-1] += Out[W*H-1]+In[W*H-1];
	// }
/******************************************************************************/	

	rt_team_barrier();

/****************************** PROFILING KERNEL ******************************/
#ifdef PROFILING_BK
	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	// 	rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	// 	rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	// 	rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// }
#endif
/******************************************************************************/

}


/*
	Convolution related function. Add two Feature Maps and then Activation.
	In:		short int * pointer
	Out:	short int * pointer
	W, H:	Size of the output
*/
void __attribute__ ((noinline)) KerAddFMReLu_fp(KerAddFM_fpT *Arg) {

/****************************** PROFILING KERNEL ******************************/
	// unsigned int CoreId = gap8_coreid();
	// // initialize the performance clock
	// if(CoreId==0) {
	// 	rt_perf_init(&perf[CoreId]);
	// 	// Configure performance counters for counting the cycles
	// 	rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	// 	rt_perf_reset(&perf[CoreId]);
	// 	rt_perf_start(&perf[CoreId]);
	// }
/******************************************************************************/

#define Max(a, b) (((a)>(b))?(a):(b))

	Word16 * __restrict__ In	= Arg->In;
	Word16 * __restrict__ Out	= Arg->Out;
	int W						= Arg->W;
	int H						= Arg->H;

/********************************** CLIPPING **********************************/
	unsigned int CoreId			= gap8_coreid();
	unsigned int ChunkCell		= ChunkSize(W*H);
	unsigned int First			= CoreId*ChunkCell;
	unsigned int Last			= Minu(First+ChunkCell, W*H);
	int i;

	for (i=First; i<Last; i++) {
		Out[i] = Max(gap8_clip(In[i]+Out[i], 15), 0);
	}
/******************************************************************************/

/********************************* VECTOR ADD *********************************/
	// unsigned int CoreId			= gap8_coreid();
	// unsigned int ChunkCell		= ChunkSize((W*H)/2);
	// unsigned int First			= CoreId*ChunkCell;
	// unsigned int Last			= Minu(First+ChunkCell, (W*H)/2);
	// v2s * VectIn  = (v2s *) In;
	// v2s * VectOut = (v2s *) Out;
	// int i;

	// for (i=First; i<Last; i++) {
	// 	VectOut[i] = gap8_max2(gap8_add2(VectIn[i],VectOut[i]), ((v2s) {0, 0}));
	// }

	// if((W*H)&0x1 && Last==(W*H)/2) {
	// 	Out[W*H-1] = Max(In[W*H-1]+Out[W*H-1], 0);
	// }
/******************************************************************************/

	rt_team_barrier();

#undef Max

/****************************** PROFILING KERNEL ******************************/
	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	// 	rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	// 	rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	// 	rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// }
/******************************************************************************/
}


/*
	Convolution related function. Set the convolution output to the bias value.
	Out:	short int * pointer
	W, H:	Size of the output
	Bias:	short int, bias value.
*/
void __attribute__ ((noinline)) KerSetInBias(KerSetInBiasT *Arg) {

	unsigned int CoreId = gap8_coreid();

#ifdef PROFILING_BK
 //    // initialize the performance clock
 //   	// if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// // }
#endif

	Word16 * __restrict__ Out = Arg->Out;
	Word16 Bias = Arg->Bias;
	int W = Arg->W;
	int H = Arg->H;
    unsigned int ChunkCell = ChunkSize(W*H);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W*H);
	v2s * __restrict__ VectOut = (v2s *) (Out+First);
	int Iter = (Last-First);
	int i;

	for(i=0; i<(Iter/2); i++) 
		VectOut[i] = (v2s) {Bias, Bias};

	if(Iter&0x1) 
		Out[Last-1] = Bias;

	wait_synch_barrier();

#ifdef PROFILING_BK
	// // if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// // }
#endif

}

void __attribute__ ((noinline)) KerSetInBiasPadded(KerSetInBiasT *Arg) {

	unsigned int CoreId = gap8_coreid();

#ifdef PROFILING_BK
 //    // initialize the performance clock
 //   	// if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// // }
#endif

	Word16 * __restrict__ Out = Arg->Out;
	Word16 Bias = Arg->Bias;
	int W = Arg->W;
	int H = Arg->H;
    unsigned int ChunkCell = ChunkSize(W*H);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W*H);
	v2s * __restrict__ VectOut = (v2s *) (Out+First);
	int Iter = (Last-First);
	int i;

	for(i=0; i<(Iter/2); i++) 
		VectOut[i] = (v2s) {Bias, Bias};

	if(Iter&0x1) 
		Out[Last-1] = Bias;

	wait_synch_barrier();

#ifdef PROFILING_BK
	// // if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// // }
#endif

}

/*
	Convolution related function. Set two convolution outputs to their respective bias values.
	Out0:	short int * pointer, first output
	Out1:	short int * pointer, second output
	W, H:	Size of the output, outputs dimension
	Bias0:	short int, bias value for Out0.
	Bias1:	short int, bias value for Out1.
*/

void __attribute__ ((noinline)) KerSetInBias2(KerSetInBias2T *Arg)

{
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;
	Word16 Bias0 = Arg->Bias0;
	Word16 Bias1 = Arg->Bias1;
	int W = Arg->W;
	int H = Arg->H;
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize((W/2)*H);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, (W/2)*H);
	v2s * __restrict__ VectOut0 = (v2s *) (Out0+First);
	v2s * __restrict__ VectOut1 = (v2s *) (Out1+First);
	int Iter = (Last-First);
	int i;

	for (i=0; i<(Iter/2); i++) {
		VectOut0[i] = (v2s) {Bias0, Bias0};
		VectOut1[i] = (v2s) {Bias1, Bias1};
	}
	if (Iter&0x1) {
		Out0[Last-1] = Bias0;
		Out1[Last-1] = Bias1;
	}

	wait_synch_barrier();
}

/*
	Convolution related function. Set three convolution outputs to their respective bias values.
	Out0:	short int * pointer, first output
	Out1:	short int * pointer, second output
	Out2:	short int * pointer, third output
	W, H:	Size of the output, outputs dimension
	Bias0:	short int, bias value for Out0.
	Bias1:	short int, bias value for Out1.
	Bias2:	short int, bias value for Out2.
*/

void __attribute__ ((noinline)) KerSetInBias3(KerSetInBias3T *Arg)

{
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;
	Word16 * __restrict__ Out2 = Arg->Out2;
	Word16 Bias0 = Arg->Bias0;
	Word16 Bias1 = Arg->Bias1;
	Word16 Bias2 = Arg->Bias2;
	int W = Arg->W;
	int H = Arg->H;
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize((W/2)*H);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, (W/2)*H);
	v2s * __restrict__ VectOut0 = (v2s *) (Out0+First);
	v2s * __restrict__ VectOut1 = (v2s *) (Out1+First);
	v2s * __restrict__ VectOut2 = (v2s *) (Out2+First);
	int Iter = (Last-First);
	int i;

	for (i=0; i<(Iter/2); i++) {
		VectOut0[i] = (v2s) {Bias0, Bias0};
		VectOut1[i] = (v2s) {Bias1, Bias1};
		VectOut2[i] = (v2s) {Bias2, Bias2};
	}
	if (Iter&0x1) {
		Out0[Last-1] = Bias0;
		Out1[Last-1] = Bias1;
		Out2[Last-1] = Bias2;
	}

	wait_synch_barrier();
}

/*
	1x1 convolution, short int inputs and output, stride=2 only even lines and columns are produced
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and clipped to 16bits before being written
	In:		short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 1x1 convolution coefficients
	Norm:	Fixed point format

*/
// void __attribute__ ((noinline)) KerConv1x1Stride2_fp(KerConv_fpT *Arg) {

//  //    unsigned int CoreId = gap8_coreid();
//  //    // initialize the performance clock
//  //   	// if(CoreId==0) {
// 	//     rt_perf_init(&perf[CoreId]);
// 	//     // Configure performance counters for counting the cycles
// 	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
// 	//     rt_perf_reset(&perf[CoreId]);
// 	//     rt_perf_start(&perf[CoreId]);
// 	// // }

// 	Word16 * __restrict__ In = Arg->In;
// 	int W = Arg->W;
// 	int H = Arg->H;
// 	Word16 Filter = Arg->Filter[0];
// 	unsigned int Norm = Arg->Norm;
// 	Word16 * __restrict__ Out = Arg->Out;

// 	unsigned int Wo = W/2;
// 	unsigned int Ho = H/2;
//     unsigned int CoreId = gap8_coreid();
//     unsigned int ChunkCell = ChunkSize(Wo);
// 	unsigned int First = CoreId*ChunkCell;
// 	unsigned int Last  = Minu(First+ChunkCell, Wo);
// 	unsigned int i, j;

// 	// v2s C = ((v2s) {Filter, Filter});
// 	// v2s V = ((v2s) {In[0], In[2]});
// 	// int Acc = Out[0];
// 	// Acc = gap8_dotp2(V, C);

// 	// if(CoreId==0) printf("V(%d %d) C(%d %d) Acc(%d)\n",In[0],In[2],Filter,Filter,Acc);

// 	// printf("Core%d - %d %d\n", CoreId, First, Last);

// 	for (j=First; j<Last; j++) {
// 		for(i=0; i<Ho; i++) {
// 			int Acc = Out[i*Wo+j]<<Norm;
// 			Acc += In[(2*i)*W+2*j]*Filter;
// 			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
// 		}
// 	}

// 	wait_synch_barrier();

// 	// // if(CoreId==0) {
// 	// 	rt_perf_stop(&perf[CoreId]);
// 	//     rt_perf_save(&perf[CoreId]);

// 	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
// 	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
// 	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
// 	// // }
// }

/*
	1x1 convolution, short int inputs and output, stride=2 only even lines and columns are produced
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/
void __attribute__ ((noinline)) KerConv1x1Stride2Multi_fp(KerConv_fpT *Arg) {

	// unsigned int CoreId = gap8_coreid();
	// if(CoreId==0) {
	// 	rt_perf_init(&perf[CoreId]);
	// 	rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	// 	rt_perf_reset(&perf[CoreId]);
	// 	rt_perf_start(&perf[CoreId]);
	// }

	int W = Arg->W;
	int H = Arg->H;
	unsigned int Norm = Arg->Norm;
	unsigned int InCh = Arg->InCh;
	Word16 * __restrict__ Out = Arg->Out;

	// // Width and Height of each single input feature map
	unsigned int Wi = W/InCh;
	unsigned int Hi = H;
	// // Width and Height of the output feature map
	unsigned int Wo = Wi/2;
	unsigned int Ho = Hi/2;
	if(Wi&0x1) Wo++;
	if(Hi&0x1) Ho++;
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(InCh);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InCh);
	unsigned int i, j, k;

    Word32 * OutBuff 	= Arg->Reduct 	+ Wo*Ho*CoreId;
	Word16 * Filter		= Arg->Filter 	+ ChunkCell*CoreId;
	Word16 * In			= Arg->In 		+ Wi*Hi*ChunkCell*CoreId;

	for(j=0; j<Wo*Ho; j++) OutBuff[j] = 0;

	for(k=First; k<Last; k++) {

		Word16 C0 = Filter[0];

		for(j=0; j<Wo; j++) {
			for(i=0; i<Ho; i++) {
				int Acc = 0;
				Acc = In[(2*i)*Wi+2*j]*C0;
				OutBuff[i*Wo+j] += Acc;
			}
		}

		Filter 	++;
		In		+= Wi*Hi;
	}


	// REDUCTION
	wait_synch_barrier();

	OutBuff = Arg->Reduct;
    ChunkCell 	= Wo*Ho/8; //ChunkSize(Wo*Ho);
	First 		= CoreId*ChunkCell;
	Last  		= Minu(First+ChunkCell, (Wo*Ho));
	if(CoreId==7) Last++;
	
	int Acc[8];
	for(i=First; i<Last; i++) {
		Acc[CoreId] = Out[i]<<Norm;
		for(j=0; j<gap8_ncore(); j++) {
			Acc[CoreId] += OutBuff[j*Wo*Ho+i];
		}
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc[CoreId], Norm), 15);
	}

	wait_synch_barrier();

	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	// 	rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	// 	rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	// 	rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// }

}

/*
	3x3 convolution, short int inputs and output
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/
void __attribute__ ((noinline)) KerConv3x3_fp(KerConv_fpT *Arg)

{

 //    unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// }

	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;
	v2s C0 = *((v2s *) &Filter[0]),				C1 = *((v2s *) &Filter[3]),
   		C2 = gap8_pack2(Filter[2], Filter[5]), 	C3 = *((v2s *) &Filter[6]),
   		C4 = gap8_pack2(Filter[8], 0);
	unsigned int Wo = W-3+1;
	unsigned int Ho = H-3+1;
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	unsigned int i, j;
	v2s V0, V1, V2, V3, V4;
	v2s *Line;

	for (j=First; j<Last; j++) {

		Line = (v2s *) (&In[j]);
		V1 = *Line++; V2 = *Line; V2 = __builtin_shuffle(V2, (v2s) {1, 0});
		Line = (v2s *) (&In[W+j]);
		V3 = *Line++; V4 = *Line;
		for(i=0; i<Ho; i++) {
			int Acc = Out[i*Wo+j]<<Norm;
			Line = (v2s *) (&In[(i+2)*W+j]);
			V0 = V1; V1 = V3;
			V2 = __builtin_shuffle(V2, V4, (v2s) {1, 2});
			V3 = *Line++; V4 = *Line;
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc);
			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		}
	}
	wait_synch_barrier();

	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// }

}


void __attribute__ ((noinline)) KerConv3x3Multi_fp(KerConv_fpT *Arg) {

 //    unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// }

	// Word16 * __restrict__ In2 = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	// Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	unsigned int InCh = Arg->InCh;
	Word16 * __restrict__ Out = Arg->Out;
	// Word16 * Reduct = Arg->Reduct;

	// Width and Height of each single input feature map
	unsigned int Wi = W/InCh;
	unsigned int Hi = H;
	// Width and Height of the output feature map
	unsigned int Wo = Wi-3+1;
	unsigned int Ho = Hi-3+1;
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(InCh);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InCh);
	unsigned int i, j, k;
	v2s C0, C1,	C2, C3,	C4;
	v2s V0, V1, V2, V3, V4;
	v2s *Line;

    Word32 * OutBuff 	= Arg->Reduct 	+ Wo*Ho*CoreId;
	Word16 * Filter		= Arg->Filter 	+ 9*ChunkCell*CoreId;
	Word16 * In			= Arg->In 		+ Wi*Hi*ChunkCell*CoreId;

	// printf("Coreid%d ADDR 0x%8x 0x%8x 0x%8x 0x%8x\n", CoreId, Arg->Reduct, Arg->Out, OutBuff, Filter);

	for(j=0; j<Wo*Ho; j++) OutBuff[j] = 0;

	// printf("Core%d: %d %d [%d] - %d %d [%d]\n", CoreId, First, Last, InCh, Wo, Ho, ChunkCell);

	// printf("Core%d: %d %d %d %d %d\n", CoreId, Filter[0], Filter[1], Filter[2], Filter[3], Filter[4]);

	for(k=First; k<Last; k++) {

		// if(CoreId==1) printf("Core%d: In %d %d - Filter %d %d\n", CoreId, In[0], In[1], Filter[0], Filter[1]);

		C0 = *((v2s *) &Filter[0]);				
		C1 = *((v2s *) &Filter[3]);
   		C2 = gap8_pack2(Filter[2], Filter[5]); 	
   		C3 = *((v2s *) &Filter[6]);
   		C4 = gap8_pack2(Filter[8], 0);

   		for(j=0; j<Wo; j++) {
			Line = (v2s *) (&In[j]);
			V1 = *Line++; V2 = *Line; V2 = __builtin_shuffle(V2, (v2s) {1, 0});
			Line = (v2s *) (&In[Wi+j]);
			V3 = *Line++; V4 = *Line;
			for(i=0; i<Ho; i++) {
				int Acc = 0;
				Line = (v2s *) (&In[(i+2)*Wi+j]);
				V0 = V1; V1 = V3;
				V2 = __builtin_shuffle(V2, V4, (v2s) {1, 2});
				V3 = *Line++; V4 = *Line;
				Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
				Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
				Acc = gap8_sumdotp2(V4, C4, Acc);
				OutBuff[i*Wo+j] += Acc;
				// if(i==0&&j==0)printf("ACC %d\n",Acc);

   			}
		}

		Filter 	+= 9;
		In		+= Wi*Hi;
	}

	// REDUCTION
	wait_synch_barrier();

	OutBuff = Arg->Reduct;

	// if(CoreId==0) {
	// 	for(i=0; i<Wo*Ho; i++) {
	// 		int Acc = Out[i]<<Norm;
	// 		for(j=0; j<gap8_ncore(); j++) {
	// 			Acc += OutBuff[j*Wo*Ho+i];
	// 		}
	// 		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	// 		printf("1o %d: %d\n", i, Out[i]);
	// 	}
	// }


    ChunkCell 	= ChunkSize(Wo*Ho);
	First 		= CoreId*ChunkCell;
	Last  		= Minu(First+ChunkCell, (Wo*Ho));

	int Acc[8];
	for(i=First; i<Last; i++) {
		Acc[CoreId] = Out[i]<<Norm;
		for(j=0; j<gap8_ncore(); j++) {
			Acc[CoreId] += OutBuff[j*Wo*Ho+i];
		}
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc[CoreId], Norm), 15);
	}

	wait_synch_barrier();

	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// }

}

/*
	3x3 convolution, short int inputs and output, stride=2 only even lines and columns are produced
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/
// void __attribute__ ((noinline)) KerConv3x3Stride2_fp(KerConv_fpT *Arg) {

//  //    unsigned int CoreId = gap8_coreid();
//  //    // initialize the performance clock
//  //   	if(CoreId==0) {
// 	//     rt_perf_init(&perf[CoreId]);
// 	//     // Configure performance counters for counting the cycles
// 	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
// 	//     rt_perf_reset(&perf[CoreId]);
// 	//     rt_perf_start(&perf[CoreId]);
// 	// }


// 	Word16 * __restrict__ In = Arg->In;
// 	int W = Arg->W;
// 	int H = Arg->H;
// 	Word16 *Filter = Arg->Filter;
// 	unsigned int Norm = Arg->Norm;
// 	Word16 * __restrict__ Out = Arg->Out;
// 	v2s C0 = *((v2s *) &Filter[0]), C1 = gap8_pack2(Filter[2], 0),
// 	    C2 = *((v2s *) &Filter[3]),	C3 = gap8_pack2(Filter[5], 0),
// 	    C4 = *((v2s *) &Filter[6]),	C5 = gap8_pack2(Filter[8], 0);
// 	unsigned int Wo = (W-3+1)/2;
// 	if((W-3+1)&0x1) Wo++;
// 	unsigned int Ho = (H-3+1)/2;
// 	if((H-3+1)&0x1) Ho++;
//     unsigned int CoreId = gap8_coreid();
//     unsigned int ChunkCell = ChunkSize(Wo);
// 	unsigned int First = CoreId*ChunkCell;
// 	unsigned int Last  = Minu(First+ChunkCell, Wo);
// 	unsigned int i, j;
// 	v2s V0, V1, V2, V3, V4, V5;
// 	v2s *Line;

// 	// for(j=0; j<Wo; j++) {
// 	for (j=First; j<Last; j++) {
// 		Line = (v2s *) (&In[2*j]);
// 		V4 = Line[0]; V5 = Line[1];
// 		for(i=0; i<Ho; i++) {
// 			int Acc = Out[i*Wo+j]<<Norm;
// 			V0 = V4; V1 = V5;
// 			Line = (v2s *) (&In[(2*i+1)*W+2*j]);
// 			V2 = Line[0]; V3 = Line[1];
// 			Line = (v2s *) (&In[(2*i+2)*W+2*j]);
// 			V4 = Line[0]; V5 = Line[1];
// 			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
// 			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
// 			Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
// 			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
// 		}
// 	}
// 	wait_synch_barrier();

// 	// if(CoreId==0) {
// 	// 	rt_perf_stop(&perf[CoreId]);
// 	//     rt_perf_save(&perf[CoreId]);

// 	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
// 	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
// 	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
// 	// }

// }


/*
	3x3 convolution, short int inputs and output, stride=2 only even lines and columns are produced
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/
void __attribute__ ((noinline)) KerConv3x3Stride2Multi_fp(KerConv_fpT *Arg) {

 //    unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	// if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// // }

	int W = Arg->W;
	int H = Arg->H;
	unsigned int Norm = Arg->Norm;
	unsigned int InCh = Arg->InCh;
	Word16 * __restrict__ Out = Arg->Out;

	// // Width and Height of each single input feature map
	unsigned int Wi = W/InCh;
	unsigned int Hi = H;
	// // Width and Height of the output feature map
	unsigned int Wo = (Wi-3+1)/2;
	unsigned int Ho = (Hi-3+1)/2;
	if((Wi-3+1)&0x1) Wo++;
	if((Hi-3+1)&0x1) Ho++;
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(InCh);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InCh);
	unsigned int i, j, k;
	v2s C0, C1,	C2, C3,	C4, C5;
	v2s V0, V1, V2, V3, V4, V5;
	v2s *Line;

    Word32 * OutBuff 	= Arg->Reduct 	+ Wo*Ho*CoreId;
	Word16 * Filter		= Arg->Filter 	+ 9*ChunkCell*CoreId;
	Word16 * In			= Arg->In 		+ Wi*Hi*ChunkCell*CoreId;

	// // printf("Coreid%d ADDR 0x%8x 0x%8x 0x%8x 0x%8x\n", CoreId, Arg->Reduct, Arg->Out, OutBuff, Filter);

	for(j=0; j<Wo*Ho; j++) OutBuff[j] = 0;

	// printf("Core%d: %d/%d %d/%d %d/%d %d %d [%d %d]\n", CoreId, W, H, Wi, Hi, Wo, Ho, InCh, ChunkCell, First, Last);

	// // printf("Core%d: %d %d %d %d %d\n", CoreId, Filter[0], Filter[1], Filter[2], Filter[3], Filter[4]);

	for(k=First; k<Last; k++) {

	// 	// if(CoreId==1) printf("Core%d: In %d %d - Filter %d %d\n", CoreId, In[0], In[1], Filter[0], Filter[1]);

   		C0 = *((v2s *) &Filter[0]);
		C1 = gap8_pack2(Filter[2], 0);
		C2 = *((v2s *) &Filter[3]);
		C3 = gap8_pack2(Filter[5], 0);
		C4 = *((v2s *) &Filter[6]);
		C5 = gap8_pack2(Filter[8], 0);

		for(j=0; j<Wo; j++) {
			Line = (v2s *) (&In[2*j]);
			V4 = Line[0]; V5 = Line[1];
			for(i=0; i<Ho; i++) {
				int Acc = 0;
				V0 = V4; V1 = V5;
				Line = (v2s *) (&In[(2*i+1)*Wi+2*j]);
				V2 = Line[0]; V3 = Line[1];
				Line = (v2s *) (&In[(2*i+2)*Wi+2*j]);
				V4 = Line[0]; V5 = Line[1];
				Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
				Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
				Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
				OutBuff[i*Wo+j] += Acc;
				// if(i==0&&j==0)printf("ACC %d\n",Acc);

			}
		}

		Filter 	+= 9;
		In		+= Wi*Hi;
	}

	// REDUCTION
	wait_synch_barrier();

	OutBuff = Arg->Reduct;

	// if(CoreId==0) {
	// 	for(i=0; i<Wo*Ho; i++) {
	// 		int Acc = Out[i]<<Norm;
	// 		for(j=0; j<gap8_ncore(); j++) {
	// 			Acc += OutBuff[j*Wo*Ho+i];
	// 		}
	// 		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
	// 		printf("1o %d: %d\n", i, Out[i]);
	// 	}
	// }


    ChunkCell 	= ChunkSize(Wo*Ho);
	First 		= CoreId*ChunkCell;
	Last  		= Minu(First+ChunkCell, (Wo*Ho));

	int Acc[8];
	for(i=First; i<Last; i++) {
		Acc[CoreId] = Out[i]<<Norm;
		for(j=0; j<gap8_ncore(); j++) {
			Acc[CoreId] += OutBuff[j*Wo*Ho+i];
		}
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc[CoreId], Norm), 15);
	}

	wait_synch_barrier();

	// // if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// // }
}


void __attribute__ ((noinline)) KerReLUConv3x3Stride2_fp(KerConv_fpT *Arg) {

    Word16 * __restrict__ In = Arg->In;
    int W = Arg->W;
    int H = Arg->H;
    Word16 *Filter = Arg->Filter;
    unsigned int Norm = Arg->Norm;
    Word16 * __restrict__ Out = Arg->Out;
    v2s C0 = *((v2s *) &Filter[0]), C1 = gap8_pack2(Filter[2], 0),
        C2 = *((v2s *) &Filter[3]), C3 = gap8_pack2(Filter[5], 0),
        C4 = *((v2s *) &Filter[6]), C5 = gap8_pack2(Filter[8], 0);
    unsigned int Wo = (W-3)/2 + 1;
    unsigned int Ho = (H-3)/2 + 1;
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(Wo);
    unsigned int First = CoreId*ChunkCell;
    unsigned int Last  = Minu(First+ChunkCell, Wo);
    unsigned int i, j;
    v2s V0, V1, V2, V3, V4, V5;
    v2s *Line;

    // for(j=0; j<Wo; j++) {
    for (j=First; j<Last; j++) {
        Line = (v2s *) (&In[2*j]);
        V4 = gap8_max2(Line[0], ((v2s) {0, 0})); V5 = gap8_max2(Line[1], ((v2s) {0, 0}));
        // V4 = Line[0]; V5 = Line[1];

        for(i=0; i<Ho; i++) {
            int Acc = Out[i*Wo+j]<<Norm;
            V0 = V4; V1 = V5;
            Line = (v2s *) (&In[(2*i+1)*W+2*j]);
            V2 = gap8_max2(Line[0], ((v2s) {0, 0})); V3 = gap8_max2(Line[1], ((v2s) {0, 0}));
            // V2 = Line[0]; V3 = Line[1];

            Line = (v2s *) (&In[(2*i+2)*W+2*j]);
            V4 = gap8_max2(Line[0], ((v2s) {0, 0})); V5 = gap8_max2(Line[1], ((v2s) {0, 0}));
            // V4 = Line[0]; V5 = Line[1];
            Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
            Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
            Acc = gap8_sumdotp2(V4, C4, Acc); Acc = gap8_sumdotp2(V5, C5, Acc);
            Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
        }
    }
    wait_synch_barrier();
}

void __attribute__ ((noinline)) KerReLUConv3x3_fp(KerConv_fpT *Arg) {

	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;
	v2s C0 = *((v2s *) &Filter[0]),				C1 = *((v2s *) &Filter[3]),
   		C2 = gap8_pack2(Filter[2], Filter[5]), 	C3 = *((v2s *) &Filter[6]),
   		C4 = gap8_pack2(Filter[8], 0);
	unsigned int Wo = W-3+1;
	unsigned int Ho = H-3+1;
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	unsigned int i, j;
	v2s V0, V1, V2, V3, V4;
	v2s *Line;

	// for(j=0; j<Wo; j++) {
	for (j=First; j<Last; j++) {

		Line = (v2s *) (&In[j]);
		// V1 = *Line++; V2 = *Line; 
		V1 = gap8_max2(*Line++, ((v2s) {0, 0})); V2 = gap8_max2(*Line, ((v2s) {0, 0})); 
		V2 = __builtin_shuffle(V2, (v2s) {1, 0});
		Line = (v2s *) (&In[W+j]);
		// V3 = *Line++; V4 = *Line;
		V3 = gap8_max2(*Line++, ((v2s) {0, 0})); V4 = gap8_max2(*Line, ((v2s) {0, 0})); 
		for(i=0; i<Ho; i++) {
			int Acc = Out[i*Wo+j]<<Norm;
			Line = (v2s *) (&In[(i+2)*W+j]);
			V0 = V1; V1 = V3;
			V2 = __builtin_shuffle(V2, V4, (v2s) {1, 2});
			// V3 = *Line++; V4 = *Line;
			V3 = gap8_max2(*Line++, ((v2s) {0, 0})); V4 = gap8_max2(*Line, ((v2s) {0, 0})); 
			Acc = gap8_sumdotp2(V0, C0, Acc); Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc);
			Out[i*Wo+j] = gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
		}
	}
	wait_synch_barrier();
}

/*
	3x3 convolution, short int inputs and output
	The result of the convolution is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/
void __attribute__ ((noinline)) KerDirectConv3x3_fp(KerConv_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;
	v2s C0 = *((v2s *) &Filter[0]),            C1 = *((v2s *) &Filter[3]),
    	    C2 = gap8_pack2(Filter[2], Filter[5]), C3 = *((v2s *) &Filter[6]),
    	    C4 = gap8_pack2(Filter[8], 0);
	unsigned int Wo = W-3+1;
	unsigned int Ho = H-3+1;
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	unsigned int i, j;
	v2s V0, V1, V2, V3, V4;
	v2s *Line;

	// for(j=0; j<Wo; j++) {
	for (j=First; j<Last; j++) {

		Line = (v2s *) (&In[j]);
		V1 = *Line++; V2 = *Line; V2 = __builtin_shuffle(V2, (v2s) {1, 0});
		Line = (v2s *) (&In[W+j]);
		V3 = *Line++; V4 = *Line;
		for(i=0; i<Ho; i++) {
			int Acc;
			Line = (v2s *) (&In[(i+2)*W+j]);
			V0 = V1; V1 = V3;
			V2 = __builtin_shuffle(V2, V4, (v2s) {1, 2});
			V3 = *Line++; V4 = *Line;
			Acc = gap8_dotp2(V0, C0);         Acc = gap8_sumdotp2(V1, C1, Acc);
			Acc = gap8_sumdotp2(V2, C2, Acc); Acc = gap8_sumdotp2(V3, C3, Acc);
			Acc = gap8_sumdotp2(V4, C4, Acc);
			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
			// Out[i*Wo+j] =  gap8_clip((Acc>>Norm), 15);
		}
	}
	wait_synch_barrier();
}

/*
	5x5 convolution, short int inputs and output
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/

void __attribute__ ((noinline)) KerConv5x5_fp(KerConv_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;
	v2s C0  = *((v2s *) &Filter[0]),            C1  = *((v2s *) &Filter[2]),
	    C2  = *((v2s *) &Filter[5]),            C3  = *((v2s *) &Filter[7]),
	    C4  = *((v2s *) &Filter[10]),           C5  = *((v2s *) &Filter[12]),
	    C6  = *((v2s *) &Filter[15]),           C7  = *((v2s *) &Filter[17]),
	    C8  = *((v2s *) &Filter[20]),           C9  = *((v2s *) &Filter[22]),
    	    C10 = gap8_pack2(Filter[4], Filter[9]), C11 = gap8_pack2(Filter[14], Filter[19]),
    	    C12 = gap8_pack2(Filter[24], 0);

	unsigned int Wo = W-5+1;
	unsigned int Ho = H-5+1;
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	unsigned int i, j;
	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12;
	v2s Mask  = {1,2};
	v2s *Line;

	// for(j=0; j<Wo; j++) {
	for (j=First; j<Last; j++) {
		V10 = (v2s) {0, In[j+4]}; V11 = (v2s) {In[j+W+4], In[j+2*W+4]}; 	Line = (v2s *) (In + j);
		V2 = Line[0]; V3 = Line[1]; 						Line = (v2s *) (In + W + j);
		V4 = Line[0]; V5 = Line[1]; 						Line = (v2s *) (In + 2*W + j);
		V6 = Line[0]; V7 = Line[1]; 						Line = (v2s *) (In + 3*W + j);
		V8 = Line[0]; V9 = Line[1]; V12 = Line[2];
		for(i=0; i<Ho; i++) {
			int S = Out[i*Wo+j]<<Norm;
			V0 = V2; V1 = V3; V2 = V4; V3 = V5; V4 = V6; V5 = V7; V6 = V8; V7 = V9;
			V10 = __builtin_shuffle(V10, V11, Mask); V11 = __builtin_shuffle(V11, V12, Mask);
			Line = (v2s *) (&In[(i+4)*W+j]);
			V8 = Line[0]; V9 = Line[1]; V12 = Line[2];
			S = gap8_sumdotp2(V0,  C0,  S);
			S = gap8_sumdotp2(V1,  C1,  S); S = gap8_sumdotp2(V2,  C2,  S);
			S = gap8_sumdotp2(V3,  C3,  S); S = gap8_sumdotp2(V4,  C4,  S);
			S = gap8_sumdotp2(V5,  C5,  S); S = gap8_sumdotp2(V6,  C6,  S);
			S = gap8_sumdotp2(V7,  C7,  S); S = gap8_sumdotp2(V8,  C8,  S);
			S = gap8_sumdotp2(V9,  C9,  S); S = gap8_sumdotp2(V10, C10, S);
			S = gap8_sumdotp2(V11, C11, S); S = gap8_sumdotp2(V12, C12, S);
			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
			// Out[i*Wo+j] =  gap8_clip((S>>Norm), 15);
		}
	}
	wait_synch_barrier();
}



/*
	5x5 convolution, short int inputs and output, stride=2 only even lines and columns are produced
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/

// void __attribute__ ((noinline)) KerConv5x5Stride2_fp(KerConv_fpT *Arg)

// {

    
//  //    unsigned int CoreId = gap8_coreid();
//  //    // initialize the performance clock
//  //   	if(CoreId==0) {
//  //    rt_perf_init(&perf[CoreId]);
//  //    // Configure performance counters for counting the cycles
//  //    rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
//  //    rt_perf_reset(&perf[CoreId]);
//  //    rt_perf_start(&perf[CoreId]);
// 	// }


// 	Word16 * __restrict__ In = Arg->In;
// 	int W = Arg->W;
// 	int H = Arg->H;
// 	Word16 *Filter = Arg->Filter;
// 	unsigned int Norm = Arg->Norm;
// 	Word16 * __restrict__ Out = Arg->Out;
// 	v2s C0  = *((v2s *) &Filter[0]),            C1  = *((v2s *) &Filter[2]),
// 	    C2  = *((v2s *) &Filter[5]),            C3  = *((v2s *) &Filter[7]),
// 	    C4  = *((v2s *) &Filter[10]),           C5  = *((v2s *) &Filter[12]),
// 	    C6  = *((v2s *) &Filter[15]),           C7  = *((v2s *) &Filter[17]),
// 	    C8  = *((v2s *) &Filter[20]),           C9  = *((v2s *) &Filter[22]),
//     	C10 = gap8_pack2(Filter[4], Filter[9]), C11 = gap8_pack2(Filter[14], Filter[19]),
//     	C12 = gap8_pack2(Filter[24], 0);

// 	unsigned int Wo = (W-5+1)/2;
// 	unsigned int Ho = (H-5+1)/2;
//     unsigned int CoreId = gap8_coreid();
//     unsigned int ChunkCell = ChunkSize(Wo);
// 	unsigned int First = CoreId*ChunkCell;
// 	unsigned int Last  = Minu(First+ChunkCell, Wo);
// 	unsigned int i, j;
// 	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12;
// 	v2s Mask  = {0,2};
// 	v2s *Line;

// 	// printf("Core%d %d - %d %d %d\n", CoreId, First, Last, Ho, Wo);

// 	// if(CoreId==0)
// 	// for(j=0; j<3;j++) {
// 	// 	for(i=0; i<204; i++) printf("%d ", In[j*204+i]);
// 	// 		printf("\n");
// 	// }
// 	// wait_synch_barrier();


// 	// for(j=0; j<Wo; j++) {
// 	for (j=First; j<Last; j++) {
// 		Line = (v2s *) (&In[2*j]);
// 		V4 = Line[0]; V5 = Line[1]; V11 = Line[2];
// 		Line = (v2s *) (&In[W + 2*j]);
// 		V6 = Line[0]; V7 = Line[1]; V0 = Line[2];
// 		V11 = __builtin_shuffle(V11, V0, Mask);
// 		Line = (v2s *) (&In[2*W + 2*j]);
// 		V8 = Line[0]; V9 = Line[1]; V12 = Line[2];
// 		for(i=0; i<Ho; i++) {
// 			v2s X;
// 			int S = Out[i*Wo+j]<<Norm;
// 			V0 = V4; V1 = V5;
// 			V2 = V6; V3 = V7;
// 			V4 = V8; V5 = V9;
// 			V10 = V11;
// 			Line = (v2s *) (&In[(2*(i+1)+1)*W+2*j]);
// 			V6 = Line[0]; V7 = Line[1]; X = Line[2];
// 			Line = (v2s *) (&In[(2*(i+2))*W+2*j]);
// 			V8 = Line[0]; V9 = Line[1];
// 			V11 = __builtin_shuffle(V12, X, Mask);
// 			V12 = Line[2];

// 			// if(j==99 && i==0) {
// 			// 	// printf("%d %d %d %d %d %d %d %d %d %d %d %d %d\n",C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12);
// 			// 	printf("%d %d %d %d %d\n", 	(short int)V0[0], (short int)V0[1], 
// 			// 								(short int)V1[0], (short int)V1[1], 
// 			// 								(short int)V10[0]);
// 			// 	printf("%d %d %d %d %d\n", 	(short int)V2[0], (short int)V2[1], 
// 			// 								(short int)V3[0], (short int)V3[1], 
// 			// 								(short int)V10[1]);
// 			// 	printf("%d %d %d %d %d\n", 	(short int)V4[0], (short int)V4[1], 
// 			// 								(short int)V5[0], (short int)V5[1], 
// 			// 								(short int)V11[0]);							
// 			// 	printf("%d %d %d %d %d\n", 	(short int)V6[0], (short int)V6[1], 
// 			// 								(short int)V7[0], (short int)V7[1], 
// 			// 								(short int)V11[1]);
// 			// 	printf("%d %d %d %d %d\n", 	(short int)V8[0], (short int)V8[1], 
// 			// 								(short int)V9[0], (short int)V9[1], 
// 			// 								(short int)V12[0]);								
// 			// }
// 			S = gap8_sumdotp2(V0,  C0,  S);
// 			S = gap8_sumdotp2(V1,  C1,  S); S = gap8_sumdotp2(V2,  C2,  S);
// 			S = gap8_sumdotp2(V3,  C3,  S); S = gap8_sumdotp2(V4,  C4,  S);
// 			S = gap8_sumdotp2(V5,  C5,  S); S = gap8_sumdotp2(V6,  C6,  S);
// 			S = gap8_sumdotp2(V7,  C7,  S); S = gap8_sumdotp2(V8,  C8,  S);
// 			S = gap8_sumdotp2(V9,  C9,  S); S = gap8_sumdotp2(V10, C10, S);
// 			S = gap8_sumdotp2(V11, C11, S); S = gap8_sumdotp2(V12, C12, S);
// 			Out[i*Wo+j] = gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
// 			// if(j==99 && i==0) {
// 			// 	printf("out %d\n", Out[i*Wo+j]);
// 			// }

// 		}
// 	}


// 	wait_synch_barrier();

// // if(CoreId==0) {
// // 	rt_perf_stop(&perf[CoreId]);
// //     rt_perf_save(&perf[CoreId]);

// // 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
// //         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
// //         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
// // }
// }

void __attribute__ ((noinline)) KerConv5x5Stride2Padded_fp(KerConv_fpT *Arg)

{

    
 //    unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	if(CoreId==0) {
 //    rt_perf_init(&perf[CoreId]);
 //    // Configure performance counters for counting the cycles
 //    rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
 //    rt_perf_reset(&perf[CoreId]);
 //    rt_perf_start(&perf[CoreId]);
	// }


	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;
	v2s C0  = *((v2s *) &Filter[0]),            C1  = *((v2s *) &Filter[2]),
	    C2  = *((v2s *) &Filter[5]),            C3  = *((v2s *) &Filter[7]),
	    C4  = *((v2s *) &Filter[10]),           C5  = *((v2s *) &Filter[12]),
	    C6  = *((v2s *) &Filter[15]),           C7  = *((v2s *) &Filter[17]),
	    C8  = *((v2s *) &Filter[20]),           C9  = *((v2s *) &Filter[22]),
    	C10 = gap8_pack2(Filter[4], Filter[9]), C11 = gap8_pack2(Filter[14], Filter[19]),
    	C12 = gap8_pack2(Filter[24], 0);

	unsigned int Wo = (W-5+1)/2;
	unsigned int Ho = (H-5+1)/2;
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	unsigned int i, j;
	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12;
	v2s Mask  = {0,2};
	v2s *Line;

	// printf("Core%d %d - %d %d %d\n", CoreId, First, Last, Ho, Wo);

	// if(CoreId==0)
	// for(j=0; j<3;j++) {
	// 	for(i=0; i<204; i++) printf("%d ", In[j*204+i]);
	// 		printf("\n");
	// }
	// wait_synch_barrier();


	// for(j=0; j<Wo; j++) {
	for (j=First; j<Last; j++) {
		Line = (v2s *) (&In[2*j]);
		V4 = Line[0]; V5 = Line[1]; V11 = Line[2];
		Line = (v2s *) (&In[W + 2*j]);
		V6 = Line[0]; V7 = Line[1]; V0 = Line[2];
		V11 = __builtin_shuffle(V11, V0, Mask);
		Line = (v2s *) (&In[2*W + 2*j]);
		V8 = Line[0]; V9 = Line[1]; V12 = Line[2];
		for(i=0; i<Ho; i++) {
			v2s X;
			// offset for Out var
			int S = Out[i*Wo+j]<<Norm;
			V0 = V4; V1 = V5;
			V2 = V6; V3 = V7;
			V4 = V8; V5 = V9;
			V10 = V11;
			Line = (v2s *) (&In[(2*(i+1)+1)*W+2*j]);
			V6 = Line[0]; V7 = Line[1]; X = Line[2];
			Line = (v2s *) (&In[(2*(i+2))*W+2*j]);
			V8 = Line[0]; V9 = Line[1];
			V11 = __builtin_shuffle(V12, X, Mask);
			V12 = Line[2];

			S = gap8_sumdotp2(V0,  C0,  S);
			S = gap8_sumdotp2(V1,  C1,  S); S = gap8_sumdotp2(V2,  C2,  S);
			S = gap8_sumdotp2(V3,  C3,  S); S = gap8_sumdotp2(V4,  C4,  S);
			S = gap8_sumdotp2(V5,  C5,  S); S = gap8_sumdotp2(V6,  C6,  S);
			S = gap8_sumdotp2(V7,  C7,  S); S = gap8_sumdotp2(V8,  C8,  S);
			S = gap8_sumdotp2(V9,  C9,  S); S = gap8_sumdotp2(V10, C10, S);
			S = gap8_sumdotp2(V11, C11, S); S = gap8_sumdotp2(V12, C12, S);
			Out[i*Wo+j] = gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
		}
	}


	wait_synch_barrier();

// if(CoreId==0) {
// 	rt_perf_stop(&perf[CoreId]);
//     rt_perf_save(&perf[CoreId]);

// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
// }
}


/*
	5x5 convolution, short int inputs and output
	The result of the convolution is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the 9 convolution coefficients
	Norm:	Fixed point format
*/
void __attribute__ ((noinline)) KerDirectConv5x5_fp(KerConv_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;
	v2s C0  = *((v2s *) &Filter[0]),            C1  = *((v2s *) &Filter[2]),
	    C2  = *((v2s *) &Filter[5]),            C3  = *((v2s *) &Filter[7]),
	    C4  = *((v2s *) &Filter[10]),           C5  = *((v2s *) &Filter[12]),
	    C6  = *((v2s *) &Filter[15]),           C7  = *((v2s *) &Filter[17]),
	    C8  = *((v2s *) &Filter[20]),           C9  = *((v2s *) &Filter[22]),
    	    C10 = gap8_pack2(Filter[4], Filter[9]), C11 = gap8_pack2(Filter[14], Filter[19]),
    	    C12 = gap8_pack2(Filter[24], 0);

	unsigned int Wo = W-5+1;
	unsigned int Ho = H-5+1;
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	unsigned int i, j;
	v2s V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12;
	v2s Mask  = {1,2};
	v2s *Line;

	// for(j=0; j<Wo; j++) {
	for (j=First; j<Last; j++) {
		V10 = (v2s) {0, In[j+4]}; V11 = (v2s) {In[j+W+4], In[j+2*W+4]}; 	Line = (v2s *) (In + j);
		V2 = Line[0]; V3 = Line[1]; 						Line = (v2s *) (In + W + j);
		V4 = Line[0]; V5 = Line[1]; 						Line = (v2s *) (In + 2*W + j);
		V6 = Line[0]; V7 = Line[1]; 						Line = (v2s *) (In + 3*W + j);
		V8 = Line[0]; V9 = Line[1]; V12 = Line[2];
		for(i=0; i<Ho; i++) {
			int S;
			V0 = V2; V1 = V3; V2 = V4; V3 = V5; V4 = V6; V5 = V7; V6 = V8; V7 = V9;
			V10 = __builtin_shuffle(V10, V11, Mask); V11 = __builtin_shuffle(V11, V12, Mask);
			Line = (v2s *) (&In[(i+4)*W+j]);
			V8 = Line[0]; V9 = Line[1]; V12 = Line[2];
			S = gap8_dotp2   (V0,  C0);
			S = gap8_sumdotp2(V1,  C1,  S); S = gap8_sumdotp2(V2,  C2,  S);
			S = gap8_sumdotp2(V3,  C3,  S); S = gap8_sumdotp2(V4,  C4,  S);
			S = gap8_sumdotp2(V5,  C5,  S); S = gap8_sumdotp2(V6,  C6,  S);
			S = gap8_sumdotp2(V7,  C7,  S); S = gap8_sumdotp2(V8,  C8,  S);
			S = gap8_sumdotp2(V9,  C9,  S); S = gap8_sumdotp2(V10, C10, S);
			S = gap8_sumdotp2(V11, C11, S); S = gap8_sumdotp2(V12, C12, S);
			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(S, Norm), 15);
			// Out[i*Wo+j] =  gap8_clip((S>>Norm), 15);
		}
	}
	wait_synch_barrier();
}

/*
	NxN convolution, short int inputs and output
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the NxN convolution coefficients
	Norm:	Fixed point format

*/
void __attribute__ ((noinline)) KerConvNxN_fp(KerConv_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int N = Arg->N;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;

	unsigned int Wo = W-N+1;
	unsigned int Ho = H-N+1;

	unsigned int CoreId = gap8_coreid();
	unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);

	unsigned int i, j;
	// for(j=0; j<Wo; j++) {
	for(i=0; i<Ho; i++) {
		for(j=First; j<Last; j++) {
			int ii, jj;
			int Acc = Out[i*Wo+j]<<Norm;
			for (ii=0; ii<N; ii++) {
				v2s *Coeff = (v2s *) (Filter + ii*N);
				v2s *Line = (v2s *) (In + (i+ii)*W + j);
				for (jj=0; jj<(N/2); jj++) Acc = gap8_sumdotp2(Coeff[jj], Line[jj], Acc);
				Acc += Filter[ii*N+(N-1)]*In[(i+ii)*W + j + (N-1)];
				// for (jj=0; jj<N; jj++) Acc += Filter[ii*N+jj]*In[(i+ii)*W + j + jj];
			}
			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
			// Out[i*Wo+j] =  gap8_clip((Acc>>Norm), 15);
		}
	}
	wait_synch_barrier();
}

/*
	NxN convolution, short int inputs and output, stride=2 only even lines and columns are produced
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the NxN convolution coefficients
	Norm:	Fixed point format

*/
void __attribute__ ((noinline)) KerConvNxNStride2_fp(KerConv_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int N = Arg->N;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int Wo = (W-N+1)/2;
        unsigned int Ho = (H-N+1)/2;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);

        unsigned int i, j;
        // for(j=0; j<Wo; j++) {
        for(i=0; i<Ho; i++) {
        	for(j=First; j<Last; j++) {
                        int ii, jj;
			int Acc = Out[i*Wo+j]<<Norm;
                        for (ii=0; ii<N; ii++) {
				v2s *Coeff = (v2s *) (Filter + ii*N);
				v2s *Line = (v2s *) (In + (2*i+ii)*W + 2*j);
				for (jj=0; jj<(N/2); jj++) Acc = gap8_sumdotp2(Coeff[jj], Line[jj], Acc);
				Acc += Filter[ii*N+(N-1)]*In[(2*i+ii)*W + 2*j + (N-1)];
                                // for (jj=0; jj<N; jj++) Acc += Filter[ii*N+jj]*In[(i+ii)*W + j + jj];
                        }
			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
                        // Out[i*Wo+j] =  gap8_clip((Acc>>Norm), 15);
                }
        }
	wait_synch_barrier();
}

/*
	NxN convolution, short int inputs and output, stride=M
	The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the NxN convolution coefficients
	Norm:	Fixed point format

*/
void __attribute__ ((noinline)) KerConvNxNStrideM_fp(KerConv_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int N = Arg->N;
	int Stride = Arg->Stride;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;

    unsigned int Wo = (W-N+1)/Stride;
    unsigned int Ho = (H-N+1)/Stride;

    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);

    unsigned int i, j;
    // for(j=0; j<Wo; j++) {
    for(i=0; i<Ho; i++) {
    	for(j=First; j<Last; j++) {
                    int ii, jj;
		int Acc = Out[i*Wo+j]<<Norm;
                    for (ii=0; ii<N; ii++) {
			v2s *Coeff = (v2s *) (Filter + ii*N);
			v2s *Line = (v2s *) (In + (Stride*i+ii)*W + Stride*j);
			for (jj=0; jj<(N/2); jj++) Acc = gap8_sumdotp2(Coeff[jj], Line[jj], Acc);
			Acc += Filter[ii*N+(N-1)]*In[(Stride*i+ii)*W + Stride*j + (N-1)];
                    }
		Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
                    // Out[i*Wo+j] =  gap8_clip((Acc>>Norm), 15);
            }
    }

	wait_synch_barrier();
}

/*
	NxN convolution, short int inputs and output
	The result of the convolution is rounded, normalized and cliped to 16bits before beeing written
	In:	short int *, convolution input
	W, H:	Input dimension [W x H]
	Filter:	short int *, pointer to the NxN convolution coefficients
	Norm:	Fixed point format
*/
void __attribute__ ((noinline)) KerDirectConvNxN_fp(KerConv_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int N = Arg->N;
	Word16 *Filter = Arg->Filter;
	unsigned int Norm = Arg->Norm;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int Wo = W-N+1;
        unsigned int Ho = H-N+1;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);

        unsigned int i, j;
        // for(j=0; j<Wo; j++) {
        for(i=0; i<Ho; i++) {
        	for(j=First; j<Last; j++) {
                        int ii, jj;
                        int Acc = 0;
                        for (ii=0; ii<N; ii++) {
				v2s *Coeff = (v2s *) (Filter + ii*N);
				v2s *Line = (v2s *) (In + (i+ii)*W + j);
				for (jj=0; jj<(N/2); jj++) Acc = gap8_sumdotp2(Coeff[jj], Line[jj], Acc);
				Acc += Filter[ii*N+(N-1)]*In[(i+ii)*W + j + (N-1)];
                                // for (jj=0; jj<N; jj++) Acc += Filter[ii*N+jj]*In[(i+ii)*W + j + jj];
                        }
			Out[i*Wo+j] =  gap8_clip(gap8_roundnorm_reg(Acc, Norm), 15);
                        // Out[i*Wo+j] =  gap8_clip((Acc>>Norm), 15);
                }
        }
	wait_synch_barrier();
}

/*
	Linear rectification
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W, H]
*/
void __attribute__ ((noinline)) KerReLU_fp(KerReLUMaxPool2x2_fpT *Arg) {

 //    unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// }

#define Max(a, b) (((a)>(b))?(a):(b))

	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 * __restrict__ Out = Arg->Out;

    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSize((W*H)/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, (W*H)/2);
	v2s * VectIn  = (v2s *) In;
	v2s * VectOut = (v2s *) Out;
	int i, j;

	for (i=First; i<Last; i++) 
		VectOut[i] = gap8_max2(VectIn[i], ((v2s) {0, 0}));

	if((W*H)&0x1 && Last==(W*H)/2) 
		Out[W*H-1] = Max(In[W*H-1], 0);

	wait_synch_barrier();
#undef Max

	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// }

}

/*
	2x2 max pooling
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerMaxPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg) {


 //    unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// }

#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int Wo = W/2;
	int Ho = H/2;
	Word16 * __restrict__ Out = Arg->Out;

    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSizeEven(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	int i, j;

	for (i=0; i<(Ho); i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s M = gap8_max2(Line1[j], Line2[j]);
                        Out[Wo*i+j] = Max(M[0], M[1]);
		}
	}
	wait_synch_barrier();
#undef Max

	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// }
}

void __attribute__ ((noinline)) KerMaxPool2x2Padded_fp(KerReLUMaxPool2x2Padded_fpT *Arg) {


// unsigned int CoreId = gap8_coreid();
// // initialize the performance clock
// if(CoreId==0) {
// rt_perf_init(&perf[CoreId]);
// // Configure performance counters for counting the cycles
// rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
// rt_perf_reset(&perf[CoreId]);
// rt_perf_start(&perf[CoreId]);
// }

#define Max(a, b) (((a)>(b))?(a):(b))

	Word16 * __restrict__ In 	= Arg->In;
	Word16 * __restrict__ Out 	= Arg->Out;
	int W 						= Arg->W;
	int H 						= Arg->H;
	unsigned int Pad 			= Arg->Pad;
	unsigned char TileIndex 	= Arg->TileIndex;
	unsigned char NTile 		= Arg->NTile;

	unsigned int FirstTile 		= (TileIndex==0);
	unsigned int LastTile 		= (TileIndex==(NTile-1));
	unsigned int CoreId 		= gap8_coreid();
	int Wo 						= (W/2)+2*Pad;
	int Ho 						= (H/2)+2*Pad;
	unsigned int N 				= (Wo*Pad)/4;
	unsigned int Chunk 			= ChunkSize(N);
	unsigned int First 			= Chunk*CoreId;
	unsigned int Last 			= Min(First+Chunk, N);
	unsigned int i, j;

	// printf("Core%d %d %d\n",CoreId,First,Last);

/********************************* First Tile *********************************/

	if(FirstTile) {
		int *LineOut = (int *) (Out);
		for(i=First; i<Last; i++) {
			LineOut[2*i] = 0; 
			LineOut[2*i+1] = 0;
		}
		for(i=4*N; i<(Wo*Pad); i++) Out[i] = 0;
	}
	rt_team_barrier();


/********************************* Inner Tile *********************************/

	N = (Ho-Pad);
	Chunk = ChunkSize(N);
	First = Chunk*CoreId;
	Last = Min(First+Chunk, N);

	Word16 * Out_tmp = Out + Wo*Pad;
	for(i=First; i<Last; i++) {
		for(j=0; j<Pad; j++) {
			Out_tmp[i*Wo+j] = 0; 
			Out_tmp[(i+1)*Wo-j-1] = 0;
		}
	}

    unsigned int ChunkCell 		= ChunkSizeEven(Wo);
	First 						= CoreId*ChunkCell;
	Last  						= Minu(First+ChunkCell, Wo);

	for(i=0; i<(Ho); i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		// for(j=First; j<Last; j++) {
		// 	v2s M = gap8_max2(Line1[j], Line2[j]);
		// 	Out[Wo*i+j] = Max(M[0], M[1]);
		// }
	}
	rt_team_barrier();


/********************************** Last Tile *********************************/

	if(LastTile) {



		N = (Wo*Pad)/2;
		Chunk = ChunkSize(N);
		First = Chunk*CoreId;
		Last = Min(First+Chunk, N);

		printf("Core%d %d %d ; %d %d %d %d\n",CoreId,First,Last,Wo,W,Ho,H);

		int *LineOut = (int *) (Out+12*Wo);
		for (unsigned int i=First; i<Last; i++) LineOut[i] = 0;
		Out[(12+Pad)*Wo-1] = 0;
	}
	rt_team_barrier();

#undef Max

// if(CoreId==0) {
// rt_perf_stop(&perf[CoreId]);
// rt_perf_save(&perf[CoreId]);

// printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
// rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
// rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
// }
}

/*
	3x3 max pooling, Stride 1
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerMaxPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	Word16 * __restrict__ Out = Arg->Out;

	unsigned int even_w = (W%2)?0:1;
	unsigned int even_h = (H%2)?0:1;

    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSizeEven(W/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W/2);

	/* Find which core has to do the last column in case of even width*/
	unsigned int isLastColumn = 0;
	if(First<Last && Last==W/2 && even_w) {
		isLastColumn = 1;
		Last--;
	}
	int i;
	unsigned int j;
	v2s V0, V1, V2, V3, V4, V5;
    v2s *Line;

	// printf("DEBUG_MAXP: Core%d, %d %d %d (%d %d) %d %d [%d]\n", CoreId, ChunkCell, First, Last, W, H, even_w, even_h, isLastColumn);

	// for (i=0; i<(H/2); i++) {
	// 	v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
	// 	v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
	// 	for (j=First; j<Last; j++) {
	// 		v2s M = gap8_max2(Line1[j], Line2[j]);
	// 		Out[(W/2)*i+j] = Max(M[0], M[1]);
	// 	}
	// }

    for (j=First; j<Last; j++) {
    	Line = (v2s *) (&In[2*j]);
        V4 = Line[0]; V5 = Line[1]; V5[1] = 0;
    	for (i=0; i<(H/2)-even_h; i++) {
			V0 = V4; V1 = V5;
			Line = (v2s *) (&In[(2*i+1)*W+2*j]);
			V2 = Line[0]; V3 = Line[1]; V3[1] = 0;
			Line = (v2s *) (&In[(2*i+2)*W+2*j]);
            V4 = Line[0]; V5 = Line[1]; V5[1] = 0;
            // printf("IT%d(3x3): %d %d %d %d %d %d %d %d %d\n",	i*(W/2)+j, 	
            // 													V0[0], V0[1], V1[0], 
            // 													V2[0], V2[1], V3[0],
            // 													V4[0], V4[1], V5[0]);
            v2s M1 = gap8_max2(V0, V1);
            v2s M2 = gap8_max2(V2, V3);
            v2s M3 = gap8_max2(V4, V5);
            v2s M4 = gap8_max2(M1, M2);
            	M1 = gap8_max2(M3, M4);
            Out[i*(W/2)+j] = Max(M1[0], M1[1]);
    	}
    	/* Last row of each tile: instead of 3x3 is a 2x3 */
    	if(even_h) {
	    	V0 = V4; V1 = V5;
	    	Line = (v2s *) (&In[(2*i+1)*W+2*j]);
	    	V2 = Line[0]; V3 = Line[1]; V3[1] = 0;
	    	// printf("IT%d(2x3): %d %d %d %d %d %d\n",	i*(W/2)+j, 	
      //             										V0[0], V0[1], V1[0], 
      //             										V2[0], V2[1], V3[0]);
			v2s M1 = gap8_max2(V0, V1);
			v2s M2 = gap8_max2(V2, V3);
			v2s M3 = gap8_max2(M1, M2);
			Out[i*(W/2)+j] = Max(M3[0], M3[1]);
		}
    }
    /* Last column of each tile: instead of 3x3 is a 3x2 */
    if(isLastColumn) {
    	Line = (v2s *) (&In[2*j]);
        V2 = Line[0];
    	for (i=0; i<(H/2)-even_h; i++) {
			V0 = V2;
			Line = (v2s *) (&In[(2*i+1)*W+2*j]);
			V1 = Line[0];
			Line = (v2s *) (&In[(2*i+2)*W+2*j]);
            V2 = Line[0];
            // printf("IT%d(3x2): %d %d %d %d %d %d \n",	i*(W/2)+j, 	
            // 											V0[0], V0[1],
            // 											V1[0], V1[1],
            // 											V2[0], V2[1]);
            v2s M1 = gap8_max2(V0, V1);
            v2s M2 = gap8_max2(M1, V2);
            Out[i*(W/2)+j] = Max(M2[0], M2[1]);
		}
		/* Last row of the last tile: instead of 3x2 is a 2x2 */
		if(even_h) {
	    	V0 = V2;
	    	Line = (v2s *) (&In[(2*i+1)*W+2*j]);
	    	V1 = Line[0];
	    	// printf("IT%d(2x2): %d %d %d %d\n",	i*(W/2)+j, 	
	     //           							 	V0[0], V0[1], 
	     //           								V1[0], V1[1]);
			v2s M1 = gap8_max2(V0, V1);
			Out[i*(W/2)+j] = Max(M1[0], M1[1]);
		}
    }

	wait_synch_barrier();
#undef Max
}

/*
	2x2 Average pooling
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerAvgPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int Wo = W/2;
	int Ho = H/2;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	int i, j;

	for (i=0; i<Ho; i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			int S = gap8_dotp2(Line1[j], ((v2s) {1,1}));
                        Out[(W/2)*i+j] = gap8_sumdotp2(Line2[j], ((v2s) {1,1}), S)>>2;
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	3x3 Average pooling, stride 1
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerAvgPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int Wo = (W-3+1)/2;
	int Ho = (H-3+1)/2;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	int i, j;

	for (i=0; i<Ho; i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		v2s * __restrict__ Line3 = (v2s *) (&In[(2*i+2)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			int S = gap8_dotp2(Line1[j], ((v2s){1,1}));
			S =  gap8_sumdotp2(Line2[j], ((v2s){1,1}), S);
			S =  gap8_sumdotp2(Line3[j], ((v2s){1,1}), S);
			S += *((short int *)(Line1+j+1)) + *((short int *)(Line2+j+1)) + *((short int *)(Line3+j+1));
                        Out[Wo*i+j] = gap8_roundnorm(S*FP2FIX((1.0/9), 11), 11);
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 max pooling, single output version
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUMaxPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int Wo = W/2;
	int Ho = H/2;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	int i, j;

	for (i=0; i<Ho; i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s M = gap8_max2(Line1[j], Line2[j]);
			M = gap8_max2(M, ((v2s) {0, 0}));
                        Out[(W/2)*i+j] = Max(M[0], M[1]);
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 Average pooling
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUAvgPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int Wo = W/2;
	int Ho = H/2;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	int i, j;

	for (i=0; i<Ho; i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s V0 = gap8_max2(Line1[j], ((v2s){0,0})), V1 = gap8_max2(Line2[j], ((v2s){0,0}));
                        Out[(W/2)*i+j] = gap8_sumdotp2(V1, ((v2s) {1,1}), gap8_dotp2(V0, ((v2s){1,1})))>>2;
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 3x3 max pooling, single output version
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUMaxPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int Wo = (W-3+1)/2;
	int Ho = (H-3+1)/2;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	int i, j;

	for (i=0; i<(Ho); i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		v2s * __restrict__ Line3 = (v2s *) (&In[(2*i+2)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s M0 = gap8_max2(gap8_max2(Line1[j], Line2[j]), Line3[i]);
			int M1 = Max(Max(*((short int *) (Line1+j+1)), *((short int *) (Line2+j+1))), *((short int *) (Line3+j+1)));
                        Out[Wo*i+j] = Max(Max(Max(M0[0], M0[1]), M1), 0);
		}
	}
	wait_synch_barrier();
#undef Max
}


/*
	Linear rectification followed by a 3x3 Average pooling, stride 1
	In:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUAvgPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int W = Arg->W;
	int H = Arg->H;
	int Wo = (W-3+1)/2;
	int Ho = (H-3+1)/2;
	Word16 * __restrict__ Out = Arg->Out;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(Wo);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, Wo);
	int i, j;
	v2s Mask = (v2s){-1,0};

	for (i=0; i<Ho; i++) {
		v2s * __restrict__ Line1 = (v2s *) (&In[(2*i  )*W]);
		v2s * __restrict__ Line2 = (v2s *) (&In[(2*i+1)*W]);
		v2s * __restrict__ Line3 = (v2s *) (&In[(2*i+2)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s V0 = gap8_max2(Line1[j],   ((v2s){0,0})), V1 = gap8_max2(Line2[j],   ((v2s){0,0})), V2 = gap8_max2(Line3[j],   ((v2s){0,0}));
			v2s V3 = gap8_max2(Line1[j+1], ((v2s){0,0})), V4 = gap8_max2(Line2[j+1], ((v2s){0,0})), V5 = gap8_max2(Line3[j+1], ((v2s){0,0}));
			int S = gap8_sumdotp2(V0, ((v2s){1,1}), gap8_sumdotp2(V1, ((v2s){1,1}), gap8_dotp2(V2, ((v2s) {1,1}))));
			S = gap8_sumdotp2(V3&Mask, ((v2s){1,1}), gap8_sumdotp2(V4&Mask, ((v2s){1,1}), gap8_sumdotp2(V5&Mask, ((v2s){1,1}), S)));
                        Out[Wo*i+j] = gap8_roundnorm(S*FP2FIX((1.0/9), 11), 11);
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 max pooling, double output version
	In0:	short int *, Input data
	In1:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out0:	short int *, Output data [W/2, H/2]
	Out1:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUMaxPool2x2_2_fp(KerReLUMaxPool2x2_2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In0 = Arg->In0;
	Word16 * __restrict__ In1 = Arg->In1;
	int W = Arg->W;
	int H = Arg->H/2;
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(W/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W/2);
	int i, j;

	for (i=0; i<(H/2); i++) {
		v2s * __restrict__ Line0_1 = (v2s *) (&In0[(2*i  )*W]);
		v2s * __restrict__ Line0_2 = (v2s *) (&In1[(2*i  )*W]);
		v2s * __restrict__ Line1_1 = (v2s *) (&In0[(2*i+1)*W]);
		v2s * __restrict__ Line1_2 = (v2s *) (&In1[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s M0 = gap8_max2(Line0_1[j], Line1_1[j]);
			v2s M1 = gap8_max2(Line0_2[j], Line1_2[j]);
			M0 = gap8_max2(M0, ((v2s) {0, 0}));
			M1 = gap8_max2(M1, ((v2s) {0, 0}));
                        Out0[(W/2)*i+j] = Max(M0[0], M0[1]);
                        Out1[(W/2)*i+j] = Max(M1[0], M1[1]);
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 average pooling, double output version
	In0:	short int *, Input data
	In1:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out0:	short int *, Output data [W/2, H/2]
	Out1:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUAvgPool2x2_2_fp(KerReLUMaxPool2x2_2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In0 = Arg->In0;
	Word16 * __restrict__ In1 = Arg->In1;
	int W = Arg->W;
	int H = Arg->H/2;
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(W/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W/2);
	int i, j;

	for (i=0; i<(H/2); i++) {
		v2s * __restrict__ Line0_1 = (v2s *) (&In0[(2*i  )*W]);
		v2s * __restrict__ Line0_2 = (v2s *) (&In1[(2*i  )*W]);
		v2s * __restrict__ Line1_1 = (v2s *) (&In0[(2*i+1)*W]);
		v2s * __restrict__ Line1_2 = (v2s *) (&In1[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			int S0 = gap8_dotp2(Line0_1[j], ((v2s) {1,1}));
			int S1 = gap8_dotp2(Line0_2[j], ((v2s) {1,1}));
                        Out0[(W/2)*i+j] = gap8_sumdotp2(Line1_1[j], ((v2s) {1,1}), S0)>>2;
                        Out1[(W/2)*i+j] = gap8_sumdotp2(Line1_2[j], ((v2s) {1,1}), S1)>>2;
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 max pooling, double output version, pad with an extra 0 at end of line in case W is odd
	In0:	short int *, Input data
	In1:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out0:	short int *, Output data [W/2, H/2]
	Out1:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUMaxPool2x2_2_Even_fp(KerReLUMaxPool2x2_2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In0 = Arg->In0;
	Word16 * __restrict__ In1 = Arg->In1;
	int W = Arg->W;
	int H = Arg->H/2;
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;
	int Wout = W/2 + (W&0x1);

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(W/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W/2);
	int i, j;

	for (i=0; i<(H/2); i++) {
		v2s * __restrict__ Line0_1 = (v2s *) (&In0[(2*i  )*W]);
		v2s * __restrict__ Line0_2 = (v2s *) (&In1[(2*i  )*W]);
		v2s * __restrict__ Line1_1 = (v2s *) (&In0[(2*i+1)*W]);
		v2s * __restrict__ Line1_2 = (v2s *) (&In1[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s M0 = gap8_max2(Line0_1[j], Line1_1[j]);
			v2s M1 = gap8_max2(Line0_2[j], Line1_2[j]);
			M0 = gap8_max2(M0, ((v2s) {0, 0}));
			M1 = gap8_max2(M1, ((v2s) {0, 0}));
                        Out0[Wout*i+j] = Max(M0[0], M0[1]);
                        Out1[Wout*i+j] = Max(M1[0], M1[1]);
		}
		if (W&0x1) {
			Out0[Wout*i + Wout - 1] = 0;
			Out1[Wout*i + Wout - 1] = 0;
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 max pooling, double output version, drops one output at the end of each line, this is the counter
	part of KerReLUMaxPool2x2_2_Even_fp where we pad with one extra 0
	In0:	short int *, Input data
	In1:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out0:	short int *, Output data [W/2, H/2]
	Out1:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUMaxPool2x2_2_Drop_fp(KerReLUMaxPool2x2_2_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In0 = Arg->In0;
	Word16 * __restrict__ In1 = Arg->In1;
	int W = Arg->W;
	int H = Arg->H/2;
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;
	int Wout = (W/2)-1;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven((W/2)-1);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, (W/2)-1);
	int i, j;

	for (i=0; i<(H/2); i++) {
		v2s * __restrict__ Line0_1 = (v2s *) (&In0[(2*i  )*W]);
		v2s * __restrict__ Line0_2 = (v2s *) (&In1[(2*i  )*W]);
		v2s * __restrict__ Line1_1 = (v2s *) (&In0[(2*i+1)*W]);
		v2s * __restrict__ Line1_2 = (v2s *) (&In1[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s M0 = gap8_max2(Line0_1[j], Line1_1[j]);
			v2s M1 = gap8_max2(Line0_2[j], Line1_2[j]);
			M0 = gap8_max2(M0, ((v2s) {0, 0}));
			M1 = gap8_max2(M1, ((v2s) {0, 0}));
                        Out0[Wout*i+j] = Max(M0[0], M0[1]);
                        Out1[Wout*i+j] = Max(M1[0], M1[1]);
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 max pooling, triple output version
	In0:	short int *, Input data
	In1:	short int *, Input data
	In2:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out0:	short int *, Output data [W/2, H/2]
	Out1:	short int *, Output data [W/2, H/2]
	Out2:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUMaxPool2x2_3_fp(KerReLUMaxPool2x2_3_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In0 = Arg->In0;
	Word16 * __restrict__ In1 = Arg->In1;
	Word16 * __restrict__ In2 = Arg->In2;
	int W = Arg->W;
	int H = Arg->H/2;
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;
	Word16 * __restrict__ Out2 = Arg->Out2;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(W/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W/2);
	int i, j;

	for (i=0; i<(H/2); i++) {
		v2s * __restrict__ Line0_1 = (v2s *) (&In0[(2*i  )*W]);
		v2s * __restrict__ Line0_2 = (v2s *) (&In1[(2*i  )*W]);
		v2s * __restrict__ Line0_3 = (v2s *) (&In2[(2*i  )*W]);
		v2s * __restrict__ Line1_1 = (v2s *) (&In0[(2*i+1)*W]);
		v2s * __restrict__ Line1_2 = (v2s *) (&In1[(2*i+1)*W]);
		v2s * __restrict__ Line1_3 = (v2s *) (&In2[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			v2s M0 = gap8_max2(Line0_1[j], Line1_1[j]);
			v2s M1 = gap8_max2(Line0_2[j], Line1_2[j]);
			v2s M2 = gap8_max2(Line0_3[j], Line1_3[j]);
			M0 = gap8_max2(M0, ((v2s) {0, 0}));
			M1 = gap8_max2(M1, ((v2s) {0, 0}));
			M2 = gap8_max2(M2, ((v2s) {0, 0}));
                        Out0[(W/2)*i+j] = Max(M0[0], M0[1]);
                        Out1[(W/2)*i+j] = Max(M1[0], M1[1]);
                        Out2[(W/2)*i+j] = Max(M2[0], M2[1]);
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear rectification followed by a 2x2 averga pooling, triple output version
	In0:	short int *, Input data
	In1:	short int *, Input data
	In2:	short int *, Input data
	W, H:	Input data dimension [W x H]
	Out0:	short int *, Output data [W/2, H/2]
	Out1:	short int *, Output data [W/2, H/2]
	Out2:	short int *, Output data [W/2, H/2]
*/
void __attribute__ ((noinline)) KerReLUAvgPool2x2_3_fp(KerReLUMaxPool2x2_3_fpT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In0 = Arg->In0;
	Word16 * __restrict__ In1 = Arg->In1;
	Word16 * __restrict__ In2 = Arg->In2;
	int W = Arg->W;
	int H = Arg->H/2;
	Word16 * __restrict__ Out0 = Arg->Out0;
	Word16 * __restrict__ Out1 = Arg->Out1;
	Word16 * __restrict__ Out2 = Arg->Out2;

        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(W/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W/2);
	int i, j;

	for (i=0; i<(H/2); i++) {
		v2s * __restrict__ Line0_1 = (v2s *) (&In0[(2*i  )*W]);
		v2s * __restrict__ Line0_2 = (v2s *) (&In1[(2*i  )*W]);
		v2s * __restrict__ Line0_3 = (v2s *) (&In2[(2*i  )*W]);
		v2s * __restrict__ Line1_1 = (v2s *) (&In0[(2*i+1)*W]);
		v2s * __restrict__ Line1_2 = (v2s *) (&In1[(2*i+1)*W]);
		v2s * __restrict__ Line1_3 = (v2s *) (&In2[(2*i+1)*W]);
		// for (j=0; j<(W/2); j++) {
		for (j=First; j<Last; j++) {
			int S0 = gap8_dotp2(Line0_1[j], ((v2s) {1,1}));
			int S1 = gap8_dotp2(Line0_2[j], ((v2s) {1,1}));
			int S2 = gap8_dotp2(Line0_3[j], ((v2s) {1,1}));
                        Out0[(W/2)*i+j] = gap8_sumdotp2(Line1_1[j], ((v2s) {1,1}), S0)>>2;
                        Out1[(W/2)*i+j] = gap8_sumdotp2(Line1_2[j], ((v2s) {1,1}), S1)>>2;
                        Out2[(W/2)*i+j] = gap8_sumdotp2(Line1_3[j], ((v2s) {1,1}), S2)>>2;
		}
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear layer:	Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
	Outputs are evaluated in sequence, a given output is evaluated in parallel

	In	short int, input vector
	InSize	Number of inputs
	Filter	short int, Linear filter coefficients. Dimension: InSize*OutSize
	Out	short int, output vector
	OutSize	output dimension
*/

void __attribute__ ((noinline)) KerLinearLayer_fp(KerLinearLayer_fpT *Arg) {

 //    unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	// if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// // }
#if 1
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word16 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word16 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
	static L1_CL_MEM int Reduct[8];
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSizeEven(InSize/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InSize/2);
    int i,j;

	v2s * __restrict__ VectIn = (v2s *) In;

       	// Linear combination
	for (i=0; i<OutSize; i++) {

		v2s * __restrict__ Filt = (v2s *) (&Filter[i*InSize]);
		int Acc = 0;

		for (j = First; j<Last; j++) {
			Acc = gap8_sumdotp2(VectIn[j], Filt[j], Acc);
		}

		Reduct[CoreId] = Acc;
		wait_synch_barrier();

		if(CoreId==0) {

			Acc = (Bias[i]<<NormBias);

			if(InSize%2) Acc += In[InSize-1]*Filter[i*InSize+InSize-1];

			for(j=0; j<gap8_ncore(); j++) {
				Acc += Reduct[j];
			}

			Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, NormFilter), 15);
		}
		wait_synch_barrier();
	}
	wait_synch_barrier();
#else
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word16 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word16 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
	static L1_CL_MEM int Reduct[8];
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSizeEven(InSize/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InSize/2);
    int i,j;

    if(gap8_coreid() == 0) {
    	int Acc = Bias[0] << NormBias;

    	for(int i=0; i<InSize; i++) {
    		fixed2string((Filter[i] * In[i]) >> NormFilter, 13, 4);
    		Acc += Filter[i] * In[i];
    	}
    	Out[0] = Acc >> NormFilter;
    }


#endif
	// // if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// // }
}

/*
	Linear layer:	Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
	Outputs are evaluated in parallel, a given output is evaluated in sequence

	In	short int, input vector
	InSize	Number of inputs
	Filter	short int, Linear filter coefficients. Dimension: InSize*OutSize
	Out	short int, output vector
	OutSize	output dimension
*/

void __attribute__ ((noinline)) KerLinearLayerParOut_fp(KerLinearLayer_fpT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word16 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word16 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSize(OutSize);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, OutSize);
        int i,j;
	v2s * __restrict__ VectIn = (v2s *) In;

       	// Linear combination
	// for (i=0; i<OutSize; i++) {
	for (i=First; i<Last; i++) {
		v2s * __restrict__ Filt = (v2s *) (&Filter[i*InSize]);
		int Acc = 0;
		for (j = 0; j<(InSize/2); j++) Acc = gap8_sumdotp2(VectIn[j], Filt[j], Acc);
		if (InSize%2) Acc += In[InSize-1]*Filter[i*InSize+InSize-1];
		Acc += (Bias[i]<<NormBias);
		Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, NormFilter), 15);
	}
	wait_synch_barrier();
}

/*
	Linear layer:	Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
	Outputs are evaluated in sequence, a given output is evaluated in parallel

	In	short int, input vector
	InSize	Number of inputs
	Filter	short int, Linear filter coefficients. Dimension: InSize*OutSize
	Out	int, output vector
	OutSize	output dimension
*/
void __attribute__ ((noinline)) KerLinearLayer_fpd(KerLinearLayer_fpdT *Arg)

{
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word16 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word32 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
	static L1_CL_MEM int Reduct[8];
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(InSize/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InSize/2);
        int i,j;
	v2s * __restrict__ VectIn = (v2s *) In;

       	// Linear combination
	for (i=0; i<OutSize; i++) {
		v2s * __restrict__ Filt = (v2s *) (&Filter[i*InSize]);
		int Acc = 0;
		// for (j = 0; j<(InSize/2); j++) Acc = gap8_sumdotp2(In[j], Filt[j], Acc); j++;
		for (j = First; j<Last; j++) Acc = gap8_sumdotp2(VectIn[j], Filt[j], Acc); j++;
		Reduct[CoreId] = Acc;
		wait_synch_barrier();
		if (CoreId==0) {
			Acc = (Bias[i]<<NormBias);
			if (InSize%2) Acc += In[InSize-1]*Filter[i*InSize+InSize-1];
			for (j=0;j<gap8_ncore();j++) Acc += Reduct[j];
			Out[i] = gap8_roundnorm_reg(Acc, NormFilter);
		}
		wait_synch_barrier();
	}
	wait_synch_barrier();
}

/*
	Linear layer:	Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
	Outputs are evaluated in sequence, a given output is evaluated in parallel

	In	Byte, input vector
	InSize	Number of inputs
	Filter	short int, Linear filter coefficients. Dimension: InSize*OutSize
	Out	short int, output vector
	OutSize	output dimension
*/
void __attribute__ ((noinline)) KerLinearLayer_fps(KerLinearLayer_fpsT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word8 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word16 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
	static L1_CL_MEM int Reduct[8];
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(InSize/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InSize/2);
        int i,j;
	v2s * __restrict__ VectIn = (v2s *) In;
       	// Linear combination
	for (i=0; i<OutSize; i++) {
		const Word8 * __restrict__ Filt1 = &Filter[InSize*i+2*First];
		const Word8 * __restrict__ Filt2 = Filt1+1; // &Filter[InSize*i+1+2*First];
		int Acc = 0;
		// for (j=0; j<(InSize/2); j++) {
		for (j=First; j<Last; j++) {
			int F1 = *Filt1; Filt1 += 2;
			int F2 = *Filt2; Filt2 += 2;
			v2s F = gap8_pack2(F1, F2);
			// Acc = gap8_sumdotp2(In[j], (F<<(v2s){8, 8}), Acc);
			Acc = gap8_sumdotp2(VectIn[j], F, Acc);
		}
		Reduct[CoreId] = Acc;
		wait_synch_barrier();
		if (CoreId==0) {
			Acc = (Bias[i]<<NormBias);
			// if (InSize%2) Acc += In[InSize-1]* (*Filt1);
			if (InSize%2) Acc += In[InSize-1]* Filter[InSize*i + InSize-1]; // [InSize-1];
			for (j=0;j<gap8_ncore();j++) Acc += Reduct[j];
			Out[i] = gap8_clip(gap8_roundnorm_reg(Acc, NormFilter), 15);
		}
		wait_synch_barrier();
	}
	wait_synch_barrier();
#undef Max
}

/*
	Linear layer:	Out[i] = ReLU((sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter)
	Outputs are evaluated in sequence, a given output is evaluated in parallel

	In	Byte, input vector
	InSize	Number of inputs
	Filter	short int, Linear filter coefficients. Dimension: InSize*OutSize
	Out	short int, output vector
	OutSize	output dimension
*/
void __attribute__ ((noinline)) KerLinearLayerReLU_fps(KerLinearLayer_fpsT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word8 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word16 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
	static L1_CL_MEM int Reduct[8];
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(InSize/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InSize/2);
        int i,j;
	v2s * __restrict__ VectIn = (v2s *) In;

       	// Linear combination
	for (i=0; i<OutSize; i++) {
		const Word8 * __restrict__ Filt1 = &Filter[InSize*i+2*First];
		const Word8 * __restrict__ Filt2 = Filt1+1; // &Filter[InSize*i+1+2*First];
		int Acc = 0;
		// for (j=0; j<(InSize/2); j++) {
		for (j=First; j<Last; j++) {
			int F1 = *Filt1; Filt1 += 2;
			int F2 = *Filt2; Filt2 += 2;
			v2s F = gap8_pack2(F1, F2);
			// Acc = gap8_sumdotp2(In[j], (F<<(v2s){8, 8}), Acc);
			Acc = gap8_sumdotp2(VectIn[j], F, Acc);
		}
		Reduct[CoreId] = Acc;
		wait_synch_barrier();
		if (CoreId==0) {
			Acc = (Bias[i]<<NormBias);
			// if (InSize%2) Acc += In[InSize-1]* (*Filt1);
			if (InSize%2) Acc += In[InSize-1]* Filter[InSize*i + InSize-1]; // [InSize-1];
			for (j=0;j<gap8_ncore();j++) Acc += Reduct[j];
			Out[i] = Max(gap8_clip(gap8_roundnorm_reg(Acc, NormFilter), 15), 0);
		}
		wait_synch_barrier();
	}
	wait_synch_barrier();
#undef Max
}


/*
	Linear layer:	Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
	Outputs are evaluated in sequence, a given output is evaluated in parallel

	In	short int, input vector
	InSize	Number of inputs
	Filter	short int, Linear filter coefficients. Dimension: InSize*OutSize
	Out	short int, output vector
	OutSize	output dimension
*/

void __attribute__ ((noinline)) KerLinearLayerReLU_fp(KerLinearLayer_fpT *Arg) {

	//     unsigned int CoreId = gap8_coreid();
 //    // initialize the performance clock
 //   	// if(CoreId==0) {
	//     rt_perf_init(&perf[CoreId]);
	//     // Configure performance counters for counting the cycles
	//     rt_perf_conf(&perf[CoreId], (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR));
	//     rt_perf_reset(&perf[CoreId]);
	//     rt_perf_start(&perf[CoreId]);
	// // }

#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word16 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word16 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
	static L1_CL_MEM int Reduct[8];
    unsigned int CoreId = gap8_coreid();
    unsigned int ChunkCell = ChunkSizeEven(InSize/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InSize/2);
        int i,j;
	v2s * __restrict__ VectIn = (v2s *) In;

       	// Linear combination
	for (i=0; i<OutSize; i++) {
		v2s * __restrict__ Filt = (v2s *) (&Filter[i*InSize]);
		int Acc = 0;
		// for (j = 0; j<(InSize/2); j++) Acc = gap8_sumdotp2(In[j], Filt[j], Acc); j++;
		for (j = First; j<Last; j++) Acc = gap8_sumdotp2(VectIn[j], Filt[j], Acc); j++;
		Reduct[CoreId] = Acc;
		wait_synch_barrier();
		if (CoreId==0) {
			Acc = (Bias[i]<<NormBias);
			if (InSize%2) Acc += In[InSize-1]*Filter[i*InSize+InSize-1];
			for (j=0;j<gap8_ncore();j++) Acc += Reduct[j];
			Out[i] = Max(gap8_clip(gap8_roundnorm_reg(Acc, NormFilter), 15), 0);
		}
		wait_synch_barrier();
	}
	wait_synch_barrier();
#undef Max

	// 	// if(CoreId==0) {
	// 	rt_perf_stop(&perf[CoreId]);
	//     rt_perf_save(&perf[CoreId]);

	// 	printf("Cycles_%d: %d\t\tInstructions: %d\n", CoreId,
	//         rt_perf_get(&perf[CoreId], RT_PERF_CYCLES), 
	//         rt_perf_get(&perf[CoreId], RT_PERF_INSTR));
	// // }
}

/*
	Linear layer:	Out[i] = ReLU((sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter)
	Outputs are evaluated in sequence, a given output is evaluated in parallel

	In	short int, input vector
	InSize	Number of inputs
	Filter	short int, Linear filter coefficients. Dimension: InSize*OutSize
	Out	int, output vector
	OutSize	output dimension
*/
void __attribute__ ((noinline)) KerLinearLayerReLU_fpd(KerLinearLayer_fpdT *Arg)

{
#define Max(a, b) (((a)>(b))?(a):(b))
	Word16 * __restrict__ In = Arg->In;
	int InSize = Arg->InSize;
	const Word16 * __restrict__ Filter = Arg->Filter;
	unsigned int NormFilter = Arg->NormFilter;
	const Word16 * __restrict__ Bias = Arg->Bias;
	unsigned int NormBias = Arg->NormBias;
	Word32 * __restrict__ Out = Arg->Out;
	int OutSize = Arg->OutSize;
	static L1_CL_MEM int Reduct[8];
        unsigned int CoreId = gap8_coreid();
        unsigned int ChunkCell = ChunkSizeEven(InSize/2);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, InSize/2);
        int i,j;
	v2s * __restrict__ VectIn = (v2s *) In;

       	// Linear combination
	for (i=0; i<OutSize; i++) {
		v2s * __restrict__ Filt = (v2s *) (&Filter[i*InSize]);
		int Acc = 0;
		// for (j = 0; j<(InSize/2); j++) Acc = gap8_sumdotp2(In[j], Filt[j], Acc); j++;
		for (j = First; j<Last; j++) Acc = gap8_sumdotp2(VectIn[j], Filt[j], Acc); j++;
		Reduct[CoreId] = Acc;
		wait_synch_barrier();
		if (CoreId==0) {
			Acc = (Bias[i]<<NormBias);
			if (InSize%2) Acc += In[InSize-1]*Filter[i*InSize+InSize-1];
			for (j=0;j<gap8_ncore();j++) Acc += Reduct[j];
			Out[i] = gap8_roundnorm_reg(Acc, NormFilter);
			Out[i] = Max(gap8_roundnorm_reg(Acc, NormFilter), 0);
		}
		wait_synch_barrier();
	}
	wait_synch_barrier();
#undef Max
}

