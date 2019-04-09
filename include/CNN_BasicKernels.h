/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the Apache License.  See the LICENSE file for details.
 *
 */


#ifndef __BASIC_KERNEL_LIB_H__
#define __BASIC_KERNEL_LIB_H__

#include "Gap8.h"
#include "StdTypes.h"

/** @addtogroup groupCNN
@{ */

/** @defgroup CNNBasicK CNNBasicKernels

@{ */

typedef struct {
        Word16 * __restrict__ In;       /**< Pointer to input tile */
        Word16 * __restrict__ Out;      /**< Pointer to an output tile */
        unsigned int Win;               /**< Width of the tile */
        unsigned int Hin;               /**< Height of the tile */
        unsigned int Pad;
        unsigned char TileIndex;
        unsigned char NTile;
} ArgExpandTile_fp_T;


void KerExpandTileInitH_fp(ArgExpandTile_fp_T *Arg);

void KerPadFirstTileH_fp(ArgExpandTile_fp_T *Arg);

void KerExpandTileH_fp(ArgExpandTile_fp_T *Arg);


// =============================================================================

typedef struct {
        short int *__restrict__ In;
        short int *__restrict__ Out;
        short int *__restrict__ Filter;
        unsigned char Fw;
        unsigned char Fh;
        unsigned char PadL;
        unsigned char PadT;
        unsigned short int W;
        unsigned short int H;
        unsigned short int Wo;
        unsigned short int Wo_F;
        unsigned short int Wo_L;
        unsigned short int Ho;
        unsigned short int Ho_F;
        unsigned short int Ho_L;
        unsigned short int Stride;
        unsigned short int Norm;
} ArgConvGeneric_T;

typedef struct {
        short int * __restrict__ In;
        unsigned short int W;
        unsigned short int H;
        unsigned short int InFeatures;
        unsigned short int OutFeatures;
        unsigned short int BaseOutFeature;
        short int * __restrict__ Filter;
        short int * __restrict__ Out;
        unsigned short int Norm;
        unsigned short int TileIndex;
        unsigned short int NTile;
        unsigned short int Orientation; /**< 1: Horizontal, 0: Vertical */
        v4s Pad;                        /**< Paddding, 0: Left, 1: Right, 2: Top, 3: Bottom */
        unsigned short int TileSize;
        unsigned short int TotalSize;
        unsigned short int N;           /**< Dimension of the convolution: NxN, used only for general versions */
        unsigned short int S;           /**< Output stride, used only for general versions */
} KerParConv_fp_T;

#ifdef NEW
typedef struct {
        short int * __restrict__ In;
        short int * __restrict__ Filter;
        short int * __restrict__ Out;
        unsigned short int W;
        unsigned short int H;

        unsigned short int InFeatures;
        unsigned short int OutFeatures;

        unsigned short int BaseFeature; /**< Relative offset of In/Out Feature in the original feature space, when NTile=0 relates to input features, otherwise relates to output features */
        unsigned short int TileIndex;   /**< Index of the current tile if NTile!=0 otherwise used to pass the total number of input features */

        unsigned short int NTile;       /**< Total number of In tiles. If 0 indicates that feature is complete */

        unsigned char Norm;
        unsigned char Orientation;      /**< 1: Horizontal, 0: Vertical */
        v4s Pad;                        /**< Paddding, 0: Left, 1: Right, 2: Top, 3: Bottom */
        unsigned short int InTileSize;  /**< Orientation=1 (Horizontal) => In tile standard height (not the last one), 0 (Vertical) => In tile standard width. Used only when In Feature is tiled */
        unsigned short int TotalInSize; /**< Orientation=1 (Horizontal) => Total In height, 0 (Vertical) => Total In width. Used only when In Feature is tiled */
        unsigned short int N;           /**< Dimension of the convolution: NxN. Used Only if generic Kernel is used */
        unsigned short int S;           /**< Output stride. Used Only if generic Kernel is used */
} KerParConv_fp_T;
#endif

typedef struct { 
        short int * __restrict__ In;    /**< Pointer to input tile  */
        unsigned short int W;           /**< Width of the input tile */
        unsigned short int H;           /**< Height of the input tile */
        short int * __restrict__ Filter;/**< Pointer to convolution coefficients. (N x N) coeffs in Q15 */
        short int * __restrict__ Out;   /**< Pointer to output tile, this tile can have up to N-1 lines and N-1 column than In depending on Pad */
        unsigned short int Norm;        /**< Fixed point format, should be <= 15 */
        unsigned short int N;           /**< Dimension of the convolution: NxN, used only for general versions */
        unsigned short int S;           /**< Output stride, used only for general versions */
        v4s Pad;                        /**< Paddding, 0: Left, 1: Right, 2: Top, 3: Bottom */
} KerPaddedConv_fpT;

typedef struct {
        short int * __restrict__ Out;
        unsigned short int W;
        unsigned short int H;
        unsigned short int OutFeatures;
        unsigned short int BaseOutFeature;
        short int * __restrict__ Bias;
} KerParSetBias_fp_T;

typedef struct {
        short int * __restrict__ In;
        unsigned short int W;
        unsigned short int H;
        unsigned short int OutFeatures;
        short int * __restrict__ Out;
        v4s Pad;
        unsigned short int DoReLU;
} KerParReLUMaxPool_fp_T;

extern void KerConv3x3Stride1_fp(KerPaddedConv_fpT *Arg);
extern void KerConv3x3Stride2_fp(KerPaddedConv_fpT *Arg);
extern void KerConv5x5Stride1_fp(KerPaddedConv_fpT *Arg);
extern void KerConv5x5Stride2_fp(KerPaddedConv_fpT *Arg);

extern void KerParConv1x1Stride1_fp(KerParConv_fp_T *Arg);
extern void KerParConv1x1Stride2_fp(KerParConv_fp_T *Arg);
extern void KerParConv3x3Stride1_fp(KerParConv_fp_T *Arg);
extern void KerParConv3x3Stride2_fp(KerParConv_fp_T *Arg);
extern void KerParConv5x5Stride1_fp(KerParConv_fp_T *Arg);
extern void KerParConv5x5Stride2_fp(KerParConv_fp_T *Arg);
extern void KerParConvNxNStrideS_fp(KerParConv_fp_T *Arg);

extern void KerParSetBias_fp(KerParSetBias_fp_T *Arg);

extern void KerParMaxPool2x2Stride2_fp(KerParReLUMaxPool_fp_T *Arg);
extern void KerParAvgPool2x2Stride2_fp(KerParReLUMaxPool_fp_T *Arg);

extern void KerParReLU_fp(KerParReLUMaxPool_fp_T *Arg);

// =============================================================================


/** @brief Template for CNN Add Feature Maps
Template for CNN Add Feature Maps
*/
typedef struct {
        Word16 * __restrict__ In;      /**< Pointer to input tile0  */
        // Word16 * __restrict__ In1;      /**< Pointer to input tile1  */
        Word16 * __restrict__ Out;      /**< Pointer to an output tile */
        int     W;                      /**< Width of the tile */
        int     H;                      /**< Height of the tile */
} KerAddFM_fpT;

/** @brief Convolution related function. Add two Feature Maps.

 Convolution related function. Add two Feature Maps.
*/
void KerAddFM_fp(KerAddFM_fpT *Arg);

/** @brief Convolution related function. Add two Feature Maps and the Activation..

 Convolution related function. Add two Feature Maps.
*/
void KerAddFMReLu_fp(KerAddFM_fpT *Arg);

/** @brief Template for CNN bias setting

Template for CNN bias setting
*/
typedef struct {
	Word16 * __restrict__ Out;	/**< Pointer to an ouput tile */
	int	W;			/**< Width of the tile */
	int	H;			/**< Height of the tile */
	Word16 Bias;			/**< Bias value to assign to all tile's elements */
} KerSetInBiasT;

/** @brief Assign a bias to all elements in the output tile

Assign Bias to all elements in the input/output tile
*/
void KerSetInBias(KerSetInBiasT *Arg);

void KerSetInBiasPadded(KerSetInBiasT *Arg);

/** @brief Template for CNN bias setting, 2 outputs, 2 bias

Template for CNN bias setting, 2 outputs, 2 bias
*/
typedef struct {
	Word16 * __restrict__ Out0;	/**< Pointer to 1st output tile */
	Word16 * __restrict__ Out1;	/**< Pointer to 2nd output tile */
	int	W;			/**< Width of the tile, both have same width */
	int	H;			/**< Height of the tile, both have same height */
	Word16 Bias0;			/**< Bias to assign to all 1st tile's elements */
	Word16 Bias1;			/**< Bias to assign to all 2nd tile's elements */
} KerSetInBias2T;

/** @brief Assign a bias to all elements in the 2 output tiles, each tile has a different bias

Assign a bias to all elements in the 2 output tiles, each tile has a different bias
*/
void KerSetInBias2(KerSetInBias2T *Arg);

/** @brief Template for CNN bias setting, 3 outputs, 3 bias

Template for CNN bias setting, 3 outputs, 3 bias
*/
typedef struct {
	Word16 * __restrict__ Out0;	/**< Pointer to 1st output tile */
	Word16 * __restrict__ Out1;	/**< Pointer to 2nd output tile */
	Word16 * __restrict__ Out2;	/**< Pointer to 3rd output tile */
	int	W;			/**< Width of the tile, all have same width */
	int	H;			/**< Height of the tile, all have same height */
	Word16 Bias0;			/**< Bias to assign to all 1st tile's elements */
	Word16 Bias1;			/**< Bias to assign to all 2nd tile's elements */
	Word16 Bias2;			/**< Bias to assign to all 3rd tile's elements */
} KerSetInBias3T;

/** @brief Assign a bias to all elements in the 3 output tiles, each tile has a different bias

Assign a bias to all elements in the 3 output tiles, each tile has a different bias
*/
extern void KerSetInBias3(KerSetInBias3T *Arg);


/* Convolution */

/** @brief Template for convolutions, CNN and pure ones

Template for convolutions, CNN and pure ones
*/
typedef struct {
	Word16 * __restrict__ In;	/**< Pointer to input tile  */
	int W;				/**< Width of the input tile */
	int H;				/**< Height of the input tile */
	Word16 * __restrict__ Filter;	/**< Pointer to convolution coefficients. (N x N) coeffs in Q15 */
	Word16 * __restrict__ Out;	/**< Pointer to output tile, this tile has N-1 lines and N-1 column than In */
	unsigned int Norm;		/**< Fixed point format, should be <= 15 */
	int N;				/**< Dimension of the convolution: NxN, used only for general versions */
	int Stride;			/**< Output stride, used only for general versions */
        unsigned int InCh;              /**< Number of Input Channels, used only for Multi Feature versions */
        Word32 * __restrict__ Reduct;   /**< Pointer to the Reduction Buffer */
} KerConv_fpT;


/** @brief CNN 1x1 convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

CNN 1x1 convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

        1x1 convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and clipped to 16bits before being written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the 9 convolution coefficients
        Norm:   Fixed point format
*/
// void KerConv1x1Stride2_fp(KerConv_fpT *Arg);

void KerConv1x1Stride2Multi_fp(KerConv_fpT *Arg);


/** @brief CNN 3x3 convolution, Q15 inputs and outputs. Accumulation with previous output.

CNN 3x3 convolution, Q15 inputs and outputs. Accumulation with previous output.

        3x3 convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the 9 convolution coefficients
        Norm:   Fixed point format
*/
void KerConv3x3_fp(KerConv_fpT *Arg);

void KerConv3x3Multi_fp(KerConv_fpT *Arg);


/** @brief CNN 3x3 convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

CNN 3x3 convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

        3x3 convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the 9 convolution coefficients
        Norm:   Fixed point format
*/
// void KerConv3x3Stride2_fp(KerConv_fpT *Arg);

void KerConv3x3Stride2Multi_fp(KerConv_fpT *Arg);



void KerReLUConv3x3Stride2_fp(KerConv_fpT *Arg);

void KerReLUConv3x3_fp(KerConv_fpT *Arg);



/** @brief Pure 3x3 convolution, Q15 inputs and outputs. No accumulation with previous output.

Pure 3x3 convolution, Q15 inputs and outputs. No accumulation with previous output.

        3x3 convolution, short int inputs and output
        The result of the convolution is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the 9 convolution coefficients
        Norm:   Fixed point format
*/
void KerDirectConv3x3_fp(KerConv_fpT *Arg);

/** @brief CNN 5x5 convolution, Q15 inputs and outputs. Accumulation with previous output.

CNN 5x5 convolution, Q15 inputs and outputs. Accumulation with previous output.

        5x5 convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the 9 convolution coefficients
        Norm:   Fixed point format
*/
void KerConv5x5_fp(KerConv_fpT *Arg);


/** @brief CNN 5x5 convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

CNN 5x5 convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

        5x5 convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the 9 convolution coefficients
        Norm:   Fixed point format
*/
// void KerConv5x5Stride2_fp(KerConv_fpT *Arg);

void KerConv5x5Stride2Padded_fp(KerConv_fpT *Arg);

/** @brief Pure 5x5 convolution, Q15 inputs and outputs. No accumulation with previous output.

Pure 5x5 convolution, Q15 inputs and outputs. No accumulation with previous output.

        5x5 convolution, short int inputs and output
        The result of the convolution is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the 9 convolution coefficients
        Norm:   Fixed point format
*/
void KerDirectConv5x5_fp(KerConv_fpT *Arg);

/** @brief CNN NxN convolution, Q15 inputs and outputs. Accumulation with previous output.

CNN NxN convolution, Q15 inputs and outputs. Accumulation with previous output.

        NxN convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the NxN convolution coefficients
        Norm:   Fixed point format
*/
void KerConvNxN_fp(KerConv_fpT *Arg);


/** @brief CNN NxN convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

CNN NxN convolution, Q15 inputs and outputs. Accumulation with previous output. stride=2 only even lines and columns are produced

        NxN convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the NxN convolution coefficients
        Norm:   Fixed point format
*/
void KerConvNxNStride2_fp(KerConv_fpT *Arg);

/** @brief CNN NxN convolution, Q15 inputs and outputs. Accumulation with previous output. stride=M.

CNN NxN convolution, Q15 inputs and outputs. Accumulation with previous output. stride=M.

        NxN convolution, short int inputs and output
        The result of the convolution is accumulated to the existing output, then it is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the NxN convolution coefficients
        Norm:   Fixed point format
*/
void KerConvNxNStrideM_fp(KerConv_fpT *Arg);

/** @brief Pure NxN convolution, Q15 inputs and outputs. No accumulation with previous output.

Pure NxN convolution, Q15 inputs and outputs. No accumulation with previous output.

        NxN convolution, short int inputs and output
        The result of the convolution is rounded, normalized and cliped to 16bits before beeing written
        In:     short int *, convolution input
        W, H:   Input dimension [W x H]
        Filter: short int *, pointer to the NxN convolution coefficients
        Norm:   Fixed point format
*/
void KerDirectConvNxN_fp(KerConv_fpT *Arg);

/* Rectification, Pooling */

/** @brief Template for rcetification and pooling basic kernels

Template for rcetification and pooling basic kernels
*/
typedef struct {
        Word16 * __restrict__ In;	/**< Pointer to input tile  */
        int W;				/**< Width of the input tile */
        int H;				/**< Height of the input tile */
        Word16 * __restrict__ Out;	/**< Pointer to output tile */
} KerReLUMaxPool2x2_fpT;

typedef struct {
        Word16 * __restrict__ In;       /**< Pointer to input tile  */
        int W;                          /**< Width of the input tile */
        int H;                          /**< Height of the input tile */
        Word16 * __restrict__ Out;      /**< Pointer to output tile */
        unsigned int Pad;
        unsigned char TileIndex;
        unsigned char NTile;
} KerReLUMaxPool2x2Padded_fpT;


/** @brief Linear rectification basic kernel

Linear rectification basic kernel

        Linear rectification
        In:     short int *, Input data [W x H] 
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W, H]
*/
void KerReLU_fp(KerReLUMaxPool2x2_fpT *Arg);

/** @brief 2x2 Max Pooling basic kernel

2x2 Max Pooling basic kernel

        2x2 max pooling
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerMaxPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg);

void KerMaxPool2x2Padded_fp(KerReLUMaxPool2x2Padded_fpT *Arg);

/** @brief 3x3 Max Pooling basic kernel

3x3 Max Pooling basic kernel

        3x3 max pooling
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerMaxPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg);

/** @brief 2x2 Average Pooling basic kernel

2x2 Average Pooling basic kernel

        2x2 Average pooling
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerAvgPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg);

/** @brief 3x3 Average Pooling basic kernel

3x3 Average Pooling basic kernel

        3x3 Average pooling
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerAvgPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg);

/** @brief Linear rectification followed by 2x2 Max Pooling basic kernel

Linear rectification followed by 2x2 Max Pooling basic kernel

        Linear rectification followed by a 2x2 max pooling, single output version
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerReLUMaxPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg);

/** @brief Linear rectification followed by 2x2 Avg Pooling basic kernel

Linear rectification followed by 2x2 Avg Pooling basic kernel

        Linear rectification followed by a 2x2 Avg pooling, single output version
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerReLUAvgPool2x2_fp(KerReLUMaxPool2x2_fpT *Arg);

/** @brief Linear rectification followed by 3x3 Max Pooling basic kernel

Linear rectification followed by 3x3 Max Pooling basic kernel

        Linear rectification followed by a 3x3 max pooling, single output version
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerReLUMaxPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg);

/** @brief Linear rectification followed by 3x3 Max Pooling basic kernel

Linear rectification followed by 3x3 Avg Pooling basic kernel

        Linear rectification followed by a 3x3 Avg pooling, single output version
        In:     short int *, Input data
        W, H:   Input data dimension [W x H]
        Out:    short int *, Output data [W/2, H/2]
*/
void KerReLUAvgPool3x3_fp(KerReLUMaxPool2x2_fpT *Arg);

 
/** @brief Template for rectification and pooling basic kernels, 2 tiles version

Template for rectification and pooling basic kernels, 2 tiles version
*/
typedef struct {
        Word16 * __restrict__ In0;	/**< Pointer to 1st input tile  */
        Word16 * __restrict__ In1;	/**< Pointer to 2nd input tile  */
        int W;				/**< Width of the tile. Input and output have the same */
        int H;				/**< Height of the tile. Input and output have the same */
        Word16 * __restrict__ Out0;	/**< Pointer to 1st output tile  */
        Word16 * __restrict__ Out1;	/**< Pointer to 2nd output tile  */
} KerReLUMaxPool2x2_2_fpT;

/** @brief Linear rectification followed by 2x2 Max Pooling basic kernel, 2 output version.

Linear rectification followed by 2x2 Max Pooling basic kernel, 2 output version.

        Linear rectification followed by a 2x2 max pooling, double output version
        In0:    short int *, Input data
        In1:    short int *, Input data
        W, H:   Input data dimension [W x H]
        Out0:   short int *, Output data [W/2, H/2]
        Out1:   short int *, Output data [W/2, H/2]
*/
void KerReLUMaxPool2x2_2_fp(KerReLUMaxPool2x2_2_fpT *Arg);

/** @brief Linear rectification followed by 2x2 Average Pooling basic kernel, 2 output version.

Linear rectification followed by 2x2 Average Pooling basic kernel, 2 output version.

        Linear rectification followed by a 2x2 Average pooling, double output version
        In0:    short int *, Input data
        In1:    short int *, Input data
        W, H:   Input data dimension [W x H]
        Out0:   short int *, Output data [W/2, H/2]
        Out1:   short int *, Output data [W/2, H/2]
*/
void KerReLUAvgPool2x2_2_fp(KerReLUMaxPool2x2_2_fpT *Arg);

/** @brief Linear rectification followed by 2x2 Average Pooling basic kernel, 2 output version. Pads with 0 if odd.

Linear rectification followed by 2x2 Average Pooling basic kernel, 2 output version. Pads with 0 if odd.

        Linear rectification followed by a 2x2 max pooling, double output version, pad with an extra 0 at end of line in case W is odd
        In0:    short int *, Input data
        In1:    short int *, Input data
        W, H:   Input data dimension [W x H]
        Out0:   short int *, Output data [W/2, H/2]
        Out1:   short int *, Output data [W/2, H/2]
*/
void KerReLUMaxPool2x2_2_Even_fp(KerReLUMaxPool2x2_2_fpT *Arg);

/** @brief Linear rectification followed by 2x2 Average Pooling basic kernel, 2 output version. Drops last.

Linear rectification followed by 2x2 Average Pooling basic kernel, 2 output version. Drops last.

        Linear rectification followed by a 2x2 max pooling, double output version, drops one output at the end of each line, this is the counter
        part of KerReLUMaxPool2x2_2_Even_fp where we pad with one extra 0
        In0:    short int *, Input data
        In1:    short int *, Input data
        W, H:   Input data dimension [W x H]
        Out0:   short int *, Output data [W/2, H/2]
        Out1:   short int *, Output data [W/2, H/2]
*/
void KerReLUMaxPool2x2_2_Drop_fp(KerReLUMaxPool2x2_2_fpT *Arg);


/** @brief Template for rectification and pooling basic kernels, 3 tiles version

Template for rectification and pooling basic kernels, 3 tiles version
*/
typedef struct {
        Word16 * __restrict__ In0;	/**< Pointer to 1st input tile  */
        Word16 * __restrict__ In1;	/**< Pointer to 2nd input tile  */
        Word16 * __restrict__ In2;	/**< Pointer to 3rd input tile  */
        int W;				/**< Width of the tile. Input and output have the same */
        int H;				/**< Height of the tile. Input and output have the same */
        Word16 * __restrict__ Out0;	/**< Pointer to 1st output tile  */
        Word16 * __restrict__ Out1;	/**< Pointer to 2nd output tile  */
        Word16 * __restrict__ Out2;	/**< Pointer to 3rd output tile  */
} KerReLUMaxPool2x2_3_fpT;

/** @brief Linear rectification followed by 2x2 Max Pooling basic kernel, 3 output version.

Linear rectification followed by 2x2 Max Pooling basic kernel, 3 output version.

        Linear rectification followed by a 2x2 max pooling, triple output version
        In0:    short int *, Input data
        In1:    short int *, Input data
        In2:    short int *, Input data
        W, H:   Input data dimension [W x H]
        Out0:   short int *, Output data [W/2, H/2]
        Out1:   short int *, Output data [W/2, H/2]
        Out2:   short int *, Output data [W/2, H/2]
*/
void KerReLUMaxPool2x2_3_fp(KerReLUMaxPool2x2_3_fpT *Arg);

/** @brief Linear rectification followed by 2x2 Average Pooling basic kernel, 3 output version.

Linear rectification followed by 2x2 Average Pooling basic kernel, 3 output version.

        Linear rectification followed by a 2x2 Average pooling, triple output version
        In0:    short int *, Input data
        In1:    short int *, Input data
        In2:    short int *, Input data
        W, H:   Input data dimension [W x H]
        Out0:   short int *, Output data [W/2, H/2]
        Out1:   short int *, Output data [W/2, H/2]
        Out2:   short int *, Output data [W/2, H/2]
*/
void KerReLUAvgPool2x2_3_fp(KerReLUMaxPool2x2_3_fpT *Arg);

/* Dense layer*/

/** @brief Template for Linear Layer, max Q15 for Input, Coeff, Bias and Output

Template for Linear Layer
*/
typedef struct {
	Word16 * __restrict__ In;	/**< Pointer to input tile, 1D */
	int InSize;			/**< Size of the input tile */
	Word16 * __restrict__ Filter;	/**< Pointer to the coefficients, 1D, size is InSize*OutSize */
	unsigned int NormFilter;	/**< Fixed point format for Filter, must be <= 15 */
	Word16 *  __restrict__ Bias;	/**< Pointer to output bias, 1D, size of OutSize */
	unsigned int NormBias;		/**< Fixed point format for Bias, must be <= 15 */
	Word16 *  __restrict__ Out;	/**< Pointer to output tile, 1D, size is OutSize */
	int OutSize;			/**< Size of the output tile */
} KerLinearLayer_fpT;

/** @brief Linear Layer, In, Out, Filter and Bias use short ints.

Linear Layer, In, Out, Filter and Bias use short ints.

        Linear layer:   Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
        Outputs are evaluated in sequence, a given output is evaluated in parallel

        In      short int, input vector
        InSize  Number of inputs
        Filter  short int, Linear filter coefficients. Dimension: InSize*OutSize
        Out     short int, output vector
        OutSize output dimension
*/
void KerLinearLayer_fp(KerLinearLayer_fpT *Arg);

/** @brief Linear Layer, In, Out, Filter and Bias use short ints.

Linear Layer, In, Out, Filter and Bias use short ints.

        Linear layer:   Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
        Outputs are evaluated in parallel, a given output is evaluated in sequence

        In      short int, input vector
        InSize  Number of inputs
        Filter  short int, Linear filter coefficients. Dimension: InSize*OutSize
        Out     short int, output vector
        OutSize output dimension
*/
void KerLinearLayerParOut_fp(KerLinearLayer_fpT *Arg);

/** @brief Linear Layer followed by linear rectification, In, Out, Filter and Bias use short ints.

Linear Layer followed by linear rectification, In, Out, Filter and Bias use short ints.

        Linear layer:   Out[i] = ReLU((sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter)
        Outputs are evaluated in sequence, a given output is evaluated in parallel

        In      short int, input vector
        InSize  Number of inputs
        Filter  short int, Linear filter coefficients. Dimension: InSize*OutSize
        Out     short int, output vector
        OutSize output dimension
*/
void KerLinearLayerReLU_fp(KerLinearLayer_fpT *Arg);


/** @brief Template for Linear Layer, max Q15 for Input, Coeff, Bias. Max Q31 for Output.

Template for Linear Layer, max Q15 for Input, Coeff, Bias. Max Q31 for Output.
*/
typedef struct {
	Word16 * __restrict__ In;	/**< Pointer to input tile, 1D */
	int InSize;			/**< Size of the input tile */
	Word16 * __restrict__ Filter;	/**< Pointer to the coefficients, 1D, size is InSize*OutSize */
	unsigned int NormFilter;	/**< Fixed point format for Filter, must be <= 15 */
	Word16 *  __restrict__ Bias;	/**< Pointer to output bias, 1D, size of OutSize */
	unsigned int NormBias;		/**< Fixed point format for Bias, must be <= 15 */
	Word32 *  __restrict__ Out;	/**< Pointer to output tile, 1D, size is OutSize */
	int OutSize;			/**< Size of the output tile */
} KerLinearLayer_fpdT;

/** @brief Linear Layer. In, Filter and Bias use short ints. Out uses plain int.

Linear Layer. In, Filter and Bias use short ints. Out uses plain int.

        Linear layer:   Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
        Outputs are evaluated in sequence, a given output is evaluated in parallel

        In      short int, input vector
        InSize  Number of inputs
        Filter  short int, Linear filter coefficients. Dimension: InSize*OutSize
        Out     int, output vector
        OutSize output dimension
*/
void KerLinearLayer_fpd(KerLinearLayer_fpdT *Arg);

/** @brief Linear Layer followed by rectification. In, Filter and Bias use short ints. Out uses plain int.

Linear Layer followed by rectification. In, Filter and Bias use short ints. Out uses plain int.

        Linear layer:   Out[i] = ReLU((sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter)
        Outputs are evaluated in sequence, a given output is evaluated in parallel

        In      short int, input vector
        InSize  Number of inputs
        Filter  short int, Linear filter coefficients. Dimension: InSize*OutSize
        Out     int, output vector
        OutSize output dimension
*/
void KerLinearLayerReLU_fpd(KerLinearLayer_fpdT *Arg);

/** @brief Template for Linear Layer. Max Q15 for Input, Output and Bias. Max Q7 for Coeffs.

Template for Linear Layer. Max Q15 for Input, Output and Bias. Max Q7 for Coeffs.
*/
typedef struct {
	Word16 * __restrict__ In;	/**< Pointer to input tile, 1D */
	int InSize;			/**< Size of the input tile */
	Word8 * __restrict__ Filter;	/**< Pointer to the coefficients, 1D, size is InSize*OutSize */
	unsigned int NormFilter;	/**< Fixed point format for Filter, must be <= 15 */
	Word16 *  __restrict__ Bias;	/**< Pointer to output bias, 1D, size of OutSize */
	unsigned int NormBias;		/**< Fixed point format for Bias, must be <= 15 */
	Word16 *  __restrict__ Out;	/**< Pointer to output tile, 1D, size is OutSize */
	int OutSize;			/**< Size of the output tile */
} KerLinearLayer_fpsT;

/** @brief Linear Layer. In, Out and Bias use short ints. Filter uses byte.

Linear Layer. In, Out and Bias use short ints. Filter uses byte.

        Linear layer:   Out[i] = (sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter
        Outputs are evaluated in sequence, a given output is evaluated in parallel

        In      short int, input vector
        InSize  Number of inputs
        Filter  short int, Linear filter coefficients. Dimension: InSize*OutSize
        Out     int, output vector
        OutSize output dimension
*/
void KerLinearLayer_fps(KerLinearLayer_fpsT *Arg);

/** @brief Linear Layer followed by rectification. In, Out and Bias use short ints. Filter uses byte.

Linear Layer followed by rectification. In, Out and Bias use short ints. Filter uses byte.

        Linear layer:   Out[i] = ReLU((sum_product(In, Filter[i], Dim: InSize) + Bias[i]<<NormBias) >> NormFilter)
        Outputs are evaluated in sequence, a given output is evaluated in parallel

        In      short int, input vector
        InSize  Number of inputs
        Filter  short int, Linear filter coefficients. Dimension: InSize*OutSize
        Out     int, output vector
        OutSize output dimension
*/
void KerLinearLayerReLU_fps(KerLinearLayer_fpsT *Arg);


/** @} */ // End of CNNBasicKernels
/** @} */

#include "CNN_HwCE.h"

#endif
