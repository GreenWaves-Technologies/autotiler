/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the Apache License.  See the LICENSE file for details.
 *
 */


#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "StdTypes.h"
#include "Gap8.h"

#define NO_UPD	0
#define UPD		1
#define UPD1	1
#define UPD2	2
#define UPD3	3

static void CNN_LoadSWPlainConvolutionLibrary()

{
/* Padding */
	LibKernel("KerExpandTileInitH_fp", CALL_PARALLEL,
		CArgs(4,
			TCArg("signed short * __restrict__", "In"),
			TCArg("unsigned short int", "Win"),
			TCArg("unsigned short int", "Hin"),
			TCArg("unsigned short int", "Pad")),
		"ArgExpandTile_fp_T"
	);
    LibKernel("KerPadFirstTileH_fp", CALL_PARALLEL,
		CArgs(3,
			TCArg("signed short * __restrict__", "In"),
			TCArg("unsigned short int", "Win"),
			TCArg("unsigned short int", "Pad")),
		"ArgExpandTile_fp_T"
	);
    LibKernel("KerExpandTileH_fp", CALL_PARALLEL,
		CArgs(7,
			TCArg("signed short * __restrict__", "In"),
			TCArg("unsigned short int", "Win"),
			TCArg("unsigned short int", "Hin"),
			TCArg("signed short * __restrict__", "Out"),
			TCArg("unsigned short int", "Pad"),
			TCArg("unsigned char", "TileIndex"),
			TCArg("unsigned char", "NTile")),
		"ArgExpandTile_fp_T"
	);
// =============================================================================
        LibKernel("KerParSetBias_fp", CALL_PARALLEL,
                  CArgs(5,
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("short int * __restrict__", "Bias")
			),
		"KerParSetBias_fp_T"
		);

        LibKernel("KerParConv1x1Stride1_fp", CALL_PARALLEL,
                  CArgs(15,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Norm"),
			TCArg("unsigned short int", "TileIndex"),
			TCArg("unsigned short int", "NTile"),
			TCArg("unsigned short int", "Orientation"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "TileSize"),
			TCArg("unsigned short int", "TotalSize"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "S")
			),
		"KerParConv_fp_T"
		);
        LibKernel("KerParConv1x1Stride2_fp", CALL_PARALLEL,
                  CArgs(17,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Norm"),
			TCArg("unsigned short int", "TileIndex"),
			TCArg("unsigned short int", "NTile"),
			TCArg("unsigned short int", "Orientation"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "TileSize"),
			TCArg("unsigned short int", "TotalSize"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "S")
			),
		"KerParConv_fp_T"
		);
        LibKernel("KerParConv3x3Stride1_fp", CALL_PARALLEL,
                  CArgs(17,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Norm"),
			TCArg("unsigned short int", "TileIndex"),
			TCArg("unsigned short int", "NTile"),
			TCArg("unsigned short int", "Orientation"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "TileSize"),
			TCArg("unsigned short int", "TotalSize"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "S")
			),
		"KerParConv_fp_T"
		);
        LibKernel("KerParConv3x3Stride2_fp", CALL_PARALLEL,
                  CArgs(17,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Norm"),
			TCArg("unsigned short int", "TileIndex"),
			TCArg("unsigned short int", "NTile"),
			TCArg("unsigned short int", "Orientation"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "TileSize"),
			TCArg("unsigned short int", "TotalSize"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "S")
			),
		"KerParConv_fp_T"
		);
        LibKernel("KerParConv5x5Stride1_fp", CALL_PARALLEL,
                  CArgs(17,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Norm"),
			TCArg("unsigned short int", "TileIndex"),
			TCArg("unsigned short int", "NTile"),
			TCArg("unsigned short int", "Orientation"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "TileSize"),
			TCArg("unsigned short int", "TotalSize"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "S")
			),
		"KerParConv_fp_T"
		);
        LibKernel("KerParConv5x5Stride2_fp", CALL_PARALLEL,
                  CArgs(17,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Norm"),
			TCArg("unsigned short int", "TileIndex"),
			TCArg("unsigned short int", "NTile"),
			TCArg("unsigned short int", "Orientation"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "TileSize"),
			TCArg("unsigned short int", "TotalSize"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "S")
			),
		"KerParConv_fp_T"
		);
        LibKernel("KerParConvNxNStrideS_fp", CALL_PARALLEL,
                  CArgs(17,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Norm"),
			TCArg("unsigned short int", "TileIndex"),
			TCArg("unsigned short int", "NTile"),
			TCArg("unsigned short int", "Orientation"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "TileSize"),
			TCArg("unsigned short int", "TotalSize"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "S")
			),
		"KerParConv_fp_T"
		);
        LibKernel("KerParReLU_fp", CALL_PARALLEL,
                  CArgs(5,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("short int * __restrict__", "Out")
			),
		"KerParReLUMaxPool_fp_T"
		);
        LibKernel("KerParMaxPool2x2Stride2_fp", CALL_PARALLEL,
                  CArgs(7,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "DoReLU")
			),
		"KerParReLUMaxPool_fp_T"
		);
        LibKernel("KerParAvgPool2x2Stride2_fp", CALL_PARALLEL,
                  CArgs(7,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned short int", "DoReLU")
			),
		"KerParReLUMaxPool_fp_T"
		);
// =============================================================================
/* Add Feature Maps */
	LibKernel("KerAddFM_fp", CALL_PARALLEL,
		CArgs(4,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("int", "W"),
			TCArg("int", "H")
			),
	   	"KerAddFM_fpT"
	);
	LibKernel("KerAddFMReLu_fp", CALL_PARALLEL,
		CArgs(4,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("int", "W"),
			TCArg("int", "H")
			),
	   	"KerAddFM_fpT"
	);
/* Book keeping */
	LibKernel("KerSetInBias", CALL_PARALLEL,
		   CArgs(4,
			 TCArg("Word16 * __restrict__", "Out"),
			 TCArg("int", "W"),
			 TCArg("int", "H"),
			 TCArg("Word16", "Bias")
			),
		   "KerSetInBiasT"
	);
	LibKernel("KerSetInBiasStride2", CALL_PARALLEL,
		   CArgs(4,
			 TCArg("Word16 * __restrict__", "Out"),
			 TCArg("int", "W"),
			 TCArg("int", "H"),
			 TCArg("Word16", "Bias")
			),
		   "KerSetInBiasT"
	);
/* Plain SW convolutions, short int input, short int filters, short int output */
		LibKernel("KerConv1x1Stride2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConv1x1Stride2Multi_fp", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm"),
			TCArg("unsigned int", "InCh"),
			TCArg("Word32 * __restrict__", "Reduct")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConv3x3_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConv3x3Multi_fp", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm"),
			TCArg("unsigned int", "InCh"),
			TCArg("Word32 * __restrict__", "Reduct")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConv3x3Stride2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
		LibKernel("KerConv3x3Stride2Multi_fp", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm"),
			TCArg("unsigned int", "InCh"),
			TCArg("Word32 * __restrict__", "Reduct")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerReLUConv3x3Stride2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerReLUConv3x3_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerDirectConv3x3_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerDirectConv3x3Stride2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConv5x5_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConv5x5Stride2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerDirectConv5x5_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerDirectConv5x5Stride2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConvNxN_fp", CALL_PARALLEL,
		  CArgs(7,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm"),
			TCArg("int", "N")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerConvNxNStride2_fp", CALL_PARALLEL,
		  CArgs(7,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm"),
			TCArg("int", "N")
		       ),
		   "KerConv_fpT"
		);
	LibKernel("KerDirectConvNxN_fp", CALL_PARALLEL,
		  CArgs(7,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("Word16 * __restrict__", "Out"),
			TCArg("unsigned int", "Norm"),
			TCArg("int", "N")
		       ),
		   "KerConv_fpT"
		);
}

static void CNN_LoadSWReLUPoolingLibrary()

{
/* ReLU, Pooling */
	LibKernel("KerReLU_fp", CALL_PARALLEL,
		  CArgs(4,
			TCArg("short int * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out")
		  ),
		  "KerReLUMaxPool2x2_fpT"
	);
	LibKernel("KerMaxPool2x2_fp", CALL_PARALLEL,
		  CArgs(4,
			TCArg("short int * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out")
		  ),
		  "KerReLUMaxPool2x2_fpT"
	);
		LibKernel("KerMaxPool2x2Padded_fp", CALL_PARALLEL,
		  CArgs(7,
			TCArg("short int * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "Pad"),
			TCArg("unsigned char", "TileIndex"),
			TCArg("unsigned char", "NTile")
		  ),
		  "KerReLUMaxPool2x2Padded_fpT"
	);
	LibKernel("KerAvgPool2x2_fp", CALL_PARALLEL,
		  CArgs(4,
			TCArg("short int * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out")
		  ),
		  "KerReLUMaxPool2x2_fpT"
	);
	LibKernel("KerMaxPool3x3_fp", CALL_PARALLEL,
		  CArgs(4,
			TCArg("short int * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out")
		  ),
		  "KerReLUMaxPool2x2_fpT"
	);
	LibKernel("KerReLUMaxPool2x2_fp", CALL_PARALLEL,
		  CArgs(4,
			TCArg("short int * __restrict__", "In"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out")
		  ),
		  "KerReLUMaxPool2x2_fpT"
	);
	LibKernel("KerReLUMaxPool2x2_2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("short int * __restrict__", "In0"),
			TCArg("short int * __restrict__", "In1"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out0"),
			TCArg("short int * __restrict__", "Out1")
		  ),
		  "KerReLUMaxPool2x2_2_fpT"
	);
	LibKernel("KerReLUAvgPool2x2_2_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("short int * __restrict__", "In0"),
			TCArg("short int * __restrict__", "In1"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out0"),
			TCArg("short int * __restrict__", "Out1")
		  ),
		  "KerReLUMaxPool2x2_2_fpT"
	);
	LibKernel("KerReLUMaxPool2x2_2_Even_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("short int * __restrict__", "In0"),
			TCArg("short int * __restrict__", "In1"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out0"),
			TCArg("short int * __restrict__", "Out1")
		  ),
		  "KerReLUMaxPool2x2_2_fpT"
	);
	LibKernel("KerReLUMaxPool2x2_2_Drop_fp", CALL_PARALLEL,
		  CArgs(6,
			TCArg("short int * __restrict__", "In0"),
			TCArg("short int * __restrict__", "In1"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out0"),
			TCArg("short int * __restrict__", "Out1")
		  ),
		  "KerReLUMaxPool2x2_2_fpT"
	);
	LibKernel("KerReLUMaxPool2x2_3_fp", CALL_PARALLEL,
		  CArgs(8,
			TCArg("short int * __restrict__", "In0"),
			TCArg("short int * __restrict__", "In1"),
			TCArg("short int * __restrict__", "In2"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out0"),
			TCArg("short int * __restrict__", "Out1"),
			TCArg("short int * __restrict__", "Out2")
		  ),
		  "KerReLUMaxPool2x2_3_fpT"
	);

	LibKernel("KerReLUAvgPool2x2_3_fp", CALL_PARALLEL,
		  CArgs(8,
			TCArg("short int * __restrict__", "In0"),
			TCArg("short int * __restrict__", "In1"),
			TCArg("short int * __restrict__", "In2"),
			TCArg("int", "W"),
			TCArg("int", "H"),
			TCArg("short int * __restrict__", "Out0"),
			TCArg("short int * __restrict__", "Out1"),
			TCArg("short int * __restrict__", "Out2")
		  ),
		  "KerReLUMaxPool2x2_3_fpT"
	);
}

static void CNN_LoadDenseLayerLibrary()

{
/* Linear Layer */
	LibKernel("KerLinearLayer_fps", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "InSize"),
			TCArg("Word8 * __restrict__", "Filter"),
			TCArg("unsigned int", "NormFilter"),
			TCArg("Word16 *  __restrict__", "Bias"),
			TCArg("unsigned int", "NormBias"),
			TCArg("Word16 *  __restrict__", "Out"),
			TCArg("int", "OutSize")
		  ),
		  "KerLinearLayer_fpsT"
	);
	LibKernel("KerLinearLayer_fp", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "InSize"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("unsigned int", "NormFilter"),
			TCArg("Word16 *  __restrict__", "Bias"),
			TCArg("unsigned int", "NormBias"),
			TCArg("Word16 *  __restrict__", "Out"),
			TCArg("int", "OutSize")
		  ),
		  "KerLinearLayer_fpT"
	);
	LibKernel("KerLinearLayer_fpd", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "InSize"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("unsigned int", "NormFilter"),
			TCArg("Word16 *  __restrict__", "Bias"),
			TCArg("unsigned int", "NormBias"),
			TCArg("Word32 *  __restrict__", "Out"),
			TCArg("int", "OutSize")
		  ),
		  "KerLinearLayer_fpdT"
	);
	LibKernel("KerLinearLayerReLU_fps", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "InSize"),
			TCArg("Word8  * __restrict__", "Filter"),
			TCArg("unsigned int", "NormFilter"),
			TCArg("Word16 *  __restrict__", "Bias"),
			TCArg("unsigned int", "NormBias"),
			TCArg("Word16 *  __restrict__", "Out"),
			TCArg("int", "OutSize")
		  ),
		  "KerLinearLayer_fpsT"
	);
	LibKernel("KerLinearLayerReLU_fp", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "InSize"),
			TCArg("Word16  * __restrict__", "Filter"),
			TCArg("unsigned int", "NormFilter"),
			TCArg("Word16 *  __restrict__", "Bias"),
			TCArg("unsigned int", "NormBias"),
			TCArg("Word16 *  __restrict__", "Out"),
			TCArg("int", "OutSize")
		  ),
		  "KerLinearLayer_fpT"
	);
	LibKernel("KerLinearLayerReLU_fpd", CALL_PARALLEL,
		  CArgs(8,
			TCArg("Word16 * __restrict__", "In"),
			TCArg("int", "InSize"),
			TCArg("Word16 * __restrict__", "Filter"),
			TCArg("unsigned int", "NormFilter"),
			TCArg("Word16 *  __restrict__", "Bias"),
			TCArg("unsigned int", "NormBias"),
			TCArg("Word32 *  __restrict__", "Out"),
			TCArg("int", "OutSize")
		  ),
		  "KerLinearLayer_fpdT"
	);
}

void CNN_LoadSoftwareKernelLibrary()

{
	CNN_LoadSWPlainConvolutionLibrary();
	CNN_LoadSWReLUPoolingLibrary();
	CNN_LoadDenseLayerLibrary();
}

static void CNN_LoadHWCEBookKeepingLibrary()

{
        LibKernel("HWCE_Enable",  CALL_SEQUENTIAL, CArgs(0), "");
        LibKernel("HWCE_Disable", CALL_SEQUENTIAL, CArgs(0), "");
        LibKernel("HWCE_GenericInit", CALL_SEQUENTIAL,
                   CArgs(3,
                        TCArg("unsigned int", "ConvType"),
                        TCArg("unsigned int", "WStride"),
                        TCArg("unsigned int", "Norm")
                   ),
                   ""
        );
        LibKernel("HwCE_SetYinMode", CALL_SEQUENTIAL,
                CArgs(1,
                        TCArg("unsigned int", "Disable")
                ),
                ""
        );
}

static void CNN_LoadHWCEConvolutionLibrary()

{
        LibKernel("KerSetInBias2", CALL_PARALLEL,
                   CArgs(6,
                         TCArg("Word16 * __restrict__", "Out0"),
                         TCArg("Word16 * __restrict__", "Out1"),
                         TCArg("int", "W"),
                         TCArg("int", "H"),
                         TCArg("Word16", "Bias0"),
                         TCArg("Word16", "Bias1")
                        ),
                   "KerSetInBias2T"
        );
        LibKernel("KerSetInBias3", CALL_PARALLEL,
                   CArgs(8,
                         TCArg("Word16 * __restrict__", "Out0"),
                         TCArg("Word16 * __restrict__", "Out1"),
                         TCArg("Word16 * __restrict__", "Out2"),
                         TCArg("int", "W"),
                         TCArg("int", "H"),
                         TCArg("Word16", "Bias0"),
                         TCArg("Word16", "Bias1"),
                         TCArg("Word16", "Bias2")
                        ),
                   "KerSetInBias3T"
        );
        LibKernel("HWCE_ProcessOneTile3x3_MultiOut", CALL_SEQUENTIAL,
                  CArgs(9,
                        TCArg("Word16 * __restrict__", "In"),
                        TCArg("Word16 * __restrict__", "Out0"),
                        TCArg("Word16 * __restrict__", "Out1"),
                        TCArg("Word16 * __restrict__", "Out2"),
                        TCArg("Word16 * __restrict__", "Filter"),
                        TCArg("Word16", "Bias"),
                        TCArg("unsigned int", "W"),
                        TCArg("unsigned int", "H"),
                        TCArg("unsigned int", "OutMask")
                       ),
                   ""
                );
        LibKernel("HWCE_ProcessOneTile5x5", CALL_SEQUENTIAL,
                  CArgs(6,
                        TCArg("Word16 * __restrict__", "In"),
                        TCArg("Word16 * __restrict__", "Out"),
                        TCArg("Word16 * __restrict__", "Filter"),
                        TCArg("Word16", "Bias"),
                        TCArg("unsigned int", "W"),
                        TCArg("unsigned int", "H")
                       ),
                   ""
                );
        LibKernel("HWCE_ProcessOneTile7x7", CALL_SEQUENTIAL,
                  CArgs(6,
                        TCArg("Word16 * __restrict__", "In"),
                        TCArg("Word16 * __restrict__", "Out"),
                        TCArg("Word16 * __restrict__", "Filter"),
                        TCArg("Word16", "Bias"),
                        TCArg("unsigned int", "W"),
                        TCArg("unsigned int", "H")
                       ),
                   ""
                );
        LibKernel("HWCE_ProcessOneTile7x4", CALL_SEQUENTIAL,
                  CArgs(6,
                        TCArg("Word16 * __restrict__", "In"),
                        TCArg("Word16 * __restrict__", "Out"),
                        TCArg("Word16 * __restrict__", "Filter"),
                        TCArg("Word16", "Bias"),
                        TCArg("unsigned int", "W"),
                        TCArg("unsigned int", "H")
                       ),
                   ""
                );
}


void CNN_LoadHWCEKernelLibrary()

{
	CNN_LoadHWCEBookKeepingLibrary();
	CNN_LoadHWCEConvolutionLibrary();
}

/* Pure SW convolutions (Without Accumulation with the output plane) */
void CNN_TiledPureConvNxN_SW_fp(char *Name, unsigned int FS, unsigned int Width, unsigned int Height)

{
	char *ConvKerName;

	switch (FS) {
		case 3:
			ConvKerName = "KerDirectConv3x3_fp"; break;
		case 5:
			ConvKerName = "KerDirectConv5x5_fp"; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("TiledPureConvNxN_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerDirectConvNxN_fp"; break;
	}
        UserKernel(Name,
                KernelDimensions(1, Width, Height, 1),
                KernelIterationOrder(KER_DIM2, KER_TILE),
                TILE_HOR,
                CArgs(5,
                      TCArg("short int * __restrict__", "In"),
                      TCArg("short int * __restrict__", "Filter"),
                      TCArg("short int * __restrict__", "Out"),
                      TCArg("unsigned int",             "Norm"),
                      TCArg("int",                      "N")
                     ),
                Calls(1,
                        Call(ConvKerName, LOC_INNER_LOOP,
                                Bindings(7,
                                         Bind("In",     BIND_K_ARG, "In",               KER_ARG_TILE),
                                         Bind("W",      BIND_K_ARG, "In",               KER_ARG_TILE_W),
                                         Bind("H",      BIND_K_ARG, "In",               KER_ARG_TILE_H),
                                         Bind("Filter", BIND_K_ARG, "Filter",           KER_ARG_TILE),
                                         Bind("Out",    BIND_K_ARG, "Out",              KER_ARG_TILE),
                                         Bind("Norm",   BIND_C_ARG, "Norm",             TC_ARG),
                                         Bind("N",      BIND_C_ARG, "N",                TC_ARG)
                                        )
                            )
                     ),
                KerArgs(3,
                        KerArg("In",     OBJ_IN_DB,             Width,        Height,        sizeof(short int), FS-1, 0, 0, "In", 0),
                        KerArg("Filter", OBJ_BUFFER_IN_NTILED,  FS,           FS,            sizeof(short int), 0,    0, 0, "Filter", 0),
                        KerArg("Out",    OBJ_OUT_DB,            (Width-FS+1), (Height-FS+1), sizeof(short int), 0,    0, 0, "Out", 0)
                        )
        );
}

/* SW convolution layer (Out is init to bias and then accumulated) */
void CNN_TiledConvNxN_SW_fp(char *Name, unsigned int FS, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height)

{
	char *ConvKerName;
	int Pad = 0;

	switch (FS) {
		case 1:
			ConvKerName = "KerConv1x1_fp"; 
			Pad = 0; break;
		case 3:
			ConvKerName = "KerConv3x3_fp"; 
			Pad = 1; break;
		case 5:
			ConvKerName = "KerConv5x5_fp"; 
			Pad = 2; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("TiledConvNxN_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerConvNxNStride2_fp"; break;
	}

	// UserKernel(Name,
	// 	KernelDimensions(InPlane, Width, Height, OutPlane),
	// 	KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
	// 	TILE_HOR,
	// 	CArgs(5,
	// 	      TCArg("short int * __restrict__", "In"),
	// 	      TCArg("short int * __restrict__", "Filter"),
	// 	      TCArg("short int * __restrict__", "Out"),
	// 	      TCArg("unsigned int", 			"Norm"),
	// 	      TCArg("short int * __restrict__",	"Bias")
	// 	     ),
	// 	Calls(5,
	// 		Call("KerExpandTileInitH_fp", LOC_PROLOG,
	// 			Bindings(4, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W),  K_Arg("InBuff", KER_ARG_TILE_H0), Imm(Pad))),

	// 		Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
	// 			Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),

	// 		Call("KerPadFirstTileH_fp", LOC_INNER_LOOP_PROLOG,
	// 			Bindings(3, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W), Imm(Pad))),

	// 		Call("KerExpandTileH_fp", LOC_INNER_LOOP,
	// 			Bindings(7, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("InBuff", KER_ARG_TILE), Imm(Pad), K_Arg("In", KER_ARG_TILEINDEX), K_Arg("In", KER_ARG_NTILES))),

	// 		Call(ConvKerName, LOC_INNER_LOOP,
	// 			Bindings(6, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W), K_Arg("InBuff", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), C_Arg("Norm")))
	// 	     ),
	// 	KerArgs(4,
	// 		KerArgP("In",        OBJ_IN_DB_3D,      	Width,   Height,     0, Pad, sizeof(short int), FS-1,   0, 0, "In",     0),
	// 		KerArg("Filter",     OBJ_IN_DB_NTILED_4D,   FS,      FS,                 sizeof(short int), 0,      0, 0, "Filter", 0),
 //            KerArgP("InBuff",    OBJ_BUFFER_ONETILE,    Width,	 Height,   Pad, Pad, sizeof(short int), FS-1,   0, 0, "",       0),
	// 		KerArg("Out",     	 OBJ_OUT_DB_3D,   		Width,   Height,			 sizeof(short int), 0, 		0, 0, "Out", 	0)
	// 		)
	// );


/********************************** NON PADDED ****************************/
	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 			"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		     ),
		Calls(2,
			Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
				Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H),
					    K_Arg("Filter", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), C_Arg("Norm")))
		     ),
		KerArgs(3,
			KerArg("In", 	 OBJ_IN_DB_3D,     		Width,      Height,      	sizeof(short int), FS-1, 0, 0, "In", 	0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D,	FS,	    	FS, 	 		sizeof(short int), 0,    0, 0, "Filter",0),
			KerArg("Out", 	 OBJ_OUT_DB_3D,			Width-FS+1, Height-FS+1, 	sizeof(short int), 0,    0, 0, "Out", 	0)
			)
	);

}


/* SW convolution layer (Out is init to bias and then accumulated) */
void CNN_TiledConvNxNMulti_SW_fp(char *Name, unsigned int FS, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height)

{
	char *ConvKerName = "KerConv3x3Multi_fp";

	UserKernel(Name,
		KernelDimensions(1, Width*InPlane, Height, OutPlane),
		KernelIterationOrder(KER_DIM3, KER_OUT_PLANE, KER_TILE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 			"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		     ),
		Calls(2,
			Call("KerSetInBias", LOC_INNER_LOOP,
				Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(8, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H),
					    K_Arg("Filter", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), C_Arg("Norm"), Imm(InPlane), K_Arg("Reduct", KER_ARG_TILE)))
		     ),
		KerArgs(4,
			KerArg("In", 	 OBJ_BUFFER_IN_3D,     	InPlane*Width,  Height,     	sizeof(short int), 	0, 	0, 0, "In", 	0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D,	InPlane*FS,	    FS, 	 		sizeof(short int), 	0,   0, 0, "Filter",	0),
			KerArg("Reduct", OBJ_BUFFER_ONETILE,	(Width-FS+1)*(Height-FS+1), 8,	sizeof(int), 		0, 	0, 0, "", 		0),
			KerArg("Out", 	 OBJ_BUFFER_OUT_3D,		Width-FS+1,		Height-FS+1, 	sizeof(short int), 	0,   0, 0, "Out", 	0)
			)
	);
}

/* SW convolution layer (Out is init to bias and then accumulated) */
void CNN_TiledConvNxNStride2_SW_fp( char *Name, 
									unsigned int FS, 
									unsigned int InPlane, 
									unsigned int OutPlane, 
									unsigned int Width, 
									unsigned int Height) {
	char *ConvKerName;
	switch (FS) {
		case 1:
			ConvKerName = "KerConv1x1Stride2_fp"; break;
		case 3:
			ConvKerName = "KerConv3x3Stride2_fp"; break;
		case 5:
			ConvKerName = "KerConv5x5Stride2_fp"; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("TiledConvNxN_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerConvNxNStride2_fp"; break;
	}
	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 			"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		     ),
		Calls(2,
			Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
				Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H),
					    K_Arg("Filter", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), C_Arg("Norm")))
		     ),
		KerArgs(3,
			KerArg("In", 	 OBJ_IN_DB_3D,     		Width,          Height,          sizeof(short int), FS-1, 0, 0, "In", 		0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D,	FS,	        	FS, 	         sizeof(short int), 0,    0, 0, "Filter", 	0),
			KerArg("Out", 	 OBJ_OUT_DB_3D,			(Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,    0, 0, "Out", 		0)
			)
	);
}

/* SW convolution layer (Out is init to bias and then accumulated) */
void CNN_TiledConvNxNStride2Multi_SW_fp( char *Name, 
									unsigned int FS, 
									unsigned int InPlane, 
									unsigned int OutPlane, 
									unsigned int Width, 
									unsigned int Height) {

	char *ConvKerName = "KerConv1x1Stride2Multi_fp";

	int strideW = (Width-FS+1)/2;
	if((Width-FS+1)&0x1) strideW++;
	int strideH = (Height-FS+1)/2;
	if((Height-FS+1)&0x1) strideH++;
	
	UserKernel(Name,
		KernelDimensions(1, Width*InPlane, Height, OutPlane),
		KernelIterationOrder(KER_DIM3, KER_OUT_PLANE, KER_TILE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 			"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		    ),
		Calls(2,
			Call("KerSetInBias", LOC_INNER_LOOP,
				Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(8, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), C_Arg("Norm"), Imm(InPlane), K_Arg("Reduct", KER_ARG_TILE)))
		    ),
		KerArgs(4,
			KerArg("In",      OBJ_BUFFER_IN_3D,     Width*InPlane,	Height, 	sizeof(short int), 	0,	0, 0, "In", 	0),
			KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS*InPlane,	   	FS, 		sizeof(short int), 	0, 	0, 0, "Filter", 0),
			KerArg("Reduct",  OBJ_BUFFER_ONETILE,	strideW*strideH,8,			sizeof(int), 		0, 	0, 0, "", 		0),
			KerArg("Out",     OBJ_OUT_DB_3D,   		strideW, 		strideH,	sizeof(short int), 	0, 	0, 0, "Out", 	0)
			)
	);

}

/* Software Composite kernels: Convolution, ReLU then Max or Avg Pooling */
void CNN_TiledConvNxNReLUPool2x2_SW_fp(char *Name, unsigned int FS, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height, unsigned int PoolMax)

{
	char *ConvKerName, *KerReLUPoolName;

	switch (FS) {
		case 3:
			ConvKerName = "KerConv3x3_fp"; break;
		case 5:
			ConvKerName = "KerConv5x5_fp"; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("TiledConvNxNReLUMaxPool2x2_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerConvNxN_fp"; break;
	}
	switch (PoolMax) {
		case 0:	/* Relu, PoolAvg */
			KerReLUPoolName = "KerReLUAvgPool2x2_fp"; break;
		case 1:	/* Relu, PoolMax */
			KerReLUPoolName = "KerReLUMaxPool2x2_fp"; break;
		case 2:	/* No Relu, PoolAvg */
			KerReLUPoolName = "KerAvgPool2x2_fp"; break;
		case 3:	/* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool2x2_fp"; break;
		case 4: /* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool3x3_fp"; break;
		case 5: /* Relu, No PoolMax */
			KerReLUPoolName = "KerReLU_fp"; break;
	}
	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		     ),
		Calls(3,
			Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H),
					    K_Arg("Filter", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE), C_Arg("Norm"))),
			Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H),
					    K_Arg("Out", KER_ARG_TILE)))
		     ),
		KerArgs(4,
			KerArg("In",      OBJ_IN_DB_3D,     	Width,	    	Height, 	 sizeof(short int), FS-1,OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS,	    	FS, 		 sizeof(short int), 0, 	0, 0, "Filter", 0),
			// KerArg("BiasOut", OBJ_BUFFER_ONETILE,	Width-FS+1,	Height-FS+1,	 sizeof(short int), 0, 	0, 0, "", 0),
			KerArg("BiasOut",  O_BUFF|O_NTILED|O_ONETILE,	Width-FS+1,	Height-FS+1,	 sizeof(short int), 0, 	0, 0, "", 0),
			KerArg("Out",     OBJ_OUT_DB_3D,   	(Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0, 	0, 0, "Out", 0)
			)
	);
}

/* Software Composite kernels: Convolution, ReLU then Max or Avg Pooling */
void CNN_TiledConvNxNStride2ReLUPool2x2_SW_fp(	char *Name, 
						unsigned int FS, 
						unsigned int InPlane, 
						unsigned int OutPlane, 
						unsigned int Width, 
						unsigned int Height, 
						unsigned int PoolMax) {
	char *ConvKerName, *KerReLUPoolName;
    int Pad = 0;

	switch (FS) {
		case 3:
			ConvKerName = "KerConv3x3Stride2_fp"; 
			Pad = 1; break;
		case 5:
			ConvKerName = "KerConv5x5Stride2_fp"; 
			Pad = 2; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("TiledConvNxNStride2ReLUMaxPool2x2_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerConvNxNStride2_fp"; break;
	}
	switch (PoolMax) {
		case 0:	/* Relu, PoolAvg */
			KerReLUPoolName = "KerReLUAvgPool2x2_fp"; break;
		case 1:	/* Relu, PoolMax */
			KerReLUPoolName = "KerReLUMaxPool2x2_fp"; break;
		case 2:	/* No Relu, PoolAvg */
			KerReLUPoolName = "KerAvgPool2x2_fp"; break;
		case 3:	/* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool2x2_fp"; break;
		case 4: /* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool3x3_fp"; break;
		case 5: /* Relu, No PoolMax */
			KerReLUPoolName = "KerReLU_fp"; break;
	}
	// UserKernel(Name,
	// 	KernelDimensions(InPlane, Width, Height, OutPlane),
	// 	KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
	// 	TILE_HOR,
	// 	CArgs(5,
	// 		TCArg("short int * __restrict__", "In"),
	// 		TCArg("short int * __restrict__", "Filter"),
	// 		TCArg("short int * __restrict__", "Out"),
	// 		TCArg("unsigned int", 			"Norm"),
	// 		TCArg("short int * __restrict__",	"Bias")
	// 	    ),
	// 	Calls(6,
	// 		Call("KerExpandTileInitH_fp", LOC_PROLOG,
	// 			Bindings(4, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W),  K_Arg("InBuff", KER_ARG_TILE_H0), Imm(Pad))),

	// 		Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
	// 			Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),

	// 		Call("KerPadFirstTileH_fp", LOC_INNER_LOOP_PROLOG,
	// 			Bindings(3, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W), Imm(Pad))),

	// 		Call("KerExpandTileH_fp", LOC_INNER_LOOP,
	// 			Bindings(7, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("InBuff", KER_ARG_TILE), Imm(Pad), K_Arg("In", KER_ARG_TILEINDEX), K_Arg("In", KER_ARG_NTILES))),

	// 		Call(ConvKerName, LOC_INNER_LOOP,
 //                Bindings(6, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W), K_Arg("InBuff", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE), C_Arg("Norm"))),

	// 		Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
	// 			Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H),K_Arg("Out", KER_ARG_TILE)))
	// 	    ),
	// 	KerArgs(5,
 //            KerArgP("In",        OBJ_IN_DB_3D,      	Width,   Height,     0, Pad, sizeof(short int), FS-1,   OBJ_CONSTRAINTS_DROP_REM, 4, "In",     0),
 //            KerArg ("Filter",    OBJ_IN_DB_NTILED_4D,   FS,      FS,                 sizeof(short int), 0,      0,                        0, "Filter", 0),
 //            KerArgP("InBuff",    OBJ_BUFFER_ONETILE,    Width,	 Height,   Pad, Pad, sizeof(short int), FS-1,   OBJ_CONSTRAINTS_DROP_REM, 0, "",       0),
 //            KerArg ("BiasOut",   OBJ_BUFFER_ONETILE,    Width/2, Height/2,           sizeof(short int), 0,      OBJ_CONSTRAINTS_DROP_REM, 0, "",       0),
 //            KerArg ("Out",       OBJ_OUT_DB_3D,         Width/4, Height/4,           sizeof(short int), 0,      0,                        0, "Out",    0)
	// 		)
	// 	);


/********************************** NON PADDED ****************************/
		UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 			"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		    ),
		Calls(3,
			Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE), C_Arg("Norm"))),
			Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H),K_Arg("Out", KER_ARG_TILE)))
		    ),
		KerArgs(4,
			KerArg("In",      OBJ_IN_DB_3D,     	Width,	    	Height, 			sizeof(short int), FS-1,	0, 0, "In", 	0),
			KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS,	   			FS, 				sizeof(short int), 0, 		0, 0, "Filter", 0),
			KerArg("BiasOut", OBJ_BUFFER_ONETILE,	(Width-FS+1)/2,	(Height-FS+1)/2,	sizeof(short int), 0, 		0, 0, "", 		0),
			KerArg("Out",     OBJ_OUT_DB_3D,   		(Width-FS+1)/4, (Height-FS+1)/4,	sizeof(short int), 0, 		0, 0, "Out", 	0)
			)
	);

}

/* Software Composite kernels: Convolution, ReLU then Max or Avg Pooling */
void CNN_TiledPaddedConvNxNStride2ReLUPool2x2_SW_fp(char *Name, 
													unsigned int FS, 
													unsigned int InPlane, 
													unsigned int OutPlane, 
													unsigned int Width, 
													unsigned int Height, 
													unsigned int PoolMax) {
	char *ConvKerName, *KerReLUPoolName;
    int Pad = 0;
    int NextPad = 1;

	switch (FS) {
		case 3:
			ConvKerName = "KerConv3x3Stride2_fp"; 
			Pad = 1; break;
		case 5:
			ConvKerName = "KerConv5x5Stride2_fp"; 
			Pad = 2; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("TiledConvNxNStride2ReLUMaxPool2x2_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerConvNxNStride2_fp"; break;
	}
	switch (PoolMax) {
		case 0:	/* Relu, PoolAvg */
			KerReLUPoolName = "KerReLUAvgPool2x2_fp"; break;
		case 1:	/* Relu, PoolMax */
			KerReLUPoolName = "KerReLUMaxPool2x2_fp"; break;
		case 2:	/* No Relu, PoolAvg */
			KerReLUPoolName = "KerAvgPool2x2_fp"; break;
		case 3:	/* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool2x2Padded_fp"; break;
		case 4: /* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool3x3_fp"; break;
		case 5: /* Relu, No PoolMax */
			KerReLUPoolName = "KerReLU_fp"; break;
	}

	UserKernel(Name,
	KernelDimensions(InPlane, Width, Height, OutPlane),
	KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
	TILE_HOR,
	CArgs(5,
	      TCArg("short int * __restrict__", "In"),
	      TCArg("short int * __restrict__", "Filter"),
	      TCArg("short int * __restrict__", "Out"),
	      TCArg("unsigned int", 			"Norm"),
	      TCArg("short int * __restrict__",	"Bias")
	    ),
	Calls(3,
		Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
			Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
		Call(ConvKerName, LOC_INNER_LOOP,
			Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE), C_Arg("Norm"))),
		Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
			Bindings(7, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H),K_Arg("Out", KER_ARG_TILE), Imm(NextPad), K_Arg("In", KER_ARG_TILEINDEX), K_Arg("In", KER_ARG_NTILES)))
	    ),
	KerArgs(4,
		KerArg("In",   OBJ_IN_DB_3D,     	Width,	 Height, 										sizeof(short int), FS-1,	0, 48, "In", 	0),
		// // KerArgP("In",     OBJ_IN_DB_3D,      	Width,   Height,     Pad, Pad, 							sizeof(short int), FS-1,   OBJ_CONSTRAINTS_DROP_REM, 4, "In",     0),
		// KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS,	   						FS, 							sizeof(short int), 0, 		0, 0, "Filter", 0),
		// KerArg("BiasOut", OBJ_BUFFER_ONETILE,	(Width-2*Pad)/2,			(Height-2*Pad)/2,				sizeof(short int), 0, 		0, 0, "", 		0),
		// KerArg("Out",     OBJ_OUT_DB_3D,   		((Width-2*Pad)/4)+2*NextPad,((Height-2*Pad)/4)+2*NextPad,	sizeof(short int), 0, 		0, 0, "Out", 	0)

	    // KerArgP("In",        OBJ_IN_DB_3D,      	Width,  Height,     2, 		2, 		sizeof(short int), FS-1,   OBJ_CONSTRAINTS_DROP_REM, 	51, "In",     0),
        KerArg("Filter",    OBJ_IN_DB_NTILED_4D,   FS,     FS,                 				sizeof(short int), 0,      0,                        	0, "Filter", 0),
        // KerArgP("InBuff",    OBJ_BUFFER_ONETILE,    Width,	 Height,   Pad, Pad, sizeof(short int), FS-1,   OBJ_CONSTRAINTS_DROP_REM, 0, "",       0),
        KerArg("BiasOut",   OBJ_BUFFER_ONETILE,    100, 	100,           						sizeof(short int), 0,      0, 	0, "",       0),
        // KerArg ("Out",       OBJ_OUT_DB_3D,         Width/4, Height/4,           sizeof(short int), 0,      0,                        0, "Out",    0)
       	// KerArgP("Out",    	OBJ_OUT_DB_3D,   		50,	 	50,   		1, 	1, 	sizeof(short int), 0,   	0, 							0, "Out",       0)
       	KerArg("Out",     OBJ_OUT_DB_3D,   		52,52,	sizeof(short int), 0, 		0, 0, "Out", 	0)

		)
	);

}


// =============================================================================


void LargeParOutFeatConvolutionPoolReLU_Ver_fp(
			char         *Name,

                        unsigned int InFeat,
                        unsigned int OutFeat,
                        unsigned int Width,
                        unsigned int Height,

                        unsigned int FSc,
			unsigned int ConvStride,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSp,
			unsigned int PoolStride,
			int          PoolDoPad,
			int          PoolDoReLU,

			int	     DoPool
			)

{
	if (DoPool==0) {
		FSp=1; PoolStride=1;
	}
	int NeedConvDim=0, NeedConvStride=0;
	unsigned int TileCons = ConvStride * PoolStride;
        int Overlap = FSc + ConvStride*(FSp-PoolStride-1);
	unsigned int Wo, Ho, Wc, Hc;
	int PadTc, PadBc, PadTp, PadBp;
	v4s PadInp = (v4s){0,0,0,0};
	char *ConvKerName, *PoolKerName;
	char *ReLUKerName = "KerParReLU_fp";

	switch (FSc) {
		case 1:
			if (ConvStride==1) ConvKerName = "KerParConv1x1Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv1x1Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		case 3:
			if (ConvStride==1) ConvKerName = "KerParConv3x3Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv3x3Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		case 5:
			if (ConvStride==1) ConvKerName = "KerParConv5x5Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv5x5Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		default:
			ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			break;
	}
	if (DoPool) {
		switch (FSp) {
			case 2:
				if (PoolStride==2) PoolKerName = "KerParMaxPool2x2Stride2_fp";
				else GenTilingError("Unsupported Pool Stride\n");
				break;
			case 3:
				GenTilingError("Unsupported Pooling\n");
				break;
			default:
				GenTilingError("Unsupported Pooling\n");
				break;
		}
	}

	if (ConvDoPad) {
		PadTc = (FSc-1)/2; PadBc = FSc/2;
		Wc = (Width- FSc+((FSc-1)/2)+(FSc/2))/ConvStride + 1;
        	Hc = (Height-FSc+((FSc-1)/2)+(FSc/2))/ConvStride + 1;
	} else {
		PadTc = 0; PadBc = 0;
		Wc = (Width- FSc)/ConvStride + 1;
        	Hc = (Height-FSc)/ConvStride + 1;
	}
	if (DoPool) {
		if (ConvDoPad) {
			PadTp = (FSp-1)/2; PadBp = FSp/2;
			PadInp = (v4s){0,0,PadTp,PadBp};
			Wo = (Wc-FSp+((FSp-1)/2)+(FSp/2))/PoolStride + 1;
        		Ho = (Hc-FSp+((FSp-1)/2)+(FSp/2))/PoolStride + 1;
		} else {
			PadTp = 0; PadBp = 0;
			Wo = (Wc-FSp)/PoolStride + 1;
        		Ho = (Hc-FSp)/PoolStride + 1;
		}
	} else {
		PadTp = 0; PadBp = 0;
		Wo = Wc; Ho = Hc;
	}


        UserKernel(Name,
                // KernelDimensions(1, Width, Height, 1),
                // KernelDimensions(InFeat, Width, Height, OutFeat),
		KernelDimensionsAndUserSymbols(InFeat, Width, Height, OutFeat,
			KerDynamicSymbols(4,
				S_Dyn("Wo", Wo),
				S_Dyn("Ho", Ho),
				S_Dyn("Wc", Wc),
				S_Dyn("Hc", Hc)
			)
		),
                KernelIterationOrder(KER_DIM3, KER_TILE, KER_TILE1),
                TILE_VER,
                CArgs(5,
                      TCArg("short int * __restrict__", "In"),
                      TCArg("short int * __restrict__", "Filter"),
					  TCArg("short int * __restrict__", "Out"),
					  TCArg("unsigned int",             "Norm"),
                      TCArg("short int * __restrict__", "Bias")
                     ),
                Calls(3,
			Call("KerParSetBias_fp", LOC_INNER_LOOP1_PROLOG,
                                Bindings(5, 
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
					UserSymb("Hc"), // Imm(Wc),
					KerDim(K_OUTP), // Imm(OutFeat),
					K_Arg("Bias", KER_ARG_TILE)
				)
			),
                        Call(ConvKerName, LOC_INNER_LOOP1,
                                Bindings(17, 
					K_Arg("In", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_W),
					KerDim(K_H), // Imm(Width),
					KerDim(K_INP), // Imm(InFeat),
					K_Arg("Filter", KER_ARG_TILE_H),
					K_Arg("Filter", KER_ARG_TILE_BASE),
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					C_Arg("Norm"),
					K_Arg("In", KER_ARG_TILEINDEX),
					K_Arg("In", KER_ARG_NTILES),
					Imm(0),
					K_Arg("In", KER_ARG_PAD),
					K_Arg("In", KER_ARG_TILE_W0),
					K_Arg("In", KER_ARG_W),
					NeedConvDim?Imm(FSc):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStride):AT_IGNORE_ARG_BINDING
					)
			),
			ConvDoReLU?
			Call(ReLUKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(5,
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
					UserSymb("Hc"), // Imm(Wc),
					KerDim(K_OUTP), // Imm(OutFeat),
					K_Arg("Out", KER_ARG_TILE)
				)
			):(DoPool?
			Call(PoolKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(7,
					K_Arg("ConvOut", KER_ARG_TILE),
					K_Arg("ConvOut", KER_ARG_TILE_W),
					UserSymb("Hc"), // Imm(Wc),
					KerDim(K_OUTP), // Imm(OutFeat),
					K_Arg("Out", KER_ARG_TILE),
					Imm((int)PadInp),
					Imm(PoolDoReLU)
				)
			):AT_NO_CALL)
                     ),
                KerArgs(5,
			ConvDoPad?
			KerArgP("In",      OBJ_IN_DB,            Width,  Height*InFeat,  PadTc, PadBc, 0, 0, sizeof(short int),         Overlap, ConvDoPad?OBJ_CONSTRAINTS_PAD_REM:OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0):
			KerArg ("In",      OBJ_IN_DB,            Width,  Height*InFeat,                      sizeof(short int),         Overlap, ConvDoPad?OBJ_CONSTRAINTS_PAD_REM:OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
                        KerArg ("Filter",  OBJ_IN_DB|O_TILE1,    InFeat, OutFeat,                            FSc*FSc*sizeof(short int), 0,       OBJ_CONSTRAINTS_TILE_HOR,     				     8,        "Filter", 0),
                        KerArg ("Bias",    OBJ_BUFFER_IN_NTILED, 1,      OutFeat,                            sizeof(short int),         0,       0,                        				     0,        "Bias", 0),
                        KerArg ("Out",     OBJ_OUT_DB,           Wo,     Ho*OutFeat,                         sizeof(short int),         0,       0,                        				     0,        "Out", 0),
			DoPool?
			KerArg ("ConvOut", OBJ_BUFFER_ONETILE,   Wc,     Hc*OutFeat,                         sizeof(short int),         0,       0,                                                          0,         "", 0)
			:AT_NO_KER_ARG
                        )
        );
}

void LargeParOutFeatConvolutionPoolReLU_Hor_fp(
			char         *Name,

                        unsigned int InFeat,
                        unsigned int OutFeat,
                        unsigned int Width,
                        unsigned int Height,

                        unsigned int FSc,
			unsigned int ConvStride,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSp,
			unsigned int PoolStride,
			int          PoolDoPad,
			int          PoolDoReLU,

			int	     DoPool
			)

{
	if (DoPool==0) {
		FSp=1; PoolStride=1;
	}
	int NeedConvDim=0, NeedConvStride=0;
	unsigned int TileCons = ConvStride * PoolStride;
        int Overlap = FSc + ConvStride*(FSp-PoolStride-1);
	unsigned int Wo, Ho, Wc, Hc;
	int PadTc, PadBc, PadTp, PadBp;
	v4s PadInp = (v4s){0,0,0,0};
	char *ConvKerName, *PoolKerName;
	char *ReLUKerName = "KerParReLU_fp";

	switch (FSc) {
		case 1:
			if (ConvStride==1) ConvKerName = "KerParConv1x1Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv1x1Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		case 3:
			if (ConvStride==1) ConvKerName = "KerParConv3x3Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv3x3Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		case 5:
			if (ConvStride==1) ConvKerName = "KerParConv5x5Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv5x5Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		default:
			ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			break;
	}
	if (DoPool) {
		switch (FSp) {
			case 2:
				if (PoolStride==2) PoolKerName = "KerParMaxPool2x2Stride2_fp";
				else GenTilingError("Unsupported Pool Stride\n");
				break;
			case 3:
				GenTilingError("Unsupported Pooling\n");
				break;
			default:
				GenTilingError("Unsupported Pooling\n");
				break;
		}
	}

	if (ConvDoPad) {
		PadTc = (FSc-1)/2; PadBc = FSc/2;
		Wc = (Width- FSc+((FSc-1)/2)+(FSc/2))/ConvStride + 1;
        	Hc = (Height-FSc+((FSc-1)/2)+(FSc/2))/ConvStride + 1;
	} else {
		PadTc = 0; PadBc = 0;
		Wc = (Width- FSc)/ConvStride + 1;
        	Hc = (Height-FSc)/ConvStride + 1;
	}
	if (DoPool) {
		if (ConvDoPad) {
			PadTp = (FSp-1)/2; PadBp = FSp/2;
			PadInp = (v4s){0,0,PadTp,PadBp};
			Wo = (Wc-FSp+((FSp-1)/2)+(FSp/2))/PoolStride + 1;
        		Ho = (Hc-FSp+((FSp-1)/2)+(FSp/2))/PoolStride + 1;
		} else {
			PadTp = 0; PadBp = 0;
			Wo = (Wc-FSp)/PoolStride + 1;
        		Ho = (Hc-FSp)/PoolStride + 1;
		}
	} else {
		PadTp = 0; PadBp = 0;
		Wo = Wc; Ho = Hc;
	}


        UserKernel(Name,
                // KernelDimensions(1, Width, Height, 1),
                // KernelDimensions(InFeat, Width, Height, OutFeat),
		KernelDimensionsAndUserSymbols(InFeat, Width, Height, OutFeat,
			KerDynamicSymbols(4,
				S_Dyn("Wo", Wo),
				S_Dyn("Ho", Ho),
				S_Dyn("Wc", Wc),
				S_Dyn("Hc", Hc)
			)
		),
                KernelIterationOrder(KER_DIM3, KER_TILE, KER_TILE1),
                TILE_HOR,
                CArgs(5,
                      TCArg("short int * __restrict__", "In"),
                      TCArg("short int * __restrict__", "Filter"),
					  TCArg("short int * __restrict__", "Out"),
					  TCArg("unsigned int",             "Norm"),
                      TCArg("short int * __restrict__", "Bias")
                     ),
                Calls(3,
			Call("KerParSetBias_fp", LOC_INNER_LOOP1_PROLOG,
                                Bindings(5, 
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					UserSymb("Wc"), // Imm(Wc),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					KerDim(K_OUTP), // Imm(OutFeat),
					K_Arg("Bias", KER_ARG_TILE)
				)
			),
                        Call(ConvKerName, LOC_INNER_LOOP1,
                                Bindings(17, 
					K_Arg("In", KER_ARG_TILE),
					KerDim(K_W), // Imm(Width),
					K_Arg("In", KER_ARG_TILE_H),
					KerDim(K_INP), // Imm(InFeat),
					K_Arg("Filter", KER_ARG_TILE_H),
					K_Arg("Filter", KER_ARG_TILE_BASE),
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					C_Arg("Norm"),
					K_Arg("In", KER_ARG_TILEINDEX),
					K_Arg("In", KER_ARG_NTILES),
					Imm(1),
					K_Arg("In", KER_ARG_PAD),
					K_Arg("In", KER_ARG_TILE_H0),
					K_Arg("In", KER_ARG_H),
					NeedConvDim?Imm(FSc):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStride):AT_IGNORE_ARG_BINDING
					)
			),
			ConvDoReLU?
			Call(ReLUKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(5,
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					UserSymb("Wc"), // Imm(Wc),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					KerDim(K_OUTP), // Imm(OutFeat),
					K_Arg("Out", KER_ARG_TILE)
				)
			):(DoPool?
			Call(PoolKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(7,
					K_Arg("ConvOut", KER_ARG_TILE),
					UserSymb("Wc"), // Imm(Wc),
					K_Arg("ConvOut", KER_ARG_TILE_H),
					KerDim(K_OUTP), // Imm(OutFeat),
					K_Arg("Out", KER_ARG_TILE),
					Imm((int)PadInp),
					Imm(PoolDoReLU)
				)
			):AT_NO_CALL)
                     ),
                KerArgs(5,
			ConvDoPad?
			KerArg2DP("In",      OBJ_IN_DB,            Width*InFeat,  Height,   Width, 0, 0, PadTc, PadBc, sizeof(short int),         Overlap, ConvDoPad?OBJ_CONSTRAINTS_PAD_REM:OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0):
			KerArg2D ("In",      OBJ_IN_DB,            Width*InFeat,  Height,   Width,                     sizeof(short int),         Overlap, ConvDoPad?OBJ_CONSTRAINTS_PAD_REM:OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
                        KerArg   ("Filter",  OBJ_IN_DB|O_TILE1,    InFeat,        OutFeat,                             FSc*FSc*sizeof(short int), 0,       0,                        				       8,        "Filter", 0),
                        KerArg   ("Bias",    OBJ_BUFFER_IN_NTILED, 1,             OutFeat,                             sizeof(short int),         0,       0,                        				       0,        "Bias", 0),
                        KerArg2D ("Out",     OBJ_OUT_DB,           Wo*OutFeat,    Ho,       Wo,                        sizeof(short int),         0,       0,                        				       0,        "Out", 0),
			DoPool?
			KerArg   ("ConvOut", OBJ_BUFFER_ONETILE,   Wc*OutFeat,    Hc,                                  sizeof(short int),         0,       0,                                                          0,         "", 0)
			:AT_NO_KER_ARG
                        )
        );
}

void MediumParOutFeatConvolutionPoolReLU_fp(
			char         *Name,

                        unsigned int InFeat,
                        unsigned int OutFeat,
                        unsigned int Width,
                        unsigned int Height,

                        unsigned int FSc,
			unsigned int ConvStride,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSp,
			unsigned int PoolStride,
			int          PoolDoPad,
			int          PoolDoReLU,

			int	     DoPool
			)

{
	if (DoPool==0) {
		FSp=1; PoolStride=1;
	}
	int NeedConvDim=0, NeedConvStride=0;
	unsigned int TileCons = ConvStride * PoolStride;
        int Overlap = FSc + ConvStride*(FSp-PoolStride-1);
	unsigned int Wo, Ho, Wc, Hc;
	int PadTc, PadBc, PadTp, PadBp;
	v4s PadInp = (v4s){0,0,0,0};
	v4s PadInc = (v4s){0,0,0,0};
	char *ConvKerName, *PoolKerName;
	char *ReLUKerName = "KerParReLU_fp";

	switch (FSc) {
		case 1:
			if (ConvStride==1) ConvKerName = "KerParConv1x1Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv1x1Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		case 3:
			if (ConvStride==1) ConvKerName = "KerParConv3x3Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv3x3Stride2_fp";
			else {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		case 5:
			if (ConvStride==1) ConvKerName = "KerParConv5x5Stride1_fp";
			else if (ConvStride == 2) ConvKerName = "KerParConv5x5Stride2_fp";
			else  {
				ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			}
			break;
		default:
			ConvKerName = "KerParConvNxNStrideS_fp"; NeedConvDim = 1; NeedConvStride = 1;
			break;
	}
	if (DoPool) {
		switch (FSp) {
			case 2:
				if (PoolStride==2) PoolKerName = "KerParMaxPool2x2Stride2_fp";
				else GenTilingError("Unsupported Pool Stride\n");
				break;
			case 3:
				GenTilingError("Unsupported Pooling\n");
				break;
			default:
				GenTilingError("Unsupported Pooling\n");
				break;
		}
	}

	if (ConvDoPad) {
		PadTc = (FSc-1)/2; PadBc = FSc/2;
		PadInc = (v4s){(FSc-1)/2,FSc/2,(FSc-1)/2, FSc/2};
		Wc = (Width- FSc+((FSc-1)/2)+(FSc/2))/ConvStride + 1;
        	Hc = (Height-FSc+((FSc-1)/2)+(FSc/2))/ConvStride + 1;
	} else {
		PadTc = 0; PadBc = 0;
		Wc = (Width- FSc)/ConvStride + 1;
        	Hc = (Height-FSc)/ConvStride + 1;
	}
	if (DoPool) {
		if (ConvDoPad) {
			PadTp = (FSp-1)/2; PadBp = FSp/2;
			PadInp = (v4s){0,0,PadTp,PadBp};
			Wo = (Wc-FSp+((FSp-1)/2)+(FSp/2))/PoolStride + 1;
        		Ho = (Hc-FSp+((FSp-1)/2)+(FSp/2))/PoolStride + 1;
		} else {
			PadTp = 0; PadBp = 0;
			Wo = (Wc-FSp)/PoolStride + 1;
        		Ho = (Hc-FSp)/PoolStride + 1;
		}
	} else {
		PadTp = 0; PadBp = 0;
		Wo = Wc; Ho = Hc;
	}

        UserKernel(Name,
		KernelDimensionsAndUserSymbols(InFeat, Width, Height, OutFeat,
			KerDynamicSymbols(4,
				S_Dyn("Wo", Wo),
				S_Dyn("Ho", Ho),
				S_Dyn("Wc", Wc),
				S_Dyn("Hc", Hc)
			)
		),
                KernelIterationOrder(KER_DIM3, KER_TILE, KER_TILE1),
                TILE_HOR,
                CArgs(5,
                      TCArg("short int * __restrict__", "In"),
                      TCArg("short int * __restrict__", "Filter"),
					  TCArg("short int * __restrict__", "Out"),
					  TCArg("unsigned int",             "Norm"),
                      TCArg("short int * __restrict__", "Bias")
                     ),
                Calls(3,
			Call("KerParSetBias_fp", LOC_INNER_LOOP1_PROLOG,
                                Bindings(5, 
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					UserSymb("Wo"), // Imm(Wo),
					UserSymb("Ho"), // Imm(Ho),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					K_Arg("Bias", KER_ARG_TILE)
				)
			),
                        Call(ConvKerName, LOC_INNER_LOOP1,
                                Bindings(17, 
					K_Arg("In", KER_ARG_TILE),
					KerDim(K_W), // Imm(Width),
					KerDim(K_H), // Imm(Height),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					K_Arg("In", KER_ARG_TILE_BASE),
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					C_Arg("Norm"),
					KerDim(K_INP), // Imm(InFeat),
					Imm(0),
					AT_IGNORE_ARG_BINDING,
					Imm((int)PadInc),
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					NeedConvDim?Imm(FSc):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStride):AT_IGNORE_ARG_BINDING
					)
			),

			ConvDoReLU?
			Call(ReLUKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(5,
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					UserSymb("Wc"), // Imm(Wc),
					UserSymb("Hc"), // Imm(Hc),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE)
				)
			):(DoPool?
			Call(PoolKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(7,
					K_Arg("ConvOut", KER_ARG_TILE),
					UserSymb("Wc"), // Imm(Wc),
					UserSymb("Hc"), // Imm(Hc),
					K_Arg("ConvOut", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					Imm((int)PadInp),
					Imm(PoolDoReLU)
				)
			):AT_NO_CALL)
                     ),
                KerArgs(5,
                        KerArg("In",      OBJ_IN_DB|O_TILE1,  Width*Height,  InFeat,  sizeof(short int),         0, 0, 0, "In", 0),
                        KerArg("Filter",  OBJ_IN_DB,          InFeat,        OutFeat, FSc*FSc*sizeof(short int), 0, 0, 8, "Filter", 0),
                        KerArg("Bias",    OBJ_BUFFER_IN,      1,             OutFeat, sizeof(short int),         0, 0, 0, "Bias", 0),
                        KerArg("Out",     OBJ_OUT_DB,         Wo*Ho,         OutFeat, sizeof(short int),         0, 0, 0, "Out", 0),
		DoPool? KerArg("ConvOut", OBJ_BUFFER_ONETILE, Wc*Hc,         OutFeat, sizeof(short int),         0, 0, 0, "", 0):AT_NO_KER_ARG
                        )
        );
}






// =============================================================================



/* Software Composite kernels: Convolution, ReLU then Max or Avg Pooling */
void CNN_TiledConvNxNStride2ReLU_SW_fp(	char *Name, 
						unsigned int FS, 
						unsigned int InPlane, 
						unsigned int OutPlane, 
						unsigned int Width, 
						unsigned int Height, 
						unsigned int PoolMax) {

	char *ConvKerName, *KerReLUPoolName;
	int Pad = 0;

	switch (FS) {
		case 3:
			ConvKerName = "KerConv3x3Stride2_fp";
			Pad = 1; break;
		case 5:
			ConvKerName = "KerConv5x5Stride2_fp"; 
			Pad = 2; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("TiledConvNxNStride2ReLUMaxPool2x2_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerConvNxNStride2_fp"; break;
	}
	switch (PoolMax) {
		case 0:	/* Relu, PoolAvg */
			KerReLUPoolName = "KerReLUAvgPool2x2_fp"; break;
		case 1:	/* Relu, PoolMax */
			KerReLUPoolName = "KerReLUMaxPool2x2_fp"; break;
		case 2:	/* No Relu, PoolAvg */
			KerReLUPoolName = "KerAvgPool2x2_fp"; break;
		case 3:	/* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool2x2_fp"; break;
		case 4: /* No Relu, PoolMax */
			KerReLUPoolName = "KerMaxPool3x3_fp"; break;
		case 5: /* Relu, No PoolMax */
			KerReLUPoolName = "KerReLU_fp"; break;
	}
	// UserKernel(Name,
	// 	KernelDimensions(InPlane, Width, Height, OutPlane),
	// 	KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
	// 	TILE_HOR,
	// 	CArgs(5,
	// 	      TCArg("short int * __restrict__", "In"),
	// 	      TCArg("short int * __restrict__", "Filter"),
	// 	      TCArg("short int * __restrict__", "Out"),
	// 	      TCArg("unsigned int", 			"Norm"),
	// 	      TCArg("short int * __restrict__",	"Bias")
	// 	    ),
	// 	Calls(6,
	// 		Call("KerExpandTileInitH_fp", LOC_PROLOG,
	// 			Bindings(4, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W),  K_Arg("InBuff", KER_ARG_TILE_H0), Imm(Pad))),

	// 		Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
	// 			Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),

	// 		Call("KerPadFirstTileH_fp", LOC_INNER_LOOP_PROLOG,
	// 			Bindings(3, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W), Imm(Pad))),

	// 		Call("KerExpandTileH_fp", LOC_INNER_LOOP,
	// 			Bindings(7, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("InBuff", KER_ARG_TILE), Imm(Pad), K_Arg("In", KER_ARG_TILEINDEX), K_Arg("In", KER_ARG_NTILES))),

	// 		Call(ConvKerName, LOC_INNER_LOOP,
 //                Bindings(6, K_Arg("InBuff", KER_ARG_TILE), K_Arg("InBuff", KER_ARG_TILE_W), K_Arg("InBuff", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE), C_Arg("Norm"))),

	// 		Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
	// 			Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H),K_Arg("Out", KER_ARG_TILE)))
	// 	    ),
	// 	KerArgs(5,
	// 	    KerArgP("In",        OBJ_IN_DB_3D,      	Width,   	 Height,     	0, 	 Pad, 	sizeof(short int), FS-1,   	0, 0, "In",     0),
 //            KerArg("Filter",     OBJ_IN_DB_NTILED_4D,   FS,      	 FS,                 		sizeof(short int), 0,      	0, 0, "Filter", 0),
 //            KerArgP("InBuff",    OBJ_BUFFER_ONETILE,    Width,	 	 Height,   		Pad, Pad, 	sizeof(short int), FS-1,   	0, 0, "",       0),
	// 		KerArg("BiasOut", 	 OBJ_BUFFER_ONETILE,	(Width+1)/2, (Height+1)/2,	 			sizeof(short int), 0, 		0, 0, "", 		0),
	// 		KerArg("Out",     	 OBJ_OUT_DB_3D,   		(Width+1)/2, (Height+1)/2,	 			sizeof(short int), 0, 		0, 0, "Out", 	0)
	// 		)
	// );


/********************************** NON PADDED ****************************/
	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 			"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		    ),
		Calls(3,
			Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE), C_Arg("Norm"))),
			Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H),K_Arg("Out", KER_ARG_TILE)))
		    ),
		KerArgs(4,
			KerArg("In",      OBJ_IN_DB_3D,     	Width,	    	Height, 			sizeof(short int), FS-1,	0, 0, "In", 	0),
			KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS,	   			FS, 				sizeof(short int), 0, 		0, 0, "Filter", 0),
			KerArg("BiasOut", OBJ_BUFFER_ONETILE,	(Width-FS+1)/2, (Height-FS+1)/2,	sizeof(short int), 0, 		0, 0, "", 		0),
			KerArg("Out",     OBJ_OUT_DB_3D,   		(Width-FS+1)/2, (Height-FS+1)/2,	sizeof(short int), 0, 		0, 0, "Out", 	0)
			)
	);
}


/* Software Composite kernels: Convolution, ReLU. Multi-channel fashion. */
void CNN_TiledConvNxNStride2ReLUMulti_SW_fp(	char *Name, 
						unsigned int FS, 
						unsigned int InPlane, 
						unsigned int OutPlane, 
						unsigned int Width, 
						unsigned int Height, 
						unsigned int PoolMax) {

	char *ConvKerName, *KerReLUPoolName;

	int strideW = (Width-FS+1)/2;
	if((Width-FS+1)&0x1) strideW++;
	int strideH = (Height-FS+1)/2;
	if((Height-FS+1)&0x1) strideH++;

	ConvKerName = "KerConv3x3Stride2Multi_fp";
	KerReLUPoolName = "KerReLU_fp";
	
	UserKernel(Name,
		KernelDimensions(1, Width*InPlane, Height, OutPlane),
		KernelIterationOrder(KER_DIM3, KER_OUT_PLANE, KER_TILE),
		TILE_HOR,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 			"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		    ),
		Calls(3,
			Call("KerSetInBias", LOC_INNER_LOOP,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(8, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE), C_Arg("Norm"), Imm(InPlane), K_Arg("Reduct", KER_ARG_TILE))),
			Call(KerReLUPoolName, LOC_INNER_LOOP,
				Bindings(4, K_Arg("BiasOut", KER_ARG_TILE), K_Arg("BiasOut", KER_ARG_TILE_W), K_Arg("BiasOut", KER_ARG_TILE_H),K_Arg("Out", KER_ARG_TILE)))
		    ),
		KerArgs(5,
			KerArg("In",      OBJ_BUFFER_IN_3D,     Width*InPlane,		Height, 	sizeof(short int), 	0,	0, 0, "In", 	0),
			KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS*InPlane,	   		FS, 		sizeof(short int), 	0, 	0, 0, "Filter", 0),
			KerArg("BiasOut", OBJ_BUFFER_ONETILE,	strideW,			strideH,	sizeof(short int), 	0, 	0, 0, "", 		0),
			KerArg("Reduct",  OBJ_BUFFER_ONETILE,	strideW*strideH, 	8,			sizeof(int), 		0, 	0, 0, "", 		0),
			KerArg("Out",     OBJ_BUFFER_OUT_3D,   	strideW, 			strideH,	sizeof(short int), 	0, 	0, 0, "Out", 	0)
			)
	);

}

/* Software Composite kernels: ReLU then Convolution */
void CNN_ReLUTiledConvNxNStride2_SW_fp(	char *Name, 
										unsigned int FS, 
										unsigned int InPlane, 
										unsigned int OutPlane, 
										unsigned int Width, 
										unsigned int Height) {

	char *ConvKerName;

	switch (FS) {
		case 3:
			ConvKerName = "KerConv3x3Stride2_fp"; break;
		case 5:
			ConvKerName = "KerConv5x5Stride2_fp"; break;
		default:
			if ((FS&1 == 0) || (FS == 0)) GenTilingError("ReLUTiledConvNxNStride2_SW_fp: Convolution dimension has to be > 0 and odd");
			ConvKerName = "KerConvNxNStride2_fp"; break;
	}

	// UserKernel(Name,
	// 	KernelDimensions(InPlane, Width, Height, OutPlane),
	// 	KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
	// 	TILE_HOR,
	// 	CArgs(5,
	// 	      TCArg("short int * __restrict__", "In"),
	// 	      TCArg("short int * __restrict__", "Filter"),
	// 	      TCArg("short int * __restrict__", "Out"),
	// 	      TCArg("unsigned int", 			"Norm"),
	// 	      TCArg("short int * __restrict__",	"Bias")
	// 	    ),
	// 	Calls(3,
	// 		Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
	// 			Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
	// 		Call("KerReLU_fp", LOC_IN_PLANE_PROLOG,
	// 			Bindings(4, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H),K_Arg("ReLuOut", KER_ARG_TILE))),
	// 		Call(ConvKerName, LOC_INNER_LOOP,
	// 			Bindings(6, K_Arg("ReLuOut", KER_ARG_TILE), K_Arg("ReLuOut", KER_ARG_TILE_W), K_Arg("ReLuOut", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), C_Arg("Norm")))
	// 	    ),
	// 	KerArgs(4,
	// 		KerArg("In",      OBJ_IN_DB_3D,     	Width,	    	Height, 			sizeof(short int), 0,	0, 0, "In", 	0),
	// 		KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS,	   			FS, 				sizeof(short int), 0, 	0, 0, "Filter", 0),
	// 		KerArg("ReLuOut", OBJ_BUFFER_ONETILE,	Width,			Height,				sizeof(short int), 0,	0, 0, "", 		0),
	// 		KerArg("Out",     OBJ_OUT_DB_3D,   		(Width-FS+1)/2, (Height-FS+1)/2,	sizeof(short int), 0, 	0, 0, "Out", 	0)
	// 		)
	// );

    UserKernel(Name,
        KernelDimensions(InPlane, Width, Height, OutPlane),
        KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
        TILE_HOR,
        CArgs(5,
              TCArg("short int * __restrict__", "In"),
              TCArg("short int * __restrict__", "Filter"),
              TCArg("short int * __restrict__", "Out"),
              TCArg("unsigned int",             "Norm"),
              TCArg("short int * __restrict__", "Bias")
            ),
        Calls(2,
                Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
                        Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
                Call("KerReLUConv3x3Stride2_fp", LOC_INNER_LOOP,
                        Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE),
                        K_Arg("Out", KER_ARG_TILE), C_Arg("Norm")))
            ),
        KerArgs(3,
                KerArg("In",      OBJ_IN_DB_3D,         Width,          Height,			sizeof(short int), 0,   OBJ_CONSTRAINTS_DROP_REM, 	0, "In",		0),
                KerArg("Filter",  OBJ_IN_DB_NTILED_4D,  FS,             FS,         	sizeof(short int), 0,   0, 							0, "Filter", 	0),
                KerArg("Out",     OBJ_OUT_DB_3D,   		(Width-FS+1)/2, (Height-FS+1)/2,sizeof(short int), 0,   0, 							0, "Out",    	0)
                )
        );
}


/* Software Composite kernels: ReLU then Convolution */
void CNN_ReLUTiledConvNxN_SW_fp(char *Name, 
								unsigned int FS, 
								unsigned int InPlane, 
								unsigned int OutPlane, 
								unsigned int Width, 
								unsigned int Height) {
	// char *ConvKerName;

	// switch (FS) {
	// 	case 3:
	// 		ConvKerName = "KerConv3x3_fp"; break;
	// 	case 5:
	// 		ConvKerName = "KerConv5x5_fp"; break;
	// 	default:
	// 		if ((FS&1 == 0) || (FS == 0)) GenTilingError("ReLUTiledConvNxN_SW_fp: Convolution dimension has to be > 0 and odd");
	// 		ConvKerName = "KerConvNxN_fp"; break;
	// }

	// UserKernel(Name,
	// 	KernelDimensions(InPlane, Width, Height, OutPlane),
	// 	KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
	// 	TILE_HOR,
	// 	CArgs(5,
	// 	      TCArg("short int * __restrict__", "In"),
	// 	      TCArg("short int * __restrict__", "Filter"),
	// 	      TCArg("short int * __restrict__", "Out"),
	// 	      TCArg("unsigned int", 			"Norm"),
	// 	      TCArg("short int * __restrict__",	"Bias")
	// 	    ),
	// 	Calls(3,
	// 		Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
	// 			Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
	// 		Call("KerReLU_fp", LOC_INNER_LOOP,
	// 			Bindings(4, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H),K_Arg("ReLuOut", KER_ARG_TILE))),
	// 		Call(ConvKerName, LOC_INNER_LOOP,
	// 			Bindings(6, K_Arg("ReLuOut", KER_ARG_TILE), K_Arg("ReLuOut", KER_ARG_TILE_W), K_Arg("ReLuOut", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), C_Arg("Norm")))
	// 	    ),
	// 	KerArgs(4,
	// 		KerArg("In",      OBJ_IN_DB_3D,     	Width,	    Height, 		sizeof(short int), 0,	0, 0, "In", 	0),
	// 		KerArg("Filter",  OBJ_IN_DB_NTILED_4D, 	FS,	   		FS, 			sizeof(short int), 0, 	0, 0, "Filter", 0),
	// 		KerArg("ReLuOut", OBJ_BUFFER_ONETILE,	Width,		Height,			sizeof(short int), 0, 	0, 0, "", 		0),
	// 		KerArg("Out",     OBJ_OUT_DB_3D,   		Width-FS+1,	Height-FS+1,	sizeof(short int), 0, 	0, 0, "Out", 	0)
	// 		)
	// );

    UserKernel(Name,
        KernelDimensions(InPlane, Width, Height, OutPlane),
        KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
        TILE_HOR,
        CArgs(5,
              TCArg("short int * __restrict__", "In"),
              TCArg("short int * __restrict__", "Filter"),
              TCArg("short int * __restrict__", "Out"),
              TCArg("unsigned int",             "Norm"),
              TCArg("short int * __restrict__", "Bias")
            ),
        Calls(2,
                Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
                        Bindings(4, K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_W), K_Arg("Out", KER_ARG_TILE_H), C_ArgIndex("Bias", KER_OUT_PLANE, 1))),
                Call("KerReLUConv3x3_fp", LOC_INNER_LOOP,
                        Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), K_Arg("Filter", KER_ARG_TILE),
                        K_Arg("Out", KER_ARG_TILE), C_Arg("Norm")))
            ),
        KerArgs(3,
                KerArg("In",      OBJ_IN_DB_3D,         Width,          Height,			sizeof(short int), 0,   OBJ_CONSTRAINTS_DROP_REM, 	0, "In",		0),
                KerArg("Filter",  OBJ_IN_DB_NTILED_4D,  FS,             FS,         	sizeof(short int), 0,   0, 							0, "Filter", 	0),
                KerArg("Out",     OBJ_OUT_DB_3D,   		(Width-FS+1), (Height-FS+1),	sizeof(short int), 0,   0, 							0, "Out",    	0)
                )
        );
}


/* Software Composite kernels: Feature Maps Add */
void CNN_MatrixAdd_SW_fp(	char *Name, 
							unsigned int InPlane, 
							unsigned int OutPlane, 
							unsigned int Width, 
							unsigned int Height) {

	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM3, KER_IN_OUT_PLANE, KER_TILE),
		TILE_HOR,
		CArgs(2,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Out")
		 ),
		Calls(1,
			Call("KerAddFM_fp", LOC_INNER_LOOP,
				Bindings(4, K_Arg("In", KER_ARG_TILE),  
							K_Arg("Out", KER_ARG_TILE), 
							K_Arg("In", KER_ARG_TILE_W), 
							K_Arg("In", KER_ARG_TILE_H)))
		    ),
		KerArgs(2,
			KerArg("In",	OBJ_IN_DB_3D,		Width,	Height, sizeof(short int), 0,	0, 0, "In", 0),
			KerArg("Out",	OBJ_IN_OUT_DB_3D,	Width,	Height,	sizeof(short int), 0, 	0, 0, "Out", 0)
			)
	);
}

/* Software Composite kernels: Feature Maps Add then Activation */
void CNN_MatrixAddReLu_SW_fp(	char *Name, 
							unsigned int InPlane, 
							unsigned int OutPlane, 
							unsigned int Width, 
							unsigned int Height) {

	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM3, KER_IN_OUT_PLANE, KER_TILE),
		TILE_HOR,
		CArgs(2,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Out")
		 ),
		Calls(1,
			Call("KerAddFMReLu_fp", LOC_INNER_LOOP,
				Bindings(4, K_Arg("In", KER_ARG_TILE),  
							K_Arg("Out", KER_ARG_TILE), 
							K_Arg("In", KER_ARG_TILE_W), 
							K_Arg("In", KER_ARG_TILE_H)))
		    ),
		KerArgs(2,
			KerArg("In",	OBJ_IN_DB_3D,		Width,	Height, sizeof(short int), 0,	0, 0, "In",	0),
			KerArg("Out",	OBJ_IN_OUT_DB_3D,	Width,	Height,	sizeof(short int), 0, 	0, 0, "Out", 0)
			)
	);
}

/* Software Composite kernels: ReLu Activation */
void CNN_ReLu_SW_fp(	char *Name, 
						unsigned int InPlane, 
						unsigned int OutPlane, 
						unsigned int Width, 
						unsigned int Height) {

	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM3, KER_IN_OUT_PLANE, KER_TILE),
		TILE_HOR,
		CArgs(2,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Out")
		 ),
		Calls(1,
			Call("KerReLU_fp", LOC_INNER_LOOP,
				Bindings(4, K_Arg("In", KER_ARG_TILE),  
							K_Arg("In", KER_ARG_TILE_W), 
							K_Arg("In", KER_ARG_TILE_H),
							K_Arg("Out", KER_ARG_TILE)))
		    ),
		KerArgs(2,
			KerArg("In",	OBJ_IN_DB_3D,		Width,	Height, sizeof(short int), 0,	0, 0, "In", 0),
			KerArg("Out",	OBJ_IN_OUT_DB_3D,	Width,	Height,	sizeof(short int), 0, 	0, 0, "Out", 0)
			)
	);
}

/* HWCE enabled convolutions */

/* Pure convolutions, no bias setting and no accumulation */
void CNN_TiledPlainConvNxN_HWCE_fp(char *Name, unsigned int FS, unsigned int Width, unsigned int Height)

{
	char *ConvKerName;
	int Fw, Fh;	/* Filter dimensions, Since FS*FS is odd and HwCE supports only 4 byte aligned accesses Fw * Fh = FS * FS + 1 */
	int ConvMode, Mode3x3 = 0;

	switch (FS) {
		case 3:
			ConvKerName = "HWCE_ProcessOneTile3x3_MultiOut"; Fw = 5; Fh = 2; ConvMode = 1; Mode3x3 = 1; break;
		case 5:
			ConvKerName = "HWCE_ProcessOneTile5x5"; Fw = 13; Fh = 2; ConvMode = 0; break;
		case 7:
			ConvKerName = "HWCE_ProcessOneTile7x7"; Fw = 25; Fh = 2; ConvMode = 2; break;
		default:
			GenTilingError("TiledPlainConvNxN_HWCE: Only 3x3, 5x5 and 7x7 are supported for HWCE enabled configurations");
	}
	UserKernel(Name,
		KernelDimensions(1, Width, Height, 1),
		KernelIterationOrder(KER_DIM2, KER_TILE),
		TILE_VER,
		CArgs(4,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 		"Norm")
		     ),
		Calls(4,
			Call("HWCE_Enable", LOC_PROLOG, Bindings(0)),
			Call("HWCE_GenericInit", LOC_PROLOG, Bindings(3, Imm(ConvMode), Imm(0), C_Arg("Norm"))),
		Mode3x3?
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(9, K_Arg("In", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), Imm(0), Imm(0),
					    K_Arg("Filter", KER_ARG_TILE), Imm(0), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), Imm(0x7))):
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), K_Arg("Filter", KER_ARG_TILE), Imm(0),
					    K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H))),
			Call("HWCE_Disable", LOC_EPILOG, Bindings(0))
		     ),
		KerArgs(3,
			KerArg("In",   	 OBJ_IN_DB,            Width,	   Height, 	sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_BUFFER_IN_NTILED, Fw,	   Fh, 		sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out",    OBJ_OUT_DB,	       Width-FS+1, Height-FS+1,	sizeof(short int), 0,  0, 0,  "Out", 0)
		       )
	);
}

/*  Convolution layer, bias setting and accumulation */
void CNN_TiledConvNxN_HWCE_fp(char *Name, unsigned int FS, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height)

{
	char *ConvKerName;
	int Fw, Fh;	/* Filter dimensions, Since FS*FS is odd and HwCE supports only 4 byte aligned accesses Fw * Fh = FS * FS + 1 */
	int ConvMode, Mode3x3 = 0;

	switch (FS) {
		case 3:
			ConvKerName = "HWCE_ProcessOneTile3x3_MultiOut"; Fw = 5; Fh = 2; ConvMode = 1; Mode3x3 = 1; break;
		case 5:
			ConvKerName = "HWCE_ProcessOneTile5x5"; Fw = 13; Fh = 2; ConvMode = 0; break;
		case 7:
			ConvKerName = "HWCE_ProcessOneTile7x7"; Fw = 25; Fh = 2; ConvMode = 2; break;
		default:
			GenTilingError("TiledConvNxN_HWCE: Only 3x3, 5x5 and 7x7 are supported for HWCE enabled configurations");
	}
	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_VER,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Bias"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 		"Norm")
		     ),
		Calls(6,
			Call("HWCE_Enable", LOC_PROLOG, Bindings(0)),
			Call("HWCE_GenericInit", LOC_PROLOG, Bindings(3, Imm(ConvMode), Imm(0), C_Arg("Norm"))),
			Call("HwCE_SetYinMode", LOC_IN_PLANE_PROLOG, Bindings(1, Imm(1))),
		Mode3x3?
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(9, K_Arg("In", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), Imm(0), Imm(0),
					    K_Arg("Filter", KER_ARG_TILE), C_ArgIndex("Bias", KER_OUT_PLANE, 1),
					    K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), Imm(0x7))):
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), K_Arg("Filter", KER_ARG_TILE),
					    C_ArgIndex("Bias", KER_OUT_PLANE, 1), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H))),
			Call("HwCE_SetYinMode", LOC_INNER_LOOP, Bindings(1, Imm(0))),
			Call("HWCE_Disable", LOC_EPILOG, Bindings(0))
		     ),
		KerArgs(3,
			KerArg("In",   	 OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, Fw,	  Fh, 	       sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out",    OBJ_OUT_DB_3D,	      Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out", 0)
		       )
	);
}

/*  Convolution layer, 3x3 convolution multiple output mode, bias setting and accumulation */
void CNN_TiledConv3x3_HWCE_MultiOut_fp(char *Name, unsigned int Nout, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height)

{
	char *ConvKerName, *SetBiasKerName;
	int FS = 3;
	int Mode, ConvMode = 1;

	ConvKerName = "HWCE_ProcessOneTile3x3_MultiOut";
	switch (Nout) {
		case 1: SetBiasKerName = "KerSetInBias"; Mode = 0x7; break;
		case 2: SetBiasKerName = "KerSetInBias2"; Mode = 0x3; break;
		case 3: SetBiasKerName = "KerSetInBias3"; Mode = 0x1; break;
		default:
			GenTilingError("TiledConv3x3MultiOut_HWCE: Only 1, 2 or 3 output mode supported for HWCE 3x3 enabled configurations");
	}

	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_VER,
	(Nout == 3)?
		CArgs(9,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out0"),
	      	      TCArg("short int * __restrict__", "Out1"),
	      	      TCArg("short int * __restrict__", "Out2"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__", "Bias0"),
		      TCArg("short int * __restrict__", "Bias1"),
		      TCArg("short int * __restrict__", "Bias2")
		     ):
	((Nout == 2)?
		CArgs(7,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out0"),
	      	      TCArg("short int * __restrict__", "Out1"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__", "Bias0"),
		      TCArg("short int * __restrict__", "Bias1")
		     ):
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out0"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__", "Bias0")
		     )),
		Calls(6,
			Call("HWCE_Enable", LOC_PROLOG, Bindings(0)),
			Call("HWCE_GenericInit", LOC_PROLOG, Bindings(3, Imm(ConvMode), Imm(0), C_Arg("Norm"))),
			Call("HwCE_SetYinMode", LOC_PROLOG, Bindings(1, Imm(0))),
			Call(SetBiasKerName, LOC_IN_PLANE_PROLOG,
				(Nout == 3)?
				Bindings(8, K_Arg("Out0", KER_ARG_TILE), K_Arg("Out1", KER_ARG_TILE), K_Arg("Out2", KER_ARG_TILE),
					    K_Arg("Out0", KER_ARG_TILE_W), K_Arg("Out0", KER_ARG_TILE_H),
					    C_ArgIndex("Bias0", KER_OUT_PLANE, 3), C_ArgIndex("Bias1", KER_OUT_PLANE, 3), C_ArgIndex("Bias2", KER_OUT_PLANE, 3)):
				((Nout == 2)?
				Bindings(6, K_Arg("Out0", KER_ARG_TILE), K_Arg("Out1", KER_ARG_TILE),
					    K_Arg("Out0", KER_ARG_TILE_W), K_Arg("Out0", KER_ARG_TILE_H),
					    C_ArgIndex("Bias0", KER_OUT_PLANE, 2), C_ArgIndex("Bias1", KER_OUT_PLANE, 2)):
				Bindings(4, K_Arg("Out0", KER_ARG_TILE), K_Arg("Out0", KER_ARG_TILE_W), K_Arg("Out0", KER_ARG_TILE_H),
					    C_ArgIndex("Bias0", KER_OUT_PLANE, 1)))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(9, K_Arg("In", KER_ARG_TILE),
					    K_Arg("Out0", KER_ARG_TILE), (Nout >=2)?K_Arg("Out1", KER_ARG_TILE):Imm(0), (Nout==3)?K_Arg("Out2", KER_ARG_TILE):Imm(0),
					    K_Arg("Filter", KER_ARG_TILE), Imm(0), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), Imm(Mode))),
			Call("HWCE_Disable", LOC_EPILOG, Bindings(0))
		     ),
		(Nout == 3)?
		KerArgs(5,
			KerArg("In",   	 OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, 7,	  4, 	       sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out0",   OBJ_OUT_DB_3D,	      Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out0", 3),
			KerArg("Out1",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out1", 3),
			KerArg("Out2",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out2", 3)
		       ):
		((Nout == 2)?
		KerArgs(4,
			KerArg("In",   	 OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, 3*2,	  3, 	       sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out0",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out0", 2),
			KerArg("Out1",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out1", 2)
		       ):
		KerArgs(3,
			KerArg("In",   	 OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, 5,	  2, 	       sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out0",   OBJ_OUT_DB_3D,	      Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out0", 0)
		       ))
	);
}


/* HWCE enabled composite kernels: NxN convolutions, ReLU and Max or Average Pooling 2x2 -> 1 */
void CNN_TiledConvNxNReLUPool2x2_HWCE_fp(char *Name, unsigned int FS, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height,
				     unsigned int PoolMax)

{
	char *ConvKerName, *KerReLUPoolName;
	int Fw, Fh;	/* Filter dimensions, Since FS*FS is odd and HwCE supports only 4 byte aligned accesses Fw * Fh = FS * FS + 1 */
	int ConvMode, Mode3x3 = 0;

	switch (FS) {
		case 3:
			ConvKerName = "HWCE_ProcessOneTile3x3_MultiOut"; Fw = 5; Fh = 2; ConvMode = 1; Mode3x3 = 1; break;
		case 5:
			ConvKerName = "HWCE_ProcessOneTile5x5"; Fw = 13; Fh = 2; ConvMode = 0; break;
		case 7:
			ConvKerName = "HWCE_ProcessOneTile7x7"; Fw = 25; Fh = 2; ConvMode = 2; break;
		default:
			GenTilingError("TiledConvNxN_HWCE: Only 3x3, 5x5 and 7x7 are supported for HWCE enabled configurations");
	}
	switch (PoolMax) {
		case 0:	/* Relu, PoolMax */
			KerReLUPoolName = "KerReLUAvgPool2x2_fp"; break;
		case 1:	/* Relu, PoolAvg */
			KerReLUPoolName = "KerReLUMaxPool2x2_fp"; break;
		case 2:	/* No Relu, PoolMax */
			KerReLUPoolName = "KerAvgPool2x2_fp"; break;
		case 3:	/* No Relu, PoolAvg */
			KerReLUPoolName = "KerMaxPool2x2_fp"; break;
	}
	// UserKernel("Conv5x5ReLUMaxPool2x2_HWCE",
	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_VER,
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__",	"Bias")
		     ),
		Calls(7,
			Call("HWCE_Enable", LOC_PROLOG, Bindings(0)),
			Call("HWCE_GenericInit", LOC_PROLOG, Bindings(3, Imm(0), Imm(0), C_Arg("Norm"))),
			Call("HwCE_SetYinMode", LOC_IN_PLANE_PROLOG, Bindings(1, Imm(1))),
			Mode3x3?
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(9, K_Arg("In", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE), Imm(0), Imm(0),
					    K_Arg("Filter", KER_ARG_TILE), C_ArgIndex("Bias", KER_OUT_PLANE, 1),
					    K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), Imm(0x7))):
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(6, K_Arg("In", KER_ARG_TILE), K_Arg("SetBiasOut", KER_ARG_TILE), K_Arg("Filter", KER_ARG_TILE),
					    C_ArgIndex("Bias", KER_OUT_PLANE, 1), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H))),
			Call("HwCE_SetYinMode", LOC_INNER_LOOP, Bindings(1, Imm(0))),
			Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
				Bindings(4, K_Arg("SetBiasOut", KER_ARG_TILE), K_Arg("SetBiasOut", KER_ARG_TILE_W), K_Arg("SetBiasOut", KER_ARG_TILE_H),
					    K_Arg("Out", KER_ARG_TILE))),
			Call("HWCE_Disable", LOC_EPILOG, Bindings(0))
		     ),
		KerArgs(4,
			KerArg("In",   	      OBJ_IN_DB_3D,        Width,	   Height, 	    sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D, Fw,	    	   Fh, 	     	    sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("SetBiasOut",  OBJ_BUFFER_ONETILE,  (Width-FS+1),   Height-FS+1,     sizeof(short int), 0,  0, 0,  "", 0),
			KerArg("Out",         OBJ_OUT_DB_3D, 	   (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,  0, 0,  "Out", 0)
		       )
	);
}

/* HWCE enabled composite kernels: 3x3 convolutions, ReLU and Max Pooling 2x2 -> 1, multiple out mode */
void CNN_TiledConv3x3ReLUPool2x2_HWCE_MultiOut_fp(char *Name, unsigned int Nout,
					      unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height,
					      unsigned int PoolMax)

{
	char *ConvKerName, *SetBiasKerName, *ReLUPoolKerName;
	int FS = 3;
	int Mode, ConvMode = 1;

	ConvKerName = "HWCE_ProcessOneTile3x3_MultiOut";
	switch (Nout) {
		case 1:
			if (PoolMax) ReLUPoolKerName = "KerReLUMaxPool2x2_fp"; else ReLUPoolKerName = "KerReLUAvgPool2x2_fp";
			SetBiasKerName = "KerSetInBias"; Mode = 0x7; break;
		case 2:
			if (PoolMax) ReLUPoolKerName = "KerReLUMaxPool2x2_2_fp"; else ReLUPoolKerName = "KerReLUAvgPool2x2_2_fp";
			SetBiasKerName = "KerSetInBias2"; Mode = 0x3; break;
		case 3:
			if (PoolMax) ReLUPoolKerName = "KerReLUMaxPool2x2_3_fp"; else ReLUPoolKerName = "KerReLUAvgPool2x2_3_fp";
			SetBiasKerName = "KerSetInBias3"; Mode = 0x1; break;
		default:
			GenTilingError("TiledConv3x3MultiOut_HWCE: Only 1, 2 or 3 output mode supported for HWCE 3x3 enabled configurations");
	}

	UserKernel(Name,
		KernelDimensions(InPlane, Width, Height, OutPlane),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
		TILE_VER,
	(Nout == 3)?
		CArgs(9,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out0"),
	      	      TCArg("short int * __restrict__", "Out1"),
	      	      TCArg("short int * __restrict__", "Out2"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__", "Bias0"),
		      TCArg("short int * __restrict__", "Bias1"),
		      TCArg("short int * __restrict__", "Bias2")
		     ):
	((Nout == 2)?
		CArgs(7,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out0"),
	      	      TCArg("short int * __restrict__", "Out1"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__", "Bias0"),
		      TCArg("short int * __restrict__", "Bias1")
		     ):
		CArgs(5,
		      TCArg("short int * __restrict__", "In"),
		      TCArg("short int * __restrict__", "Filter"),
		      TCArg("short int * __restrict__", "Out0"),
		      TCArg("unsigned int", 		"Norm"),
		      TCArg("short int * __restrict__", "Bias0")
		     )),
		Calls(6,
			Call("HWCE_Enable", LOC_PROLOG, Bindings(0)),
			Call("HWCE_GenericInit", LOC_PROLOG, Bindings(3, Imm(ConvMode), Imm(0), C_Arg("Norm"))),
			Call("HwCE_SetYinMode", LOC_PROLOG, Bindings(1, Imm(0))),
			Call(SetBiasKerName, LOC_IN_PLANE_PROLOG,
				(Nout == 3)?
				Bindings(8, K_Arg("SetBiasOut0", KER_ARG_TILE), K_Arg("SetBiasOut1", KER_ARG_TILE), K_Arg("SetBiasOut2", KER_ARG_TILE),
					    K_Arg("SetBiasOut0", KER_ARG_TILE_W), K_Arg("SetBiasOut0", KER_ARG_TILE_H),
					    C_ArgIndex("Bias0", KER_OUT_PLANE, 3), C_ArgIndex("Bias1", KER_OUT_PLANE, 3), C_ArgIndex("Bias2", KER_OUT_PLANE, 3)):
				((Nout == 2)?
				Bindings(6, K_Arg("SetBiasOut0", KER_ARG_TILE), K_Arg("SetBiasOut1", KER_ARG_TILE),
					    K_Arg("SetBiasOut0", KER_ARG_TILE_W), K_Arg("SetBiasOut0", KER_ARG_TILE_H),
					    C_ArgIndex("Bias0", KER_OUT_PLANE, 2), C_ArgIndex("Bias1", KER_OUT_PLANE, 2)):
				Bindings(4, K_Arg("SetBiasOut0", KER_ARG_TILE), K_Arg("SetBiasOut0", KER_ARG_TILE_W), K_Arg("SetBiasOut0", KER_ARG_TILE_H),
					    C_ArgIndex("Bias0", KER_OUT_PLANE, 1)))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(9, K_Arg("In", KER_ARG_TILE),
					    K_Arg("SetBiasOut0", KER_ARG_TILE),
					    (Nout >= 2)?K_Arg("SetBiasOut1", KER_ARG_TILE):Imm(0),
					    (Nout == 3)?K_Arg("SetBiasOut2", KER_ARG_TILE):Imm(0),
					    K_Arg("Filter", KER_ARG_TILE), Imm(0), K_Arg("In", KER_ARG_TILE_W), K_Arg("In", KER_ARG_TILE_H), Imm(Mode))),

			Call(ReLUPoolKerName, LOC_IN_PLANE_EPILOG,
				(Nout == 3)?
				Bindings(8, K_Arg("SetBiasOut0", KER_ARG_TILE), K_Arg("SetBiasOut1", KER_ARG_TILE), K_Arg("SetBiasOut2", KER_ARG_TILE),
					    K_Arg("SetBiasOut0", KER_ARG_TILE_W), K_Arg("SetBiasOut0", KER_ARG_TILE_H),
					    K_Arg("Out0", KER_ARG_TILE), K_Arg("Out1", KER_ARG_TILE), K_Arg("Out2", KER_ARG_TILE)):
				((Nout == 2)?
				Bindings(6, K_Arg("SetBiasOut0", KER_ARG_TILE), K_Arg("SetBiasOut1", KER_ARG_TILE),
					    K_Arg("SetBiasOut0", KER_ARG_TILE_W), K_Arg("SetBiasOut0", KER_ARG_TILE_H),
					    K_Arg("Out0", KER_ARG_TILE), K_Arg("Out1", KER_ARG_TILE)):
				Bindings(4, K_Arg("SetBiasOut0", KER_ARG_TILE), K_Arg("SetBiasOut0", KER_ARG_TILE_W), K_Arg("SetBiasOut0", KER_ARG_TILE_H),
					    K_Arg("Out", KER_ARG_TILE))
				)),

			Call("HWCE_Disable", LOC_EPILOG, Bindings(0))
		     ),
		(Nout == 3)?
		KerArgs(8,
			KerArg("In",   	      OBJ_IN_DB_3D,     	Width,	    	Height, 	 		sizeof(short int), FS-1, 	OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D, 	7,	    		4, 	     	 		sizeof(short int), 0,  		0, 0,  "Filter", 0),
			KerArg("SetBiasOut0", OBJ_BUFFER_ONETILE,	(Width-FS+1),	(Height-FS+1),   	sizeof(short int), 0,  		0, 0,  "", 0),
			KerArg("SetBiasOut1", OBJ_BUFFER_ONETILE,	(Width-FS+1),	(Height-FS+1),   	sizeof(short int), 0,  		0, 0,  "", 0),
			KerArg("SetBiasOut2", OBJ_BUFFER_ONETILE,	(Width-FS+1),	(Height-FS+1),   	sizeof(short int), 0,  		0, 0,  "", 0),
			KerArg("Out0",        OBJ_OUT_DB_3D, 		(Width-FS+1)/2, (Height-FS+1)/2, 	sizeof(short int), 0,  		0, 0,  "Out0", 3),
			KerArg("Out1",        OBJ_OUT_DB_3D, 		(Width-FS+1)/2, (Height-FS+1)/2, 	sizeof(short int), 0,  		0, 0,  "Out1", 3),
			KerArg("Out2",        OBJ_OUT_DB_3D, 		(Width-FS+1)/2, (Height-FS+1)/2, 	sizeof(short int), 0,  		0, 0,  "Out2", 3)
		       ):
		((Nout == 2) ?
		KerArgs(6,
			KerArg("In",   	      OBJ_IN_DB_3D,     	Width,	    	Height, 	 		sizeof(short int), FS-1, 	OBJ_CONSTRAINTS_EVEN, 	0,	"In", 		0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D, 	3*2,	    	3, 	     	 		sizeof(short int), 0,  		0, 						0,  "Filter", 	0),
			KerArg("SetBiasOut0", OBJ_BUFFER_ONETILE,	(Width-FS+1),	(Height-FS+1),   	sizeof(short int), 0,  		0, 						0,  "", 		0),
			KerArg("SetBiasOut1", OBJ_BUFFER_ONETILE,	(Width-FS+1),	(Height-FS+1),   	sizeof(short int), 0,  		0, 						0,  "", 		0),
			KerArg("Out0",        OBJ_OUT_DB_3D, 		(Width-FS+1)/2, (Height-FS+1)/2, 	sizeof(short int), 0,  		0, 						0,  "Out0", 	2),
			KerArg("Out1",        OBJ_OUT_DB_3D, 		(Width-FS+1)/2, (Height-FS+1)/2, 	sizeof(short int), 0,  		0, 						0,  "Out1", 	2)
		       ):
		KerArgs(4,
			KerArg("In",   	      OBJ_IN_DB_3D,     	Width,	    	Height,				sizeof(short int), FS-1, 	OBJ_CONSTRAINTS_EVEN, 	0, "In", 		0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D, 	5,	    		2, 	     	 		sizeof(short int), 0,  		0, 						0,  "Filter", 	0),
			KerArg("SetBiasOut0", OBJ_BUFFER_ONETILE,	(Width-FS+1),	Height-FS+1,    	sizeof(short int), 0,  		0, 						0,  "", 		0),
			KerArg("Out0",        OBJ_OUT_DB_3D, 		(Width-FS+1)/2, (Height-FS+1)/2,	sizeof(short int), 0,  		0, 						0,  "Out0", 	0)
		       )
		)
	);
}

/* Software composite kernels: ReLU then linear layer, pure linear layer on different data format */
void CNN_TiledLinearLayer(char *Name, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height, int ModeSize, int ReLU, int CoeffInL3)

{
	char *KerName, *CoeffType, *OutType;
	unsigned int CSize, OSize;
	unsigned int InSize = (Width*Height)*InPlane;
	unsigned int OutSize = OutPlane;

	switch (ModeSize) {
		case 0:
			CoeffType = "Word8 * __restrict__"; CSize = sizeof(Word8); OutType = "Word16 * __restrict__";  OSize = sizeof(Word16);
			if (ReLU) KerName ="KerLinearLayerReLU_fps"; else KerName = "KerLinearLayer_fps";
			break;
		case 1:
			CoeffType = "Word16 * __restrict__"; CSize = sizeof(Word16); OutType = "Word16 * __restrict__";  OSize = sizeof(Word16);
			if (ReLU) KerName ="KerLinearLayerReLU_fp"; else KerName = "KerLinearLayer_fp";
			break;
		case 2:
			CoeffType = "Word16 * __restrict__"; CSize = sizeof(Word16); OutType = "Word32 * __restrict__";  OSize = sizeof(Word32);
			if (ReLU) KerName ="KerLinearLayerReLU_fpd"; else KerName = "KerLinearLayer_fpd";
			break;
		default:
			GenTilingError("TiledLinearLayer: valid ModeSize = [0, 1, 2] for [Coeff, Out] = [(Byte,Short), (Short, Short), (Short, Int)]");
	}

	UserKernel(Name,
		KernelDimensions(1, Width, Height, 1),
		KernelIterationOrder(KER_DIM2, KER_TILE),
		TILE_HOR,
		CArgs(7,
		      TCArg("Word16 * __restrict__", "In"),
		      TCArg(CoeffType, "Filter"),
		      TCArg("unsigned int", "NormFilter"),
		      TCArg("Word16 * __restrict__", "Bias"),
		      TCArg("unsigned int", "NormBias"),
		      TCArg(OutType, "Out"),
		      TCArg("int", "OutSize")
		     ),
		Calls(1,
			Call(KerName, LOC_INNER_LOOP,
				Bindings(8, K_Arg("In", KER_ARG_TILE), K_Arg("In", KER_ARG_TILE_W), K_Arg("Filter", KER_ARG_TILE), C_Arg("NormFilter"),
					    K_Arg("Bias", KER_ARG_TILE), C_Arg("NormBias"), K_Arg("Out", KER_ARG_TILE), K_Arg("Out", KER_ARG_TILE_H))
			    )
		     ),
		KerArgs(4,
			KerArg("In",   	OBJ_BUFFER_IN_NTILED,				InSize,	1, 			sizeof(Word16), 0, 0, 0, "In", 		0),
			KerArg("Bias", 	OBJ_BUFFER_IN,						1,		OutSize, 	sizeof(Word16), 0, 0, 0, "Bias", 	0),
			KerArg("Filter",CoeffInL3?OBJ_IN_DB_L2DB:OBJ_IN_DB,	InSize,	OutSize, 	CSize, 	 		0, 0, 0, "Filter", 	0),
			KerArg("Out",   OBJ_BUFFER_OUT,						1,		OutSize, 	OSize, 	 		0, 0, 0, "Out", 	0)
			)
	);
}

void CNNConfiguration(unsigned int L1Memory)

{
        SetInlineMode(ALWAYS_INLINE); // SetInlineMode(NEVER_INLINE);
        SetSymbolNames("CNN_L1_Memory", "CNN_L2_Memory", "CNN_KernelDescr", "CNN_KernelArgs");
        SetSymbolDynamics();

        SetUsedFilesNames("KernelLibStdTypes.h", 1, "CNN_BasicKernels.h");
        SetGeneratedFilesNames("CNN_KernelsInit.c", "CNN_KernelsInit.h", "CNN_Kernels.c", "CNN_Kernels.h");

        SetL1MemorySize(L1Memory);
}

