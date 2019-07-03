#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"

#define Max(a, b) (((a)>(b))?(a):(b))
#define Min(a, b) (((a)<(b))?(a):(b))

void LoadCNNLibrary()

{
	/* Templates for Features and coefficients on 16 bits */
	LibKernelTemplate("KerParSetBias_fp_T",
		CArgs(5,
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("short int * __restrict__", "Bias")
		)
	);
	LibKernelTemplate("KerSetBias_fp_T",
		CArgs(4,
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("short int * __restrict__", "Bias")
		)
	);
	LibKernelTemplate("KerSetNormedBias_fp_fps_T",
		CArgs(5,
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("short int", "Bias"),
			TCArg("unsigned short int", "Norm")
		)
	);
	LibKernelTemplate("KerSetNormedBias_fpd_fp_T",
		CArgs(5,
			TCArg("int * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("short int", "Bias"),
			TCArg("unsigned short int", "Norm")
		)
	);
	LibKernelTemplate("KerParConv_fp_T",
		CArgs(18,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseInFeature"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("short int", "TotalInFeatures"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "N"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "D"),
			TCArg("unsigned char", "Ny"),
			TCArg("unsigned char", "Sy"),
			TCArg("unsigned char", "Dy")
		)
	);
	LibKernelTemplate("KerConv_fp_T",
		CArgs(14,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "N"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "Orientation"),
			TCArg("unsigned char", "D"),
			TCArg("unsigned char", "Ny"),
			TCArg("unsigned char", "Sy"),
			TCArg("unsigned char", "Dy")
		)
	);
	LibKernelTemplate("KerParReLUPool_fp_T",
		CArgs(12,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "M"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "Orientation"),
			TCArg("unsigned char", "DoReLU"),
			TCArg("unsigned char", "My"),
			TCArg("unsigned char", "Sy")
		)
	);
	LibKernelTemplate("KerReLUPool_fp_T",
		CArgs(13,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "M"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "Orientation"),
			TCArg("unsigned short int", "DoReLU"),
			TCArg("unsigned char", "D"),
			TCArg("unsigned char", "My"),
			TCArg("unsigned char", "Sy"),
			TCArg("unsigned char", "Dy")
			)
	);
	LibKernelTemplate("KerLinearLayerReLU_fp_T",
		CArgs(10,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "InSize"),
			TCArg("unsigned short int", "TotalInSize"),
			TCArg("unsigned short int", "InBase"),
			TCArg("unsigned short int", "OutSize"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Bias"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "DoReLU")
		)
	);
	LibKernelTemplate("KerParSoftMax_fp_T",
		CArgs(4,
			TCArg("short int *__restrict__", "In"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "Norm"),
			TCArg("short int *__restrict__", "Out")
		)
	);

	/* Templates for Features and coefficients on 8 bits */
	LibKernelTemplate("KerParSetBias_fps_T",
		CArgs(5,
			TCArg("signed char * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("signed char * __restrict__", "Bias")
		)
	);
	LibKernelTemplate("KerSetBias_fps_T",
		CArgs(4,
			TCArg("signed char * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("signed char * __restrict__ ", "Bias")
		)
	);

	LibKernelTemplate("KerParConv_fps_T",
		CArgs(18,
			TCArg("signed char * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "InFeatures"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "BaseInFeature"),
			TCArg("unsigned short int", "BaseOutFeature"),
			TCArg("unsigned short int", "TotalInFeatures"),
			
			TCArg("signed char * __restrict__", "Filter"),
			TCArg("signed char * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "N"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "D"),
			TCArg("unsigned char", "Ny"),
			TCArg("unsigned char", "Sy"),
			TCArg("unsigned char", "Dy")
		)
	);
	LibKernelTemplate("KerConv_fps_T",
		CArgs(14,
			TCArg("signed char * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("signed char * __restrict__", "Filter"),
			TCArg("signed char * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "N"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "Orientation"),
			TCArg("unsigned char", "D"),
			TCArg("unsigned char", "Ny"),
			TCArg("unsigned char", "Sy"),
			TCArg("unsigned char", "Dy")
		)
	);
	LibKernelTemplate("KerParReLUPool_fps_T",
		CArgs(12,
			TCArg("signed char * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("signed char * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "M"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "Orientation"),
			TCArg("unsigned char", "DoReLU"),
			TCArg("unsigned char", "My"),
			TCArg("unsigned char", "Sy")
		)
	);
	LibKernelTemplate("KerReLUPool_fps_T",
		CArgs(13,
			TCArg("signed char * __restrict__", "In"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("signed char * __restrict__", "Out"),
			TCArg("v4s", "Pad"),
			TCArg("unsigned char", "M"),
			TCArg("unsigned char", "S"),
			TCArg("unsigned char", "Orientation"),
			TCArg("unsigned char", "DoReLU"),
			TCArg("unsigned char", "D"),
			TCArg("unsigned char", "My"),
			TCArg("unsigned char", "Sy"),
			TCArg("unsigned char", "Dy")
		)
	);
	LibKernelTemplate("KerLinearLayerReLU_fps_T",
		CArgs(10,
			TCArg("signed char * __restrict__", "In"),
			TCArg("unsigned short int", "InSize"),
			TCArg("unsigned short int", "TotalInSize"),
			TCArg("unsigned short int", "InBase"),
			TCArg("unsigned short int", "OutSize"),
			TCArg("signed char * __restrict__", "Filter"),
			TCArg("signed char * __restrict__", "Bias"),
			TCArg("signed char * __restrict__", "Out"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "DoReLU")
		)
	);
	LibKernelTemplate("KerParSoftMax_fps_T",
		CArgs(4,
			TCArg("signed char *__restrict__", "In"),
			TCArg("unsigned short int", "N"),
			TCArg("unsigned short int", "Norm"),
			TCArg("signed char *__restrict__", "Out")
		)
	);

	/* Templates for Features and coefficients on hybrid formats */
	LibKernelTemplate("KerParSetNormedBias_fp_fps_T",
		CArgs(6,
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "Norm"),
			TCArg("signed char * __restrict__", "Bias")
		)
	);
	LibKernelTemplate("KerParSetNormedBias_fpd_fp_T",
		CArgs(6,
			TCArg("int * __restrict__", "Out"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("unsigned short int", "OutFeatures"),
			TCArg("unsigned short int", "Norm"),
			TCArg("short int * __restrict__", "Bias")
		)
	);
	LibKernelTemplate("KerLinearLayerReLU_fp_fps_fp_T",
		CArgs(11,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "InSize"),
			TCArg("unsigned short int", "TotalInSize"),
			TCArg("unsigned short int", "InBase"),
			TCArg("unsigned short int", "OutSize"),
			TCArg("signed char * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Bias"),
			TCArg("short int * __restrict__", "Out"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "NormBias"),
			TCArg("unsigned char", "DoReLU")
		)
	);
	LibKernelTemplate("KerLinearLayerReLU_fp_fp_fpd_T",
		CArgs(11,
			TCArg("short int * __restrict__", "In"),
			TCArg("unsigned short int", "InSize"),
			TCArg("unsigned short int", "TotalInSize"),
			TCArg("unsigned short int", "InBase"),
			TCArg("unsigned short int", "OutSize"),
			TCArg("short int * __restrict__", "Filter"),
			TCArg("short int * __restrict__", "Bias"),
			TCArg("int * __restrict__", "Out"),
			TCArg("unsigned char", "Norm"),
			TCArg("unsigned char", "NormBias"),
			TCArg("unsigned char", "DoReLU")
		)
	);

	LibKernelTemplate("KerAdd_fp_T",
		CArgs(5,
			TCArg("short int * __restrict__", "In1"),
			TCArg("short int * __restrict__", "In2"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("short int * __restrict__", "Out")
		)
	);

	LibKernelTemplate("KerAddReLU_fp_T",
		CArgs(5,
			TCArg("short int * __restrict__", "In1"),
			TCArg("short int * __restrict__", "In2"),
			TCArg("unsigned short int", "W"),
			TCArg("unsigned short int", "H"),
			TCArg("short int * __restrict__", "Out")
		)
	);

	/****************************************************************************************************************/
	/* Kernels for features and coefficients on 16 bits. Kernels for multiple output features evaluated in parallel */
	/****************************************************************************************************************/

	/* Bias setting */
		LibKernel("KerParSetBias_fp", CALL_PARALLEL, 0, "KerParSetBias_fp_T");

	/* Convolutions */
		LibKernel("KerParConv1x1Stride1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv1x1Stride2_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv1x1StrideS_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");

		LibKernel("KerParConv3x1Stride1x1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv3x1Stride2x1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv1x3Stride1x1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv1x3Stride1x2_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");

		LibKernel("KerParConv3x3Stride1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv3x3Stride2_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv3x3StrideS_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");

		LibKernel("KerParConv5x1Stride1x1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv5x1Stride2x1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv1x5Stride1x1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv1x5Stride1x2_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");

		LibKernel("KerParConv5x5Stride1_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv5x5Stride2_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConv5x5StrideS_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");

		LibKernel("KerParConvNxNStrideS_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");
		LibKernel("KerParConvNxMStrideSxSy_fp", CALL_PARALLEL, 0, "KerParConv_fp_T");

	/* Pooling (Max or Avg) followed by an optional ReLU */
		LibKernel("KerParPool2x2Stride2_fp", CALL_PARALLEL, 0, "KerParReLUPool_fp_T");
		LibKernel("KerParPoolNxNStrideS_fp", CALL_PARALLEL, 0, "KerParReLUPool_fp_T");
		LibKernel("KerParPoolNxMStrideSxSy_fp", CALL_PARALLEL, 0, "KerParReLUPool_fp_T");

	/* Linear Rectification (ReLU) */
		LibKernel("KerParReLU_fp", CALL_PARALLEL, 0, "KerParReLUPool_fp_T");

	/* Linear layer followed by an optional ReLU */
		LibKernel("KerParLinearLayerReLU_fp", CALL_PARALLEL, 0, "KerLinearLayerReLU_fp_T");

	/* SoftMax */
		LibKernel("KerParSoftMax_fp", CALL_PARALLEL, 0, "KerParSoftMax_fp_T");


	/****************************************************************************************************************/
	/* Kernels for Features and coefficients on 16 bits. Kernels for a single feature evaluated in parallel         */
	/****************************************************************************************************************/

	/* Bias setting */
		LibKernel("KerSetBias_fp", CALL_PARALLEL, 0, "KerSetBias_fp_T");

	/* Convolutions */
		LibKernel("KerConv1x1Stride1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv1x1Stride2_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv1x1StrideS_fp", CALL_PARALLEL, 0, "KerConv_fp_T");

		LibKernel("KerConv3x1Stride1x1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv3x1Stride2x1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv1x3Stride1x1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv1x3Stride1x2_fp", CALL_PARALLEL, 0, "KerConv_fp_T");

		LibKernel("KerConv3x3Stride1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv3x3Stride2_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv3x3StrideS_fp", CALL_PARALLEL, 0, "KerConv_fp_T");

		LibKernel("KerConv5x1Stride1x1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv5x1Stride2x1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv1x5Stride1x1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv1x5Stride1x2_fp", CALL_PARALLEL, 0, "KerConv_fp_T");

		LibKernel("KerConv5x5Stride1_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv5x5Stride2_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConv5x5StrideS_fp", CALL_PARALLEL, 0, "KerConv_fp_T");

		LibKernel("KerConvNxNStrideS_fp", CALL_PARALLEL, 0, "KerConv_fp_T");
		LibKernel("KerConvNxMStrideSxSy_fp", CALL_PARALLEL, 0, "KerConv_fp_T");

	/* Pooling (Max or Avg) followed by an optional ReLU */
		LibKernel("KerPool2x2Stride2_fp", CALL_PARALLEL, 0, "KerReLUPool_fp_T");
		LibKernel("KerPoolNxNStrideS_fp", CALL_PARALLEL, 0, "KerReLUPool_fp_T");
		LibKernel("KerPoolNxMStrideSxSy_fp", CALL_PARALLEL, 0, "KerReLUPool_fp_T");

		LibKernel("KerPoolMxNStrideSxT_fp", CALL_PARALLEL, 0, "KerReLUPool_fp_T");

	/* Linear Rectification (ReLU) */
		LibKernel("KerReLU_fp", CALL_PARALLEL, 0, "KerReLUPool_fp_T");

	/* Linear layer followed by an optional ReLU */
		LibKernel("KerLinearLayerReLU_fp", CALL_PARALLEL, 0, "KerLinearLayerReLU_fp_T");

	/****************************************************************************************************************/
	/* Kernels for features and coefficients on 8 bits. Kernels for multiple output features evaluated in parallel  */
	/****************************************************************************************************************/

	/* Bias setting */
		LibKernel("KerParSetBias_fps", CALL_PARALLEL, 0, "KerParSetBias_fps_T");

	/* Convolutions */
		LibKernel("KerParConv1x1Stride1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv1x1Stride2_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv1x1StrideS_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");

		LibKernel("KerParConv3x1Stride1x1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv3x1Stride2x1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv1x3Stride1x1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv1x3Stride1x2_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");

		LibKernel("KerParConv3x3Stride1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv3x3Stride2_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv3x3StrideS_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");

		LibKernel("KerParConv5x1Stride1x1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv5x1Stride2x1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv1x5Stride1x1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv1x5Stride1x2_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");

		LibKernel("KerParConv5x5Stride1_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv5x5Stride2_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConv5x5StrideS_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");

		LibKernel("KerParConvNxNStrideS_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");
		LibKernel("KerParConvNxMStrideSxSy_fps", CALL_PARALLEL, 0, "KerParConv_fps_T");

	/* Pooling (Max or Avg) followed by an optional ReLU */
		LibKernel("KerParPool2x2Stride2_fps", CALL_PARALLEL, 0, "KerParReLUPool_fps_T");
		LibKernel("KerParPoolNxNStrideS_fps", CALL_PARALLEL, 0, "KerParReLUPool_fps_T");
		LibKernel("KerParPoolNxMStrideSxSy_fps", CALL_PARALLEL, 0, "KerParReLUPool_fps_T");

	/* Linear Rectification (ReLU) */
		LibKernel("KerParReLU_fps", CALL_PARALLEL, 0, "KerParReLUPool_fps_T");

	/* Linear layer followed by an optional ReLU */
		LibKernel("KerParLinearLayerReLU_fps", CALL_PARALLEL, 0, "KerLinearLayerReLU_fps_T");

	/* SoftMax */
		LibKernel("KerParSoftMax_fps", CALL_PARALLEL, 0, "KerParSoftMax_fps_T");

	/****************************************************************************************************************/
	/* Kernels for features and coefficients on 8 bits. Kernels for a single feature evaluated in parallel          */
	/****************************************************************************************************************/

	/* Bias setting */
		LibKernel("KerSetBias_fps", CALL_PARALLEL, 0, "KerSetBias_fps_T");

	/* Convolutions */
		LibKernel("KerConv1x1Stride1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv1x1Stride2_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv1x1StrideS_fps", CALL_PARALLEL, 0, "KerConv_fps_T");

		LibKernel("KerConv3x1Stride1x1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv3x1Stride2x1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv1x3Stride1x1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv1x3Stride1x2_fps", CALL_PARALLEL, 0, "KerConv_fps_T");

		LibKernel("KerConv3x3Stride1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv3x3Stride2_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv3x3StrideS_fps", CALL_PARALLEL, 0, "KerConv_fps_T");

		LibKernel("KerConv5x1Stride1x1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv5x1Stride2x1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv1x5Stride1x1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv1x5Stride1x2_fps", CALL_PARALLEL, 0, "KerConv_fps_T");

		LibKernel("KerConv5x5Stride1_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv5x5Stride2_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConv5x5StrideS_fps", CALL_PARALLEL, 0, "KerConv_fps_T");

		LibKernel("KerConvNxNStrideS_fps", CALL_PARALLEL, 0, "KerConv_fps_T");
		LibKernel("KerConvNxMStrideSxSy_fps", CALL_PARALLEL, 0, "KerConv_fps_T");

	/* Pooling (Max or Avg) followed by an optional ReLU */
		LibKernel("KerPool2x2Stride2_fps", CALL_PARALLEL, 0, "KerReLUPool_fps_T");
		LibKernel("KerPoolNxNStrideS_fps", CALL_PARALLEL, 0, "KerReLUPool_fps_T");
		LibKernel("KerPoolNxMStrideSxSy_fps", CALL_PARALLEL, 0, "KerReLUPool_fps_T");

	/* Linear Rectification (ReLU) */
		LibKernel("KerReLU_fps", CALL_PARALLEL, 0, "KerReLUPool_fps_T");

	/* Linear layer followed by an optional ReLU */
		LibKernel("KerLinearLayerReLU_fps", CALL_PARALLEL, 0, "KerLinearLayerReLU_fps_T");

	/****************************************************************************************************************/
	/* Kernels for features/coeffs on different sizes. Kernels for multiple output features evaluated in parallel   */
	/****************************************************************************************************************/

	/* Bias setting */
		LibKernel("KerParSetNormedBias_fp_fps", CALL_PARALLEL, 0, "KerParSetNormedBias_fp_fps_T");
		LibKernel("KerParSetNormedBias_fpd_fp", CALL_PARALLEL, 0, "KerParSetNormedBias_fpd_fp_T");

	/* Linear layer followed by an optional ReLU */
		LibKernel("KerParLinearLayerReLU_fp_fps_fp", CALL_PARALLEL, 0, "KerLinearLayerReLU_fp_fps_fp_T");
		LibKernel("KerParLinearLayerReLU_fp_fp_fpd", CALL_PARALLEL, 0, "KerLinearLayerReLU_fp_fp_fpd_T");

	/****************************************************************************************************************/
	/* Kernels for features/coeffs on different sizes. Kernels for a single feature evaluated in parallel.          */
	/****************************************************************************************************************/

	/* Bias setting */
		LibKernel("KerSetNormedBias_fp_fps", CALL_PARALLEL, 0, "KerSetNormedBias_fp_fps_T");
		LibKernel("KerSetNormedBias_fpd_fp", CALL_PARALLEL, 0, "KerSetNormedBias_fpd_fp_T");

	/* Linear layer followed by an optional ReLU */
		LibKernel("KerLinearLayerReLU_fp_fps_fp", CALL_PARALLEL, 0, "KerLinearLayerReLU_fp_fps_fp_T");
		LibKernel("KerLinearLayerReLU_fp_fp_fpd", CALL_PARALLEL, 0, "KerLinearLayerReLU_fp_fp_fpd_T");

	/****************************************************************************************************************/
	/* Kernels for Add followed by optional Relu.                                                                   */
	/****************************************************************************************************************/

		LibKernel("KerAdd_fp",     CALL_PARALLEL, 0, "KerAddReLU_fp_T");
		LibKernel("KerAddReLU_fp", CALL_PARALLEL, 0, "KerAddReLU_fp_T");

}

static void SelectKernels(
	unsigned int ParallelOutFeat,
	unsigned int DataSize,      // 2: short int, 1: signed char

	unsigned int FScW,
	unsigned int FScH,
	unsigned int ConvStrideW,
	unsigned int ConvStrideH,

	unsigned int DoPool,
	unsigned int FSpW,
	unsigned int FSpH,
	unsigned int PoolStrideW,
	unsigned int PoolStrideH,

	char **ConvKerName,
	char **PoolKerName,
	char **ReLUKerName,
	char **SetBiasKerName,

	unsigned int *NeedConvDim,
	unsigned int *NeedConvStride,
	unsigned int *NeedPoolDim,
	unsigned int *NeedPoolStride
	)
{
	char *_ConvKerName, *_PoolKerName, *_ReLUKerName, *_SetBiasKerName;
	unsigned int _NeedConvDim, _NeedConvStride, _NeedPoolDim, _NeedPoolStride;
	char *KerSuffix=(DataSize==2)?"_fp":"_fps";
	int FSc=0; 
	int FSp=0;
	int ConvStrideSame=0;
	//Setting FSc to !=0 just if the filter is quare and in case == 1 or 3 or 5
	//otherwise using default case
	if(FScH == FScW && (FScW==1 || FScW==3 ||FScW==5)) FSc = FScW;
	if(ConvStrideH == ConvStrideW ) ConvStrideSame = ConvStrideW;
	if(FSpH == FSpW &&  FSpW==2) FSp = FSpW;

	if (DataSize!=2 && DataSize!=1) GenTilingError("SelectKernels: Unsupported data size: %d\n", DataSize);
	
	_NeedConvDim = 0; _NeedConvStride = 0; _NeedPoolDim = 0; _NeedPoolStride = 0;

	if (ParallelOutFeat) {
		_ReLUKerName = "KerParReLU";
		_SetBiasKerName = "KerParSetBias";
		
		if(FScW == 1 && FScH == 1 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv1x1Stride1";
		}
		else if(FScW == 1 && FScH == 1 && ConvStrideW==2 && ConvStrideH==2)
		{
			_ConvKerName = "KerParConv1x1Stride2";
		}
		else if(FScW == 1 && FScH == 1 && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerParConv1x1StrideS"; _NeedConvStride = 1;
		}
		else if(FScW == 3 && FScH == 1 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv3x1Stride1x1";
		}
		else if(FScW == 3 && FScH == 1 && ConvStrideW==2 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv3x1Stride2x1";
		}
		else if(FScW == 1 && FScH == 3 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv1x3Stride1x1";
		}	
		else if(FScW == 1 && FScH == 3 && ConvStrideW==1 && ConvStrideH==2)
		{
			_ConvKerName = "KerParConv1x3Stride1x2";
		}
		else if(FScW == 3 && FScH == 3 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv3x3Stride1";
		}
		else if(FScW == 3 && FScH == 3 && ConvStrideW==2 && ConvStrideH==2)
		{
			_ConvKerName = "KerParConv3x3Stride2";
		}
		else if(FScW == 3 && FScH == 3 && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerParConv3x3StrideS"; _NeedConvStride = 1;
		}
		else if(FScW == 5 && FScH == 1 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv5x1Stride1x1";
		}
		else if(FScW == 5 && FScH == 1 && ConvStrideW==2 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv5x1Stride2x1";
		}
		else if(FScW == 1 && FScH == 5 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv1x5Stride1x1";
		}	
		else if(FScW == 1 && FScH == 5 && ConvStrideW==1 && ConvStrideH==2)
		{
			_ConvKerName = "KerParConv1x5Stride1x2";
		}
		else if(FScW == 5 && FScH == 5 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerParConv5x5Stride1";
		}
		else if(FScW == 5 && FScH == 5 && ConvStrideW==2 && ConvStrideH==2)
		{
			_ConvKerName = "KerParConv5x5Stride2";
		}
		else if(FScW == 5 && FScH == 5 && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerParConv5x5StrideS"; _NeedConvStride = 1;
		}
		else if(FScW == FScH && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerParConvNxNStrideS"; _NeedConvDim = 1; _NeedConvStride = 1;
		}
		else
		{
			_ConvKerName = "KerParConvNxMStrideSxSy"; _NeedConvDim = 1; _NeedConvStride = 1;
		}

		if (DoPool) {
			if(FSpW == 2 && FSpH == 2 && PoolStrideW==2 && PoolStrideH==2)
			{
				_PoolKerName = "KerParPool2x2Stride2"; 
			}
			else if(FSpW == FSpH && PoolStrideW==PoolStrideH)
			{
				_PoolKerName = "KerParPoolNxNStrideS";  _NeedPoolDim = 1; _NeedPoolStride = 1;
			}
			else
			{
				_PoolKerName = "KerParPoolNxMStrideSxSy";  _NeedPoolDim = 1; _NeedPoolStride = 1;
			}
		} else _PoolKerName = "";


	} else {
		_ReLUKerName = "KerReLU";
		_SetBiasKerName = "KerSetBias";
		
		if(FScW == 1 && FScH == 1 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv1x1Stride1";
		}
		else if(FScW == 1 && FScH == 1 && ConvStrideW==2 && ConvStrideH==2)
		{
			_ConvKerName = "KerConv1x1Stride2";
		}
		else if(FScW == 1 && FScH == 1 && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerConv1x1StrideS"; _NeedConvStride = 1;
		}
		else if(FScW == 3 && FScH == 1 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv3x1Stride1x1";
		}
		else if(FScW == 3 && FScH == 1 && ConvStrideW==2 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv3x1Stride2x1";
		}
		else if(FScW == 1 && FScH == 3 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv1x3Stride1x1";
		}	
		else if(FScW == 1 && FScH == 3 && ConvStrideW==1 && ConvStrideH==2)
		{
			_ConvKerName = "KerConv1x3Stride1x2";
		}
		else if(FScW == 3 && FScH == 3 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv3x3Stride1";
		}
		else if(FScW == 3 && FScH == 3 && ConvStrideW==2 && ConvStrideH==2)
		{
			_ConvKerName = "KerConv3x3Stride2";
		}
		else if(FScW == 3 && FScH == 3 && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerConv3x3StrideS"; _NeedConvStride = 1;
		}
		else if(FScW == 5 && FScH == 1 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv5x1Stride1x1";
		}
		else if(FScW == 5 && FScH == 1 && ConvStrideW==2 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv5x1Stride2x1";
		}
		else if(FScW == 1 && FScH == 5 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv1x5Stride1x1";
		}	
		else if(FScW == 1 && FScH == 5 && ConvStrideW==1 && ConvStrideH==2)
		{
			_ConvKerName = "KerConv1x5Stride1x2";
		}
		else if(FScW == 5 && FScH == 5 && ConvStrideW==1 && ConvStrideH==1)
		{
			_ConvKerName = "KerConv5x5Stride1";
		}
		else if(FScW == 5 && FScH == 5 && ConvStrideW==2 && ConvStrideH==2)
		{
			_ConvKerName = "KerConv5x5Stride2";
		}
		else if(FScW == 5 && FScH == 5 && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerConv5x5StrideS"; _NeedConvStride = 1;
		}
		else if(FScW == FScH && ConvStrideW==ConvStrideH)
		{
			_ConvKerName = "KerConvNxNStrideS"; _NeedConvDim = 1; _NeedConvStride = 1;
		}
		else
		{
			_ConvKerName = "KerConvNxMStrideSxSy"; _NeedConvDim = 1; _NeedConvStride = 1;
		}

		if (DoPool) {
			if(FSpW == 2 && FSpH == 2 && PoolStrideW==2 && PoolStrideH==2)
			{
				_PoolKerName = "KerPool2x2Stride2"; 
			}
			else if(FSpW == FSpH && PoolStrideW==PoolStrideH)
			{
				_PoolKerName = "KerPoolNxNStrideS";  _NeedPoolDim = 1; _NeedPoolStride = 1;
			}
			else
			{
				_PoolKerName = "KerPoolNxMStrideSxSy";  _NeedPoolDim = 1; _NeedPoolStride = 1;
			}
		} else _PoolKerName = "";


	}
	if (ReLUKerName) *ReLUKerName = AppendNames(_ReLUKerName, KerSuffix);
	if (SetBiasKerName) *SetBiasKerName = AppendNames(_SetBiasKerName, KerSuffix);
	if (ConvKerName) *ConvKerName = AppendNames(_ConvKerName, KerSuffix);
	if (PoolKerName) *PoolKerName = AppendNames(_PoolKerName, KerSuffix);

	if (NeedConvDim) *NeedConvDim = _NeedConvDim;
	if (NeedConvStride) *NeedConvStride = _NeedConvStride;
	if (NeedPoolDim) *NeedPoolDim = _NeedPoolDim;
	if (NeedPoolStride) *NeedPoolStride = _NeedPoolStride;
}

/*********************************************************************************************************************************************************************
	Generators for Convolutions, followed by an optional pooling (Max or Average), followed by an optional linear rectification (ReLU).

	Template:
		Name:       Name of the generated user kernel

		In_DataSize:    1: byte, 2: half word, 4: word
		Filter_DataSize:1: byte, 2: half word, 4: word
		Bias_DataSize:  1: byte, 2: half word, 4: word
		Out_DataSize:   1: byte, 2: half word, 4: word

		In_InL3:    0: In is in L2, 1: In is in L3 memory
		Filter_InL3:    0: Filter is in L2, 1: Filter is in L3 memory
		Bias_InL3:  0: Bias is in L2, 1: Bias is in L3 memory
		Out_InL3:   0: Out is in L2, 1: Out is in L3 memory

		InFeat:     Number of input feature's maps
		OutFeat:    Number of output feature's maps
		Width:      Number of columns of a given feature map
		Height:     Number of lines of a given feature map

		FSc:        Dimension of the convolution (FSc x FSc)
		ConvStride: Stride between 2 points in the input feature map where convolution is evaluated
		ConvDoPad:  0: No padding, 1: Zero padding
		ConvDoReLU: Performs linear rectification after all convolutions for a given output feature's map have been evaluated

		FSp:        Size of the pooling region (FSp x FSp)
		PoolStride: Stride between 2 points in the input feature map where pooling is evaluated
		PoolDoPad:  0: No padding, 1: Zero padding
		PoolDoReLU: 0: No linear rectification, 1: Linear rectification after pooling
		DoPool:     0: No pooling, 1: Max pooling, 2: Average pooling

	Currently only homegeneous data size are supported (bytes and hald words)

	CNN_LargeConvolutionPoolReLU_Hor
		For a given output feature's map all input feature's maps are traversed. A given input feature map is tiled and the evaluation of the tile is
		dispatched on all cores for parallel evaluation. Before starting the traversal of all input feature's maps bias is assigned to the output tile.
		Once all input feature's maps have been traversed the produced output tile is optionaly passed to a pooling layer and/or to a linear rectification layer.
		Note that with this approach all input featire's maps are traversed several time (OutFeat time).
		Tile orientation is horizontal.

	CNN_LargeConvolutionPoolReLU_Ver
		For a given output feature's map all input feature's maps are traversed. A given input feature map is tiled and the evaluation of the tile is
		dispatched on all cores for parallel evaluation. Before starting the traversal of all input feature's maps bias is assigned to the output tile.
		Once all input feature's maps have been traversed the produced output tile is optionaly passed to a pooling layer and/or to a linear rectification layer.
		Note that with this approach all input featire's maps are traversed several time (OutFeat time).
		Tile orientation is vertical.

	CNN_LargeParOutFeatConvolutionPoolReLU_Hor
		All output feature's maps are evaluated in parallel. A tile crosses the entire set of output feature's maps. Input feature's maps are traversed in sequence
		through tiles correspondingi, in the input maps, to the currently evalauted output feature map's tile and combined with corresponding filter (if filter does
		not fit into cluster's L1 then filter is tiled in the inner loop).
		Here input feature's maps are traversed only once therefore this generator is good when input features are in L3, the real constraints comes from the fact that the
		tile traversing all the input features must fit into cluster's L1.
		Tiling is horizontal.

	CNN_LargeParOutFeatConvolutionPoolReLU_Ver
		All output feature's maps are evaluated in parallel. A tile crosses the entire set of output feature's maps. Input feature's maps are traversed in sequence
		through tiles correspondingi, in the input maps, to the currently evalauted output feature map's tile and combined with corresponding filter (if filter does
		not fit into cluster's L1 then filter is tiled in the inner loop).
		Here input feature's maps are traversed only once therefore this generator is good when input features are in L3, the real constraints comes from the fact that the
		tile traversing all the input features must fit into cluster's L1.
		Tiling is vertical.

	CNN_SmallParOutFeatConvolutionPoolReLU
		Output feature's maps are evaluated in parallel, an output tile contains several full output feature's maps (prefered to be a multiple of number of cores for
		efficient parallelization), filter is the same iteration space. If cluster's L1 memory can accomodate the entire set of input feature's maps they are installed first,
		This generator is generally good when InFeat and OutFeat are large but Width and Height small (10 or bellow), if input can be promoted to buffer in cluster's L1
		then inputs are loaded only once, if not they will be loaded as many time as they are tiles in the output feature's maps (usually a small number).
		if not the set is tiled in the inner loop of the user's kernel, a tile contains a group of full input feature's maps.


*********************************************************************************************************************************************************************/


void CNN_LargeConvolutionPoolReLU_Hor(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Filter_DataSize,
			unsigned int Bias_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Filter_InL3,
			unsigned int Bias_InL3,
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FScW,
			unsigned int FScH,
			unsigned int ConvStrideW,
			unsigned int ConvStrideH,
			unsigned int ConvDilationW,
			unsigned int ConvDilationH,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			
			int          PoolDoPad,
			int          PoolDoReLU,
			int          DoPool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	if (DoPool==0) {

		FSpW=1;FSpH=1; PoolStrideW=1;PoolStrideH=1;
	}

	unsigned int TileCons = (ConvStrideH * FSpH) + FScH - ConvStrideH;
	int Overlap = FScH + ConvStrideH * (FSpH-PoolStrideH-1);
	int ConvOverlap = FScH - ConvStrideH;

	unsigned int Wo, Ho, Wc, Hc;
	//  PadTc Pading Top    convolutions
	//  PadBc Pading Bottom convolutions
	//  PadLc Pading Left   convolutions
	//  PadRc Pading Right  convolutions
	//  PadTp Pading Top    Pooling
	//  PadBp Pading Bottom Pooling
	//  PadRp Pading Right  Pooling
	//  PadLp Pading Left   Pooling
	int PadTc, PadBc, PadRc, PadLc, PadTp, PadBp,PadLp, PadRp;

	//TODO: Variables for test, needs to be updated
	v4s PadInc = (v4s){0,0,0,0};
	v4s PadInp = (v4s){0,0,0,0};

	char *ConvKerName, *PoolKerName, *ReLUKerName, *SetBiasKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (In_DataSize != Filter_DataSize || Filter_DataSize != Bias_DataSize || Bias_DataSize != Out_DataSize)
		GenTilingError("CNN_LargeConvolutionPoolReLU_Hor %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(0, DataSize, FScW,FScH , ConvStrideW, ConvStrideH, DoPool, FSpW,FSpH, PoolStrideW, PoolStrideH,
			&ConvKerName, &PoolKerName, &ReLUKerName, &SetBiasKerName,
			&NeedConvDim, &NeedConvStride, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (ConvDoPad)
	{
		PadTc = (FScH-1)/2; PadBc = FScH/2;
		PadLc = (FScW-1)/2; PadRc = FScW/2;
		
		PadInc = (v4s){(FScW-1)/2,FScW/2,(FScH-1)/2, FScH/2};
		
		//Output Width and Height
		Wc = (Width  - FScW+((FScW-1)/2)+(FScW/2))/ConvStrideW + 1;
		Hc = (Height - FScH+((FScH-1)/2)+(FScH/2))/ConvStrideH + 1;
	} 
	else 
	{
		PadTc = 0; PadBc = 0;
		Wc = (Width- FScW)/ConvStrideW + 1;
		Hc = (Height-FScH)/ConvStrideH + 1;
	}
	
	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadLp = (FSpW-1)/2; PadRp = FSpW/2;
			
			PadInp = (v4s){PadLp,PadRp,PadTp,PadBp};
			
			Wo = (Wc-FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Hc-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			PadTp = 0; PadBp = 0;
			Wo = (Wc-FSpW)/PoolStrideW + 1;
			Ho = (Hc-FSpH)/PoolStrideH + 1;
		}
	} else {
		PadLp = 0; PadRp = 0;PadTp = 0; PadBp = 0;
		Wo = Wc; Ho = Hc;
	}

		UserKernel(Name,
		KernelDimensions(InFeat, Width, Height, OutFeat),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
			TILE_HOR,
			CArgs(5,
				  TCArg(DataType,       "In"),
				  TCArg(DataType,       "Filter"),
				  TCArg(DataType,       "Bias"),
				  TCArg(DataType,       "Out"),
				  TCArg("unsigned int", "Norm")
				 ),
			Calls(3,
				Call(SetBiasKerName, LOC_IN_PLANE_PROLOG,
				Bindings(4, 
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE), 
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H), 
					C_ArgIndirectIndex("Bias", KER_OUT_PLANE, 1))),
				Call(ConvKerName, LOC_IN_PLANE,
				Bindings(14,
					K_Arg("In", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_W),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_PAD),
					C_Arg("Norm"),
					NeedConvDim?Imm(FScW):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideW):AT_IGNORE_ARG_BINDING,
					Imm(1), //Orientation
					Imm(1), //Will be dilation x
					NeedConvDim?Imm(FScH):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideH):AT_IGNORE_ARG_BINDING,
					Imm(1) //Will be dilation y
					)
				),
				ConvDoReLU?
				Call(ReLUKerName, LOC_IN_PLANE_EPILOG,
				Bindings(13,
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING
				)
				):(DoPool?
				Call(PoolKerName, LOC_IN_PLANE_EPILOG,
				Bindings(13,
					K_Arg("ConvOut", KER_ARG_TILE),
					K_Arg("ConvOut", KER_ARG_TILE_W),
					K_Arg("ConvOut", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					K_Arg("ConvOut", KER_ARG_TILE_PAD),
					NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,
					Imm(1),
					Imm(((DoPool-1)<<1)|PoolDoReLU),
					Imm(1), //Dilation x
					NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING,
					Imm(1) //Dilation y
				)
				):AT_NO_CALL)
			),
			KerArgs(4,
				ConvDoPad?
				KerArgP("In",      OBJ_IN_DB_3D|InL3,            Width, Height, PadLc,PadRc,PadTc,PadBc, In_DataSize,     Overlap, OBJ_CONSTRAINTS_PAD_REM,  TileCons, "In", 0):
				KerArg ("In",      OBJ_IN_DB_3D|InL3,            Width, Height,                          In_DataSize,     Overlap, OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
				KerArg ("Filter",  OBJ_IN_DB_NTILED_4D|FilterL3, FScW,   FScH,                           Filter_DataSize,       0,                        0,        0, "Filter", 0),
				KerArg ("Out",     OBJ_OUT_DB_3D|OutL3,          Wo,    Ho,                              Out_DataSize,          0,                        0,        0, "Out", 0),
				DoPool?(PoolDoPad?
				KerArgP("ConvOut", OBJ_BUFFER_ONETILE,           Wc,    Hc,     PadLp,PadRp,PadTp,PadBp, Out_DataSize, ConvOverlap, OBJ_CONSTRAINTS_PAD_REM,         0,  "", 0):
				KerArg ("ConvOut", OBJ_BUFFER_ONETILE,           Wc,    Hc,                              Out_DataSize, ConvOverlap, OBJ_CONSTRAINTS_DROP_REM,        0,  "", 0)):
				AT_NO_KER_ARG
			)
		);

	AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize},
			  (v4s) {InL3!=0,FilterL3!=0,BiasL3!=0,OutL3!=0},
			  InFeat, OutFeat, Width, Height,
			  FScW, FScH, ConvStrideW, ConvStrideH, PadInc, ConvDoReLU, 1,
			  FSpW, FSpH, PoolStrideW, PoolStrideH, PadInp, PoolDoReLU, DoPool,
			  0, 0, 0, (DataSize==1)?7:15, 0);

}



void CNN_LargeConvolutionPoolReLU_Ver(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Filter_DataSize,
			unsigned int Bias_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Filter_InL3,
			unsigned int Bias_InL3,
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FScW,
			unsigned int FScH,
			unsigned int ConvStrideW,
			unsigned int ConvStrideH,
			unsigned int ConvDilationW,
			unsigned int ConvDilationH,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			
			int          PoolDoPad,
			int          PoolDoReLU,
			int          DoPool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	if (DoPool==0) {
		FSpW=1; FSpH=1; PoolStrideW=1; PoolStrideH=1; 
	}

	unsigned int TileCons = (ConvStrideW * FSpW) + FScW - ConvStrideW;
	int Overlap = FScW + ConvStrideW * (FSpW-PoolStrideW-1);
	int ConvOverlap = FScW - ConvStrideW;

	unsigned int Wo, Ho, Wc, Hc;
	int PadTc, PadBc, PadRc, PadLc,PadTp, PadBp,PadLp, PadRp;

	v4s PadInc = (v4s){0,0,0,0};
	v4s PadInp = (v4s){0,0,0,0};
	char *ConvKerName, *PoolKerName, *ReLUKerName, *SetBiasKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (In_DataSize != Filter_DataSize || Filter_DataSize != Bias_DataSize || Bias_DataSize != Out_DataSize)
		GenTilingError("CNN_LargeConvolutionPoolReLU_Ver %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(0, DataSize, FScW, FScH, ConvStrideW, ConvStrideH, DoPool, FSpW,FSpH, PoolStrideW, PoolStrideH,
			  &ConvKerName, &PoolKerName, &ReLUKerName, &SetBiasKerName,
			  &NeedConvDim, &NeedConvStride, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (ConvDoPad) 
	{

		PadTc = (FScH-1)/2; PadBc = FScH/2;
		PadLc = (FScW-1)/2; PadRc = FScW/2;
		
		PadInc = (v4s){(FScW-1)/2,FScW/2,(FScH-1)/2, FScH/2};
		
		Wc = (Width- FScW+((FScW-1)/2)+(FScW/2))/ConvStrideW + 1;
		Hc = (Height-FScH+((FScH-1)/2)+(FScH/2))/ConvStrideH + 1;
	} else {
		PadTc = 0; PadBc = 0;
		PadLc = 0; PadRc = 0;

		Wc = (Width- FScW)/ConvStrideW + 1;
		Hc = (Height-FScH)/ConvStrideH + 1;
	}
	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadLp = (FSpW-1)/2; PadRp = FSpW/2;
			PadInp = (v4s){PadTp,PadBp,0,0};
			Wo = (Wc-FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Hc-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			PadTp = 0; PadBp = 0; PadLp = 0; PadRp = 0;
			Wo = (Wc-FSpW)/PoolStrideW + 1;
			Ho = (Hc-FSpH)/PoolStrideH + 1;
		}
	} else {
		PadTp = 0; PadBp = 0; PadLp = 0; PadRp = 0;
		Wo = Wc; Ho = Hc;
	}


		UserKernel(Name,
		KernelDimensions(InFeat, Width, Height, OutFeat),
		KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
			TILE_VER,
			CArgs(5,
				  TCArg(DataType,       "In"),
				  TCArg(DataType,       "Filter"),
				  TCArg(DataType,       "Bias"),
				  TCArg(DataType,       "Out"),
				  TCArg("unsigned int", "Norm")
				 ),
			Calls(3,
				Call(SetBiasKerName, LOC_IN_PLANE_PROLOG,
				Bindings(4, 
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE), 
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H), 
					C_ArgIndirectIndex("Bias", KER_OUT_PLANE, 1))),
				Call(ConvKerName, LOC_IN_PLANE,
				Bindings(14,
					K_Arg("In", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_W),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_PAD),
					C_Arg("Norm"),
					NeedConvDim?Imm(FScW):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideW):AT_IGNORE_ARG_BINDING,
					Imm(0), //Orientation
					Imm(1), //Will be dilation x
					NeedConvDim?Imm(FScH):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideH):AT_IGNORE_ARG_BINDING,
					Imm(1) //Will be dilation y
					)
				),
				ConvDoReLU?
					Call(ReLUKerName, LOC_IN_PLANE_EPILOG,
					Bindings(13,
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
						K_Arg("Out", KER_ARG_TILE),
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING
					))
				:(DoPool?
					Call(PoolKerName, LOC_IN_PLANE_EPILOG,
					Bindings(13,
						K_Arg("ConvOut", KER_ARG_TILE),
						K_Arg("ConvOut", KER_ARG_TILE_W),
						K_Arg("ConvOut", KER_ARG_TILE_H),
						K_Arg("Out", KER_ARG_TILE),
						K_Arg("ConvOut", KER_ARG_TILE_PAD),
						NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,
						NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,
						Imm(0), //Orientation
						Imm(((DoPool-1)<<1)|PoolDoReLU),
						Imm(1), //Dilation x
						NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
						NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING,
						Imm(1) //Dilation y
					))
				:AT_NO_CALL)
			),
			KerArgs(4,
				ConvDoPad?
				KerArgP("In",      OBJ_IN_DB_3D|InL3,            Width, Height, PadLc,PadRc,PadTc,PadBc, In_DataSize,     Overlap, OBJ_CONSTRAINTS_PAD_REM,  TileCons, "In", 0):
				KerArg ("In",      OBJ_IN_DB_3D|InL3,            Width, Height,                          In_DataSize,     Overlap, OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
				KerArg ("Filter",  OBJ_IN_DB_NTILED_4D|FilterL3, FScW,   FScH,                           Filter_DataSize,       0,                        0,        0, "Filter", 0),
				KerArg ("Out",     OBJ_OUT_DB_3D|OutL3,          Wo,    Ho,                              Out_DataSize,          0,                        0,        0, "Out", 0),
				DoPool?(PoolDoPad?
				KerArgP("ConvOut", OBJ_BUFFER_ONETILE,           Wc,    Hc,     PadLp,PadRp,PadTp,PadBp, Out_DataSize, ConvOverlap, OBJ_CONSTRAINTS_PAD_REM,         0,  "", 0):
				KerArg ("ConvOut", OBJ_BUFFER_ONETILE,           Wc,    Hc,                              Out_DataSize, ConvOverlap, OBJ_CONSTRAINTS_DROP_REM,        0,  "", 0)):
				AT_NO_KER_ARG
		)
		);
	
	AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize},
			  (v4s) {InL3!=0,FilterL3!=0,BiasL3!=0,OutL3!=0},
			  InFeat, OutFeat, Width, Height,
			  FScW, FScH, ConvStrideW,ConvStrideH, PadInc, ConvDoReLU, 1,
			  FSpW, FSpH, PoolStrideW,PoolStrideH, PadInp, PoolDoReLU, DoPool,
			  0, 0, 0, (DataSize==1)?7:15, 0);
}



void CNN_LargeParOutFeatConvolutionPoolReLU_Hor(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Filter_DataSize,
			unsigned int Bias_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Filter_InL3,
			unsigned int Bias_InL3,
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FScW,
			unsigned int FScH,
			unsigned int ConvStrideW,
			unsigned int ConvStrideH,
			unsigned int ConvDilationW,
			unsigned int ConvDilationH,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			
			int          PoolDoPad,
			int          PoolDoReLU,
			int          DoPool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	unsigned int InType;
	if (DoPool==0) {
		FSpW=1;FSpH=1; PoolStrideW=1;PoolStrideH=1;
	}
	
	unsigned int TileCons = (ConvStrideH * FSpH) + FScH - ConvStrideH;
	int Overlap = FScH + ConvStrideH * (FSpH-PoolStrideH-1);
	int ConvOverlap = FScH - ConvStrideH;


	unsigned int Wo, Ho, Wc, Hc;
	int PadLc, PadRc, PadTc, PadBc, PadTp, PadBp,  PadLp, PadRp;
	v4s PadInp = (v4s){0,0,0,0};
	v4s PadInc = (v4s){0,0,0,0};
	char *ConvKerName, *PoolKerName, *ReLUKerName, *SetBiasKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (In_DataSize != Filter_DataSize || Filter_DataSize != Bias_DataSize || Bias_DataSize != Out_DataSize)
		GenTilingError("CNN_LargeParOutFeatConvolutionPoolReLU_Hor %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(1, DataSize, FScW,FScH, ConvStrideW, ConvStrideH, DoPool, FSpW,FSpH, PoolStrideW, PoolStrideH,
			  &ConvKerName, &PoolKerName, &ReLUKerName, &SetBiasKerName,
			  &NeedConvDim, &NeedConvStride, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (ConvDoPad) {
		PadTc = (FScH-1)/2; PadBc = FScH/2;
		PadRc = (FScW-1)/2; PadLc = FScW/2;

		PadInc = (v4s){(FScW-1)/2,FScW/2,(FScH-1)/2, FScH/2};
		Wc = (Width- FScW+((FScW-1)/2)+(FScW/2))/ConvStrideW + 1;
		Hc = (Height-FScH+((FScH-1)/2)+(FScH/2))/ConvStrideH + 1;
	} else {
		PadTc = 0; PadBc = 0; PadLc = 0; PadRc = 0;
		Wc = (Width- FScW)/ConvStrideW + 1;
		Hc = (Height-FScH)/ConvStrideH + 1;
	}
	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadLp = (FSpW-1)/2; PadRp = FSpW/2;
			PadInp = (v4s){0,0,PadTp,PadBp};
			Wo = (Wc-FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Hc-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			PadTp = 0; PadBp = 0;
			PadLp = 0; PadRp = 0;
			Wo = ((Wc-FSpW)/PoolStrideW) + 1;
			Ho = (Hc-FSpH)/PoolStrideH + 1;
		}
	} else {
		PadTp = 0; PadBp = 0;
		PadLp = 0; PadRp = 0;
		Wo = Wc; Ho = Hc;
	}
	
	unsigned int K=1; // Min(InFeat, 4);

	if (InFeat==1) InType = OBJ_IN_DB;
	else {
		if      ((InFeat % 32) == 0) K = 32;
		else if ((InFeat % 16) == 0) K = 16;
		else if ((InFeat % 8) == 0) K = 8;
		else if ((InFeat % 4) == 0) K = 4;
		else if ((InFeat % 2) == 0) K = 2;
	}
	while (1) {
		Kernel_T *U_Ker;
		if (InFeat==K) InType = OBJ_IN_DB; else InType = OBJ_IN_DB_3D;
			U_Ker = UserKernel(Name,
			KernelDimensionsAndUserSymbols(InFeat/K, Width, Height, OutFeat,
				KerDynamicSymbols(2,
					S_Dyn("Wc", Wc),
					S_Dyn("Hc", Hc)
				)
			),
			(InFeat==1)?  KernelIterationOrder(KER_DIM3, KER_TILE, KER_TILE1):KernelIterationOrder(KER_DIM4, KER_TILE, KER_IN_PLANE, KER_TILE1),
					TILE_HOR,
					CArgs(5,
						  TCArg(DataType,       "In"),
						  TCArg(DataType,       "Filter"),
						  TCArg(DataType,       "Bias"),
						  TCArg(DataType,       "Out"),
						  TCArg("unsigned int", "Norm")
						 ),
					Calls(3,
				Call(SetBiasKerName, (InFeat==1)?LOC_INNER_LOOP1_PROLOG:LOC_IN_PLANE_PROLOG,
					Bindings(5,
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
						UserSymb("Wc"),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
						KerDim(K_OUTP),
						K_Arg("Bias", KER_ARG_TILE)
					)
				),
				Call(ConvKerName, LOC_INNER_LOOP1,
					Bindings(18,
						K_Arg("In", KER_ARG_TILE),
						KerDim(K_W),
						K_Arg("In", KER_ARG_TILE_H),
						Imm(K),                     // Number of input features in this tile
						K_Arg("Filter", KER_ARG_TILE_H),        // Number of output features in this tile
						K_ArgOper("In", KER_ARG_INPLANEINDEX, '*', K),  // Where to start in In Features, should be multiplied by K
						K_Arg("Filter", KER_ARG_TILE_BASE),         // From where to start Out Features in this tile
						Imm(InFeat),                    // Total number of input features
						K_Arg("Filter", KER_ARG_TILE),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
						K_Arg("In", KER_ARG_TILE_PAD),
						C_Arg("Norm"),
						NeedConvDim?Imm(FScW):AT_IGNORE_ARG_BINDING,
						NeedConvStride?Imm(ConvStrideW):AT_IGNORE_ARG_BINDING,
						Imm(1), //This will be dilation x
						NeedConvDim?Imm(FScH):AT_IGNORE_ARG_BINDING,
						NeedConvStride?Imm(ConvStrideH):AT_IGNORE_ARG_BINDING,
						Imm(1) //This will be dilation y
						)
				),
				ConvDoReLU?
				Call(ReLUKerName, (InFeat==1)?LOC_INNER_LOOP1_EPILOG:LOC_IN_PLANE_EPILOG,
					Bindings(12,
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
						UserSymb("Wc"),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
						KerDim(K_OUTP),
						K_Arg("Out", KER_ARG_TILE),
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING
					)
				):(DoPool?
				Call(PoolKerName, (InFeat==1)?LOC_INNER_LOOP1_EPILOG:LOC_IN_PLANE_EPILOG,
					Bindings(12,
						K_Arg("ConvOut", KER_ARG_TILE),
						UserSymb("Wc"),
						K_Arg("ConvOut", KER_ARG_TILE_H),
						KerDim(K_OUTP),
						K_Arg("Out", KER_ARG_TILE),
						K_Arg("ConvOut", KER_ARG_TILE_PAD),
						NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,            
						NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,  						
						AT_IGNORE_ARG_BINDING,
						Imm(((DoPool-1)<<1)|PoolDoReLU),
						NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
						NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING
					)
				):AT_NO_CALL)
						 ),
					KerArgs(5,
				ConvDoPad?
				KerArg2DP("In",      InType|InL3,                 Width*K, Height,   Width, PadLc, PadRc, PadTc, PadBc, In_DataSize,             Overlap, OBJ_CONSTRAINTS_PAD_REM,  TileCons, "In", 0):
				KerArg2D ("In",      InType|InL3,                 Width*K, Height,   Width,                             In_DataSize,             Overlap, OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
				KerArg   ("Filter",  OBJ_IN_DB|O_TILE1|FilterL3,  InFeat, OutFeat,                                      FScW*FScH*Filter_DataSize, 0,       0,                        8,        "Filter", 0),
				KerArg   ("Bias",    OBJ_BUFFER_IN_NTILED|BiasL3, 1,      OutFeat,                                      Bias_DataSize,             0,       0,                        0,        "Bias", 0),
				KerArg2D ("Out",     OBJ_OUT_DB|OutL3,            Wo*OutFeat,  Ho,      Wo,                             Out_DataSize,              0,       0,                        0,        "Out", 0),
				DoPool?(PoolDoPad?
				KerArgP ("ConvOut",  OBJ_BUFFER_ONETILE,          Wc*OutFeat,  Hc,         PadLp, PadRp, PadTp, PadBp,  Out_DataSize,     ConvOverlap,       OBJ_CONSTRAINTS_PAD_REM,  0,        "", 0):
				KerArg  ("ConvOut",  OBJ_BUFFER_ONETILE,          Wc*OutFeat,  Hc,                                      Out_DataSize,     ConvOverlap,       OBJ_CONSTRAINTS_DROP_REM, 0,        "", 0)):
				AT_NO_KER_ARG
			)
			);
		if (U_Ker==0) {
			if (K!=1) K=K/2;
			else GenTilingError("Kernel: %s, not enough shared L1 memory to find a tiling solution", Name);
		} else break;
	}

	AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize},
			  (v4s) {InL3!=0,FilterL3!=0,BiasL3!=0,OutL3!=0},
			  InFeat, OutFeat, Width, Height,
			  FScW, FScH, ConvStrideW, ConvStrideH, PadInc, ConvDoReLU, 1,
			  FSpW, FSpH, PoolStrideW, PoolStrideH, PadInp, PoolDoReLU, DoPool,
			  0, 0, 0, (DataSize==1)?7:15, 0);
}



void CNN_LargeParOutFeatConvolutionPoolReLU_Ver(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Filter_DataSize,
			unsigned int Bias_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Filter_InL3,
			unsigned int Bias_InL3,
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FScW,
			unsigned int FScH,
			unsigned int ConvStrideW,
			unsigned int ConvStrideH,
			unsigned int ConvDilationW,
			unsigned int ConvDilationH,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			
			int          PoolDoPad,
			int          PoolDoReLU,
			int          DoPool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	unsigned int InType;
	if (DoPool==0) {
		FSpW=1; FSpH=1; PoolStrideW=1; PoolStrideH=1;
	}
	unsigned int TileCons = (ConvStrideW * FSpW) + FScW - ConvStrideW;
	int Overlap = FScW + ConvStrideW * (FSpW-PoolStrideW-1);
	int ConvOverlap = FScW - ConvStrideW;
	unsigned int Wo, Ho, Wc, Hc;
	int PadLc, PadRc, PadTc, PadBc, PadTp, PadBp,PadRp, PadLp;;
	v4s PadInp = (v4s){0,0,0,0};
	v4s PadInc = (v4s){0,0,0,0};
	char *ConvKerName, *PoolKerName, *ReLUKerName, *SetBiasKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (In_DataSize != Filter_DataSize || Filter_DataSize != Bias_DataSize || Bias_DataSize != Out_DataSize)
		GenTilingError("CNN_LargeParOutFeatConvolutionPoolReLU_Ver %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(1, DataSize, FScW,FScH, ConvStrideW, ConvStrideH, DoPool, FSpW, FSpH, PoolStrideW, PoolStrideH,
			  &ConvKerName, &PoolKerName, &ReLUKerName, &SetBiasKerName,
			  &NeedConvDim, &NeedConvStride, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (ConvDoPad) {
		PadTc = (FScH-1)/2; PadBc = FScH/2;
		PadLc = (FScW-1)/2; PadRc = FScW/2;
		PadInc = (v4s){(FScW-1)/2,FScW/2,(FScH-1)/2, FScH/2};
		Wc = (Width- FScW+((FScW-1)/2)+(FScW/2))/ConvStrideW + 1;
		Hc = (Height-FScH+((FScH-1)/2)+(FScH/2))/ConvStrideH + 1;
	} else {
		PadTc = 0; PadBc = 0; PadLc = 0; PadRc = 0;
		Wc = (Width- FScW)/ConvStrideW + 1;
		Hc = (Height-FScH)/ConvStrideH + 1;
	}
	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadRp = (FSpW-1)/2; PadLp = FSpW/2;
			PadInp = (v4s){0,0,PadTp,PadBp};
			Wo = (Wc-FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Hc-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			PadTp = 0; PadBp = 0, PadRp = 0; PadLp = 0;;
			Wo = (Wc-FSpW)/PoolStrideW + 1;
			Ho = (Hc-FSpH)/PoolStrideH + 1;
		}
	} else {
		PadTp = 0; PadBp = 0; PadRp = 0; PadLp = 0;
		Wo = Wc; Ho = Hc;
	}

	unsigned int K=1; //Min(InFeat, 4);

	if (InFeat==1) InType = OBJ_IN_DB; else InType = OBJ_IN_DB_3D;

	if (InFeat==1) InType = OBJ_IN_DB;
	else {
		if      ((InFeat % 32) == 0) K = 32;
		else if ((InFeat % 16) == 0) K = 16;
		else if ((InFeat % 8) == 0) K = 8;
		else if ((InFeat % 4) == 0) K = 4;
		else if ((InFeat % 2) == 0) K = 2;
	}
	while (1) {
		Kernel_T *U_Ker;
		if (InFeat==K) InType = OBJ_IN_DB; else InType = OBJ_IN_DB_3D;
			U_Ker = UserKernel(Name,
			KernelDimensionsAndUserSymbols(InFeat/K, Width, Height, OutFeat,
				KerDynamicSymbols(2,
					S_Dyn("Wc", Wc),
					S_Dyn("Hc", Hc)
				)
			),
			(InFeat==1)? KernelIterationOrder(KER_DIM3, KER_TILE, KER_TILE1):KernelIterationOrder(KER_DIM4, KER_TILE, KER_IN_PLANE, KER_TILE1),
					TILE_VER,
					CArgs(5,
						  TCArg(DataType,       "In"),
						  TCArg(DataType,       "Filter"),
						  TCArg(DataType,       "Bias"),
						  TCArg(DataType,       "Out"),
						  TCArg("unsigned int", "Norm")
						 ),
					Calls(3,
				Call(SetBiasKerName, (InFeat==1)?LOC_INNER_LOOP1_PROLOG:LOC_IN_PLANE_PROLOG,
					Bindings(5,
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
						UserSymb("Hc"),
						KerDim(K_OUTP),
						K_Arg("Bias", KER_ARG_TILE)
					)
				),
				Call(ConvKerName, LOC_INNER_LOOP1,
					Bindings(18,
						K_Arg("In", KER_ARG_TILE),
						K_Arg("In", KER_ARG_TILE_W),
						KerDim(K_H),
						Imm(K),                     // Number of input features in this tile
						K_Arg("Filter", KER_ARG_TILE_H),        // Number of output features in this tile
						K_ArgOper("In", KER_ARG_INPLANEINDEX, '*', K),  // Where to start in In Features, should be multiplied by K
						K_Arg("Filter", KER_ARG_TILE_BASE),         // From where to start Out Features in this tile
						Imm(InFeat),                    // Total number of input features
						K_Arg("Filter", KER_ARG_TILE),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
						K_Arg("In", KER_ARG_TILE_PAD),
						C_Arg("Norm"),
						NeedConvDim?Imm(FScW):AT_IGNORE_ARG_BINDING,
						NeedConvStride?Imm(ConvStrideW):AT_IGNORE_ARG_BINDING,
						Imm(1), //This will be dilation x
						NeedConvDim?Imm(FScH):AT_IGNORE_ARG_BINDING,
						NeedConvStride?Imm(ConvStrideH):AT_IGNORE_ARG_BINDING,
						Imm(1) //This will be dilation y
						)
				),
				ConvDoReLU?
				Call(ReLUKerName, (InFeat==1)?LOC_INNER_LOOP1_EPILOG:LOC_IN_PLANE_EPILOG,
					Bindings(12,
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
						UserSymb("Hc"),
						KerDim(K_OUTP),
						K_Arg("Out", KER_ARG_TILE),
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING)
				):(DoPool?
				Call(PoolKerName, (InFeat==1)?LOC_INNER_LOOP1_EPILOG:LOC_IN_PLANE_EPILOG,
					Bindings(12,
						K_Arg("ConvOut", KER_ARG_TILE),
						K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_W),
						UserSymb("Hc"),
						KerDim(K_OUTP),
						K_Arg("Out", KER_ARG_TILE),
						K_Arg("ConvOut", KER_ARG_TILE_PAD),
						NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,            
						NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,  						
						AT_IGNORE_ARG_BINDING,
						Imm(((DoPool-1)<<1)|PoolDoReLU),
						NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
						NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING
					)
				):AT_NO_CALL)
						 ),
					KerArgs(5,
				ConvDoPad?
				KerArgP("In",      InType|InL3,                 Width, Height*K, PadLc, PadRc, PadTc, PadBc, In_DataSize,             Overlap, OBJ_CONSTRAINTS_PAD_REM,  TileCons, "In", 0):
				KerArg ("In",      InType|InL3,                 Width, Height*K,                             In_DataSize,             Overlap, OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
				KerArg ("Filter",  OBJ_IN_DB|O_TILE1|FilterL3,  InFeat, OutFeat,                             FScW*FScH*Filter_DataSize, 0,       0,                        8,        "Filter", 0),
				KerArg ("Bias",    OBJ_BUFFER_IN_NTILED|BiasL3, 1,      OutFeat,                             Bias_DataSize,           0,       0,                        0,        "Bias", 0),
				KerArg ("Out",     OBJ_OUT_DB|OutL3,            Wo,  Ho*OutFeat,                             Out_DataSize,            0,       0,                        0,        "Out", 0),
				DoPool?(PoolDoPad?
				KerArgP("ConvOut",  OBJ_BUFFER_ONETILE,         Wc,  Hc*OutFeat, PadTp, PadBp, PadTp, PadBp, Out_DataSize,      ConvOverlap,       OBJ_CONSTRAINTS_PAD_REM,  0,        "", 0):
				KerArg ("ConvOut",  OBJ_BUFFER_ONETILE,         Wc,  Hc*OutFeat,                             Out_DataSize,      ConvOverlap,       OBJ_CONSTRAINTS_DROP_REM, 0,        "", 0)):
				AT_NO_KER_ARG
							)
			);
		if (U_Ker==0) {
			if (K!=1) K=K/2;
			else GenTilingError("Kernel: %s, not enough shared L1 memory to find a tiling solution", Name);
		} else break;
	}
	AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize},
			  (v4s) {InL3!=0,FilterL3!=0,BiasL3!=0,OutL3!=0},
			  InFeat, OutFeat, Width, Height,
			  FScW, FScH, ConvStrideW, ConvStrideH, PadInc, ConvDoReLU, 1,
			  FSpW, FSpH, PoolStrideW, PoolStrideH, PadInp, PoolDoReLU, DoPool,
			  0, 0, 0, (DataSize==1)?7:15, 0);
}




void CNN_SmallParOutFeatConvolutionPoolReLU(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Filter_DataSize,
			unsigned int Bias_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Filter_InL3,
			unsigned int Bias_InL3,
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FScW,
			unsigned int FScH,
			unsigned int ConvStrideW,
			unsigned int ConvStrideH,
			unsigned int ConvDilationW,
			unsigned int ConvDilationH,
			int          ConvDoPad,
			int          ConvDoReLU,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			
			int          PoolDoPad,
			int          PoolDoReLU,
			int          DoPool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;

	unsigned int InType;

	unsigned int TileCons = ConvStrideH * PoolStrideH;

	//FP: Overlap in horizontal tiling I suppose that needs the height of the filter: FScH
	int Overlap = FScH + ConvStrideH*(FSpH-PoolStrideH-1);
	unsigned int Wo, Ho, Wc, Hc;
	int PadTc, PadBc, PadLc, PadRc, PadTp, PadBp, PadLp, PadRp;

	v4s PadInp = (v4s){0,0,0,0};
	v4s PadInc = (v4s){0,0,0,0};
	char *ConvKerName, *PoolKerName, *ReLUKerName, *SetBiasKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (In_DataSize != Filter_DataSize || Filter_DataSize != Bias_DataSize || Bias_DataSize != Out_DataSize)

		GenTilingError("CNN_SmallParOutFeatConvolutionPoolReLU %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(1, DataSize, FScW,FScH, ConvStrideW, ConvStrideH, DoPool, FSpW, FSpH, PoolStrideW, PoolStrideH,
			  &ConvKerName, &PoolKerName, &ReLUKerName, &SetBiasKerName,
			  &NeedConvDim, &NeedConvStride, &NeedPoolDim, &NeedPoolStride);

	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";


	if (ConvDoPad) {
		PadTc = (FScH-1)/2; PadBc = FScH/2;
		PadLc = (FScW-1)/2; PadRc = FScW/2;
		PadInc = (v4s){(FScW-1)/2,FScW/2,(FScH-1)/2, FScH/2};
		Wc = (Width- FScW+((FScW-1)/2)+(FScW/2))/ConvStrideW + 1;
		Hc = (Height-FScH+((FScH-1)/2)+(FScH/2))/ConvStrideH + 1;
	} else {
		PadTc = 0; PadBc = 0; PadLc = 0; PadRc = 0;
		Wc = (Width- FScW)/ConvStrideW + 1;
		Hc = (Height-FScH)/ConvStrideH + 1;
	}
	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadRp = (FSpW-1)/2; PadLp = FSpW/2;
			PadInp = (v4s){0,0,PadTp,PadBp};
			Wo = (Wc-FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Hc-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			PadTp = 0; PadBp = 0, PadRp = 0; PadLp = 0;;
			Wo = (Wc-FSpW)/PoolStrideW + 1;
			Ho = (Hc-FSpH)/PoolStrideH + 1;
		}
	} else {
		PadTp = 0; PadBp = 0; PadRp = 0; PadLp = 0;
		Wo = Wc; Ho = Hc;
	}

		UserKernel(Name,
		KernelDimensionsAndUserSymbols(InFeat, Width, Height, OutFeat,
			KerDynamicSymbols(2,
				S_Dyn("Wc", Wc),
				S_Dyn("Hc", Hc)
			)
		),
				KernelIterationOrder(KER_DIM3, KER_TILE, KER_TILE1),
				TILE_HOR,
				CArgs(5,
					  TCArg(DataType,       "In"),
					  TCArg(DataType,       "Filter"),
					  TCArg(DataType,       "Bias"),
					  TCArg(DataType,       "Out"),
					  TCArg("unsigned int", "Norm")
					 ),
				Calls(3,
			Call(SetBiasKerName, LOC_INNER_LOOP1_PROLOG,
				Bindings(5,
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					UserSymb("Wc"),
					UserSymb("Hc"),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					K_Arg("Bias", KER_ARG_TILE)
				)
			),
			Call(ConvKerName, LOC_INNER_LOOP1,
				Bindings(18,
					K_Arg("In", KER_ARG_TILE),
					KerDim(K_W), // Imm(Width),
					KerDim(K_H), // Imm(Height),
					K_Arg("In", KER_ARG_TILE_H),            // Number of In features in this tile
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),  // Number of Out features in this slice
					K_Arg("In", KER_ARG_TILE_BASE),         // Starting point for In features
					Imm(0),                     // Starting point for Out Features
					KerDim(K_INP),                  // Imm(InFeat), Total number of input features
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					Imm((int)PadInc),
					C_Arg("Norm"),
					NeedConvDim?Imm(FScW):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideW):AT_IGNORE_ARG_BINDING,
					Imm(1), //This will be Dilation x
					NeedConvDim?Imm(FScH):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideH):AT_IGNORE_ARG_BINDING,
					Imm(1) //This will be Dilation y
				)
			),
			ConvDoReLU?
			Call(ReLUKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(12,
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE),
					UserSymb("Wc"),
					UserSymb("Hc"),
					K_Arg(DoPool?"ConvOut":"Out", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING
				)
			):(DoPool?
			Call(PoolKerName, LOC_INNER_LOOP1_EPILOG,
				Bindings(12,
					K_Arg("ConvOut", KER_ARG_TILE),
					UserSymb("Wc"),
					UserSymb("Hc"),
					K_Arg("ConvOut", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					Imm((int)PadInp),
					NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					Imm(((DoPool-1)<<1)|PoolDoReLU),
					NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING
				)
			):AT_NO_CALL)
					 ),
				KerArgs(5,
						KerArg("In",      OBJ_IN_DB|O_TILE1|InL3, Width*Height,  InFeat,  In_DataSize,             0, 0, 0, "In", 0),
						KerArg("Filter",  OBJ_IN_DB|FilterL3,     InFeat,        OutFeat, FScW*FScH*Filter_DataSize, 0, 0, 8, "Filter", 0),
						KerArg("Bias",    OBJ_BUFFER_IN|BiasL3,   1,             OutFeat, Bias_DataSize,           0, 0, 0, "Bias", 0),
						KerArg("Out",     OBJ_OUT_DB|OutL3,       Wo*Ho,         OutFeat, Out_DataSize,            0, 0, 0, "Out", 0),
		DoPool? KerArg("ConvOut", OBJ_BUFFER_ONETILE,     Wc*Hc,         OutFeat, Out_DataSize,            0, 0, 0, "", 0):AT_NO_KER_ARG
						)
		);

	AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize},
			  (v4s) {InL3!=0,FilterL3!=0,BiasL3!=0,OutL3!=0},
			  InFeat, OutFeat, Width, Height,
			  FScW, FScH, ConvStrideW, ConvStrideH, PadInc, ConvDoReLU, 1,
			  FSpW, FSpH, PoolStrideW, PoolStrideH, PadInp, PoolDoReLU, DoPool,
			  0, 0, 0, (DataSize==1)?7:15, 0);

}

/*********************************************************************************************************************************************************************
	Generators for DepthWise Convolutions, followed by an optional linear rectification (ReLU).
	Template:
		Name:       Name of the generated user kernel
		In_DataSize:    1: byte, 2: half word, 4: word
		Filter_DataSize:1: byte, 2: half word, 4: word
		Bias_DataSize:  1: byte, 2: half word, 4: word
		Out_DataSize:   1: byte, 2: half word, 4: word
		In_InL3:        0: In is in L2, 1: In is in L3 memory
		Filter_InL3:    0: Filter is in L2, 1: Filter is in L3 memory
		Bias_InL3:      0: Bias is in L2, 1: Bias is in L3 memory
		Out_InL3:       0: Out is in L2, 1: Out is in L3 memory
		InOutFeat:  Number of input and output feature's maps
		Width:      Number of columns of a given feature map
		Height:     Number of lines of a given feature map
		FSc:        Dimension of the convolution (FSc x FSc)
		ConvStride: Stride between 2 points in the input feature map where convolution is evaluated
		ConvDoPad:  0: No padding, 1: Zero padding
		ConvDoReLU: Performs linear rectification after all convolutions for a given output feature's map have been evaluated
	Currently only homegeneous data size are supported (bytes and hald words)
	CNN_LargeDepthWiseConvolutionReLU_Hor
		A given input feature map is tiled and the evaluation of the tile is dispatched on all cores for parallel evaluation. 
		Before starting the traversal of all input feature's maps bias is assigned to the output tile.
		Tile orientation is horizontal.
	CNN_LargeDepthWiseConvolutionReLU_Ver
		A given input feature map is tiled and the evaluation of the tile is dispatched on all cores for parallel evaluation. 
		Before starting the traversal of all input feature's maps bias is assigned to the output tile.
		Tile orientation is vertical.
*********************************************************************************************************************************************************************/


void CNN_LargeDepthWiseConvolutionReLU_Hor(
			char         *Name,
	
			unsigned int In_DataSize,
			unsigned int Filter_DataSize,
			unsigned int Bias_DataSize,
			unsigned int Out_DataSize,
	
			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Filter_InL3,
			unsigned int Bias_InL3,
			unsigned int Out_InL3,
	
			unsigned int InOutFeat,
			unsigned int Width,
			unsigned int Height,
	
			unsigned int FScW,
			unsigned int FScH,
			unsigned int ConvStrideW,
			unsigned int ConvStrideH,
			int          ConvDoPad,
			int          ConvDoReLU
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	int FSpW=1, PoolStrideW=1; int FSpH=1, PoolStrideH=1;
	
	unsigned int TileCons = ConvStrideH * PoolStrideH;
	int Overlap = FScH + ConvStrideH*(FSpH-PoolStrideH-1);
	unsigned int Wo, Ho, Wc, Hc;
	//  PadTc Pading Top    convolutions
	//  PadBc Pading Bottom convolutions
	//  PadLc Pading Left   convolutions
	//  PadRc Pading Right  convolutions
	//  PadTp Pading Top    Pooling
	//  PadBp Pading Bottom Pooling
	int PadTc, PadBc, PadRc, PadLc,PadTp, PadBp;

	//TODO: Variables for test, needs to be updated
	v4s PadInc = (v4s){0,0,0,0};
	v4s PadInp = (v4s){0,0,0,0};

	char *ConvKerName, *PoolKerName, *ReLUKerName, *SetBiasKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (In_DataSize != Filter_DataSize || Filter_DataSize != Bias_DataSize || Bias_DataSize != Out_DataSize)
		GenTilingError("CNN_LargeConvolutionPoolReLU_Hor %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(0, DataSize, FScW,FScH, ConvStrideW, ConvStrideH, 0, FSpW,FSpH,PoolStrideW,PoolStrideH,
			  &ConvKerName, &PoolKerName, &ReLUKerName, &SetBiasKerName,
			  &NeedConvDim, &NeedConvStride, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (ConvDoPad)
	{
		PadTc = (FScH-1)/2; PadBc = FScH/2;
		PadLc = (FScW-1)/2; PadRc = FScW/2;
		
		//TODO PadInc used for tests need to be updated with FScW FScH 
		PadInc = (v4s){PadLc,PadRc,PadTc, PadBc};
		
		//Output Width and Height
		Wc = (Width  - FScW+((FScW-1)/2)+(FScW/2))/ConvStrideW + 1;
		Hc = (Height - FScH+((FScH-1)/2)+(FScH/2))/ConvStrideH + 1;
	} 
	else 
	{
		PadTc = 0; PadBc = 0; PadLc = 0; PadRc = 0;
		Wc = (Width- FScW)/ConvStrideW + 1;
		Hc = (Height-FScH)/ConvStrideH + 1;
	}

	Wo = Wc; Ho = Hc;
	
	UserKernel(Name,
		KernelDimensions(InOutFeat, Width, Height, InOutFeat),
		KernelIterationOrder(KER_DIM3, KER_IN_OUT_PLANE, KER_TILE),
		TILE_HOR,
		CArgs(5,
			TCArg(DataType,       "In"),
			TCArg(DataType,       "Filter"),
			TCArg(DataType,       "Bias"),
			TCArg(DataType,       "Out"),
			TCArg("unsigned int", "Norm")
		),
		Calls(3,
			Call(SetBiasKerName, LOC_INNER_LOOP,
				Bindings(4, 
					K_Arg("Out", KER_ARG_TILE), 
					K_Arg("Out", KER_ARG_TILE_W),
					K_Arg("Out", KER_ARG_TILE_H), 
					C_ArgIndirectIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(14,
					K_Arg("In", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_W),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg("Out", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_PAD),
					C_Arg("Norm"),
					NeedConvDim?Imm(FScW):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideW):AT_IGNORE_ARG_BINDING,
					Imm(1), //Orientation
					Imm(1), //Dilation x
					NeedConvDim?Imm(FScH):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideH):AT_IGNORE_ARG_BINDING,
					Imm(1) //Dilation y
					)
			),
			ConvDoReLU?Call(ReLUKerName, LOC_INNER_LOOP,
					Bindings(13,
						K_Arg("Out", KER_ARG_TILE),
						K_Arg("Out", KER_ARG_TILE_W),
						K_Arg("Out", KER_ARG_TILE_H),
						K_Arg("Out", KER_ARG_TILE),
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING
					)
			):AT_NO_CALL
		),
		KerArgs(3,
			ConvDoPad?
			KerArgP("In",      OBJ_IN_DB_3D|InL3,                Width, Height, PadLc,PadRc,PadTc,PadBc, In_DataSize, Overlap, OBJ_CONSTRAINTS_PAD_REM,  0, "In", 0):
			KerArg ("In",      OBJ_IN_DB_3D|InL3,                Width, Height,                          In_DataSize, Overlap, OBJ_CONSTRAINTS_DROP_REM, 0, "In", 0),
			KerArg ("Filter",  OBJ_BUFFER_IN_NTILED_3D|FilterL3,   FScW,    FScH,                      Filter_DataSize,       0,                        0, 0, "Filter", 0),
			KerArg ("Out",     OBJ_OUT_DB_3D|OutL3,                 Wo,     Ho,                         Out_DataSize,       0,                        0, 0, "Out", 0)
		)
	);

}


void CNN_LargeDepthWiseConvolutionReLU_Ver(
			char         *Name,
	
			unsigned int In_DataSize,
			unsigned int Filter_DataSize,
			unsigned int Bias_DataSize,
			unsigned int Out_DataSize,
	
			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Filter_InL3,
			unsigned int Bias_InL3,
			unsigned int Out_InL3,
	
			unsigned int InOutFeat,
			unsigned int Width,
			unsigned int Height,
	
			unsigned int FScW,
			unsigned int FScH,
			unsigned int ConvStrideW,
			unsigned int ConvStrideH,
			int          ConvDoPad,
			int          ConvDoReLU
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	//Set Pooling to 1 since it is not supported in depthwise
	int FSpW=1, PoolStrideW=1; int FSpH=1, PoolStrideH=1;
	
	unsigned int TileCons = ConvStrideW * PoolStrideW;
	int Overlap = FScW + ConvStrideW*(FSpW-PoolStrideW-1);
	unsigned int Wo, Ho, Wc, Hc;
	//  PadTc Pading Top    convolutions
	//  PadBc Pading Bottom convolutions
	//  PadLc Pading Left   convolutions
	//  PadRc Pading Right  convolutions
	//  PadTp Pading Top    Pooling
	//  PadBp Pading Bottom Pooling
	int PadTc, PadBc, PadRc, PadLc;

	//TODO: Variables for test, needs to be updated
	v4s PadInc = (v4s){0,0,0,0};
	v4s PadInp = (v4s){0,0,0,0};

	char *ConvKerName, *PoolKerName, *ReLUKerName, *SetBiasKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (In_DataSize != Filter_DataSize || Filter_DataSize != Bias_DataSize || Bias_DataSize != Out_DataSize)
		GenTilingError("CNN_LargeConvolutionPoolReLU_Ver %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(0, DataSize, FScW,FScH, ConvStrideW,ConvStrideH, 0, FSpW,FSpH,PoolStrideW,PoolStrideH,
			  &ConvKerName, &PoolKerName, &ReLUKerName, &SetBiasKerName,
			  &NeedConvDim, &NeedConvStride, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (ConvDoPad)
	{
		PadTc = (FScH-1)/2; PadBc = FScH/2;
		PadLc = (FScW-1)/2; PadRc = FScW/2;
		
		//TODO PadInc used for tests need to be updated with FScW FScH 
		PadInc = (v4s){PadLc,PadRc,PadTc,PadBc};
		
		//Output Width and Height
		Wc = (Width  - FScW+((FScW-1)/2)+(FScW/2))/ConvStrideW + 1;
		Hc = (Height - FScH+((FScH-1)/2)+(FScH/2))/ConvStrideH + 1;
	} 
	else 
	{
		PadTc = 0; PadBc = 0, PadLc = 0; PadRc = 0; 
		Wc = (Width- FScW)/ConvStrideW + 1;
		Hc = (Height-FScH)/ConvStrideH + 1;
	}

	Wo = Wc; Ho = Hc;

	UserKernel(Name,
		KernelDimensions(InOutFeat, Width, Height, InOutFeat),
		KernelIterationOrder(KER_DIM3, KER_IN_OUT_PLANE, KER_TILE),
		TILE_VER,
		CArgs(5,
			TCArg(DataType,       "In"),
			TCArg(DataType,       "Filter"),
			TCArg(DataType,       "Bias"),
			TCArg(DataType,       "Out"),
			TCArg("unsigned int", "Norm")
		),
		Calls(3,
			Call(SetBiasKerName, LOC_INNER_LOOP,
				Bindings(4, 
					K_Arg("Out", KER_ARG_TILE), 
					K_Arg("Out", KER_ARG_TILE_W),
					K_Arg("Out", KER_ARG_TILE_H), 
					C_ArgIndirectIndex("Bias", KER_OUT_PLANE, 1))),
			Call(ConvKerName, LOC_INNER_LOOP,
				Bindings(14,
					K_Arg("In", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_W),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Filter", KER_ARG_TILE),
					K_Arg("Out", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_PAD),
					C_Arg("Norm"),
					NeedConvDim?Imm(FScW):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideW):AT_IGNORE_ARG_BINDING,
					Imm(0), //Orientation
					Imm(1), //Dilation x
					NeedConvDim?Imm(FScH):AT_IGNORE_ARG_BINDING,
					NeedConvStride?Imm(ConvStrideH):AT_IGNORE_ARG_BINDING,
					Imm(1) //Dilation y
					)
			),
			ConvDoReLU?Call(ReLUKerName, LOC_INNER_LOOP,
					Bindings(13,
						K_Arg("Out", KER_ARG_TILE),
						K_Arg("Out", KER_ARG_TILE_W),
						K_Arg("Out", KER_ARG_TILE_H),
						K_Arg("Out", KER_ARG_TILE),
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING,
						AT_IGNORE_ARG_BINDING, 
						AT_IGNORE_ARG_BINDING
					)
			):AT_NO_CALL
		),
		KerArgs(3,
			ConvDoPad?
			KerArgP("In",      OBJ_IN_DB_3D|InL3,                Width, Height, PadLc,PadRc,PadTc,PadBc, In_DataSize, Overlap, OBJ_CONSTRAINTS_PAD_REM,  0, "In", 0):
			KerArg ("In",      OBJ_IN_DB_3D|InL3,                Width, Height,                          In_DataSize, Overlap, OBJ_CONSTRAINTS_DROP_REM, 0, "In", 0),
			KerArg ("Filter",  OBJ_BUFFER_IN_NTILED_3D|FilterL3,   FScW,    FScH,                      Filter_DataSize,       0,                        0, 0, "Filter", 0),
			KerArg ("Out",     OBJ_OUT_DB_3D|OutL3,                 Wo,     Ho,                         Out_DataSize,       0,                        0, 0, "Out", 0)
		)
	);

}

/*********************************************************************************************************************************************************************
	Generators for Pooling (Max or Average) followed by an optional linear rectification (ReLU) or linear rectification only

	Template:
		Name:       Name of the generated user kernel

		In_DataSize:    1: byte, 2: half word, 4: word
		Out_DataSize:   1: byte, 2: half word, 4: word

		In_InL3:    0: In is in L2, 1: In is in L3 memory
		Out_InL3:   0: Out is in L2, 1: Out is in L3 memory

		InFeat:     Number of input feature's maps
		OutFeat:    Number of output feature's maps (InFeat has to be equal to OutFeat for these generators
		Width:      Number of columns of a given feature map
		Height:     Number of lines of a given feature map

		FSp:        Size of the pooling region (FSp x FSp)
		PoolStride: Stride between 2 points in the input feature map where pooling is evaluated
		PoolDoPad:  0: No padding, 1: Zero padding
		PoolDoReLU: 0: No linear rectification, 1: Linear rectification. If DoPool=1 ReLU is performed after pooling otherwise only ReLU is performed.
		DoPool:     0: No pooling, 1: Max pooling, 2: Average pooling. If DoPool=0 then PoolDoReLU must be 1

	Currently only homegeneous data size are supported (bytes and hald words)

	CNN_LargePoolReLU
		Input feature's maps are evaluated one after the other. A given feature map is tiled and the evaluation of a tile is dispatched on all cores for
		parallel evaluation. Tiling is horizontal.

	CNN_LargeParOutFeatPoolReLU
		Input feature's maps are processed as a group, a tile crosses all input feature maps and then each core is given a subset of input feature's maps.
		Tiling is horizontal.

	CNN_SmallParOutFeatPoolReLU:
		Input feature's maps are processed as a group but it is assumed than an entire feature maps can fit into cluster's L1 memory. A tile is a group of
		full input feature's maps, the size of the group is constrained to be a multiple of number of cores if possible.

*********************************************************************************************************************************************************************/

void CNN_LargePoolReLU(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			int          PoolDoPad,
			int          PoolDoReLU,

			int      DoPool     // 0: NoPool, 1: Max Pool, 2: Average Pool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	if (DoPool==0) {
		FSpW=1; FSpH=1; PoolStrideW=1; PoolStrideH=1;
	}
	
	unsigned int TileCons = PoolStrideH;
	int Overlap = 1 + 1*(FSpH-PoolStrideH-1);
	unsigned int Wo, Ho;
	int PadTp, PadBp, PadLp, PadRp;;
	v4s PadInp = (v4s){0,0,0,0};
	char *PoolKerName, *ReLUKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int DataSize = In_DataSize;

	if (InFeat!=OutFeat) GenTilingError("LargePoolReLU %s: InFeat != OutFeat (%d, %d)\n", Name, InFeat, OutFeat);
	if (DoPool==0 && PoolDoReLU==0) GenTilingError("LargePoolReLU %s: At least DoPool or PoolDoReLU must be non null", Name);
	if (In_DataSize != Out_DataSize) GenTilingError("CNN_LargePoolReLU %s: Only homogeneous data types are supported (byte and half word)\n", Name);


	SelectKernels(0, DataSize, 1, 1, 1,1, DoPool, FSpW,FSpH, PoolStrideW,PoolStrideH, 0, &PoolKerName, &ReLUKerName, 0, 0, 0, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadLp = (FSpW-1)/2; PadRp = FSpW/2;
			PadInp = (v4s){PadTp,PadBp,PadTp,PadBp};
			Wo = (Width-FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Height-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			Wo = (Width-FSpW)/PoolStrideW + 1;
			Ho = (Height-FSpH)/PoolStrideH + 1;
		}
	} else {
		Wo = Width; Ho = Height;
	}

		UserKernel(Name,
		KernelDimensions(InFeat, Width, Height, OutFeat),
		KernelIterationOrder(KER_DIM3, KER_IN_OUT_PLANE, KER_TILE),
				TILE_HOR,
				CArgs(2,
					  TCArg(DataType, "In"),
					  TCArg(DataType, "Out")
					 ),
				Calls(1,
			DoPool?
			Call(PoolKerName, LOC_INNER_LOOP,
				Bindings(13,
					K_Arg("In", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_W),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_PAD),
					NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,
					Imm(1),  //Orientation
					Imm(((DoPool-1)<<1)|PoolDoReLU),
					Imm(1), //Dilation x
					NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING,
					Imm(1) //Dilation y
				)
			):
			Call(ReLUKerName, LOC_INNER_LOOP,
				Bindings(13,
					K_Arg("In", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_W),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING
				)
			)
					 ),
				KerArgs(2,
			(DoPool&&PoolDoPad)?
			KerArgP("In",      OBJ_IN_DB_3D|InL3,            Width, Height, PadLp,PadRp,PadTp,PadBp, In_DataSize,  Overlap, OBJ_CONSTRAINTS_PAD_REM,  TileCons, "In", 0):
			KerArg ("In",      OBJ_IN_DB_3D|InL3,            Width, Height,                          In_DataSize,  Overlap, OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
			KerArg ("Out",     OBJ_OUT_DB_3D|OutL3,          Wo,    Ho,                              Out_DataSize,       0,                        0,        0, "Out", 0)
		)
		);

	AT_PrepareForTest(Name,
			(v4s) {In_DataSize, 0, 0, Out_DataSize},
			(v4s) {InL3!=0,0,0,OutL3!=0},
			InFeat, OutFeat, Width, Height,
			0, 0, 0, 0, (v4s) 0, 0, 0,
			FSpW, FSpH, PoolStrideW, PoolStrideH, PadInp, PoolDoReLU, DoPool,
			0, (DoPool==0)&&(PoolDoReLU), 0, (DataSize==1)?7:15, 0);
}




void CNN_LargeParOutFeatPoolReLU(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			int          PoolDoPad,
			int          PoolDoReLU,

			int      DoPool     // 0: NoPool, 1: Max Pool, 2: Average Pool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	if (DoPool==0) {
		FSpW=1; PoolStrideW=1; FSpH=1; PoolStrideH=1;
	}
	unsigned int TileCons = PoolStrideH;
	int Overlap = 1 + 1*(FSpH-PoolStrideH-1);
	unsigned int Wo, Ho;
	int PadTp, PadBp, PadLp, PadRp;
	v4s PadInp = (v4s){0,0,0,0};
	char *PoolKerName, *ReLUKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (PoolStrideW>FSpW) GenTilingError("LargeParOutFeatPoolReLU_fp %s: PoolStrideW > pooling region (FSpW) (%d, %d)\n", Name, PoolStrideW, FSpW);
	if (PoolStrideH>FSpH) GenTilingError("LargeParOutFeatPoolReLU_fp %s: PoolStrideH > pooling region (FSpH) (%d, %d)\n", Name, PoolStrideH, FSpH);
	if (InFeat!=OutFeat) GenTilingError("LargeParOutFeatPoolReLU_fp %s: InFeat != OutFeat (%d, %d)\n", Name, InFeat, OutFeat);
	if (DoPool==0 && PoolDoReLU==0) GenTilingError("LargeParOutFeatPoolReLU_fp %s: At least DoPool or PoolDoReLU must be non null", Name);
	if (In_DataSize != Out_DataSize) GenTilingError("CNN_LargePoolReLU %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(1, DataSize, 1, 1, 1, 1, DoPool, FSpW,FSpH, PoolStrideW,PoolStrideH, 0, &PoolKerName, &ReLUKerName, 0, 0, 0, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadLp = (FSpW-1)/2; PadRp = FSpW/2;
			PadInp = (v4s){PadTp,PadBp,PadTp,PadBp};
			Wo = (Width- FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Height-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			Wo = (Width-FSpW)/PoolStrideW + 1;
			Ho = (Height-FSpH)/PoolStrideH + 1;
		}
	} else {
		Wo = Width; Ho = Height;
	}

		UserKernel(Name,
		KernelDimensions(InFeat, Width, Height, OutFeat),
		KernelIterationOrder(KER_DIM2, KER_TILE),
				TILE_HOR,
				CArgs(2,
					  TCArg(DataType, "In"),
					  TCArg(DataType, "Out")
					 ),
				Calls(1,
			DoPool?
			Call(PoolKerName, LOC_INNER_LOOP,
				Bindings(12,
					K_Arg("In", KER_ARG_TILE),
					KerDim(K_W),
					K_Arg("In", KER_ARG_TILE_H),
					KerDim(K_OUTP),
					K_Arg("Out", KER_ARG_TILE),
					K_Arg("In", KER_ARG_TILE_PAD),
					NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					Imm(((DoPool-1)<<1)|PoolDoReLU),
					NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING
				)
			):
			Call(ReLUKerName, LOC_INNER_LOOP,
				Bindings(12,
					K_Arg("In", KER_ARG_TILE),
					KerDim(K_W),
					K_Arg("In", KER_ARG_TILE_H),
					KerDim(K_OUTP),
					K_Arg("Out", KER_ARG_TILE),
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING
					)
				)
				),
		KerArgs(2,
			(DoPool&&PoolDoPad)?
			KerArg2DP("In",  OBJ_IN_DB|InL3,         Width*InFeat,  Height, Width, PadLp, PadRp, PadTp, PadBp, In_DataSize,  Overlap, OBJ_CONSTRAINTS_PAD_REM,  TileCons, "In", 0):
			KerArg2D ("In",  OBJ_IN_DB|InL3,         Width*InFeat,  Height, Width,                             In_DataSize,  Overlap, OBJ_CONSTRAINTS_DROP_REM, TileCons, "In", 0),
			KerArg2D ("Out", OBJ_OUT_DB|OutL3,         Wo*OutFeat,  Ho,     Wo,                                Out_DataSize, 0,       0,                        0,        "Out", 0)
			)
		);

	AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, 0, 0, Out_DataSize},
			  (v4s) {InL3!=0,0,0,OutL3!=0},
			  InFeat, OutFeat, Width, Height,
			  0, 0, 0, 0, (v4s) 0, 0, 0,
			  FSpW, FSpH, PoolStrideW, PoolStrideH, PadInp, PoolDoReLU, DoPool,
			  0, (DoPool==0)&&(PoolDoReLU), 0, (DataSize==1)?7:15, 0);

}

void CNN_SmallParOutFeatPoolReLU(
			char         *Name,

			unsigned int In_DataSize,
			unsigned int Out_DataSize,

			unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
			unsigned int Out_InL3,

			unsigned int InFeat,
			unsigned int OutFeat,
			unsigned int Width,
			unsigned int Height,

			unsigned int FSpW,
			unsigned int FSpH,
			unsigned int PoolStrideW,
			unsigned int PoolStrideH,
			int          PoolDoPad,
			int          PoolDoReLU,

			int      DoPool     // 0: NoPool, 1: Max Pool, 2: Average Pool
			)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	if (DoPool==0) {
		FSpW=1; PoolStrideW=1; FSpH=1; PoolStrideH=1;
	}
	unsigned int TileCons = PoolStrideH;
	int Overlap = 1 + 1*(FSpH-PoolStrideH-1);
	unsigned int Wo, Ho;
	int PadTp, PadBp, PadLp, PadRp;
	v4s PadInp = (v4s){0,0,0,0};
	char *PoolKerName, *ReLUKerName, *DataType;
	unsigned int NeedConvDim, NeedConvStride, NeedPoolDim, NeedPoolStride;
	int          DataSize = In_DataSize;

	if (InFeat!=OutFeat) GenTilingError("SmallParOutFeatPoolReLU %s: InFeat != OutFeat (%d, %d)\n", Name, InFeat, OutFeat);
	if (DoPool==0 && PoolDoReLU==0) GenTilingError("SmallParOutFeatPoolReLU %s: At least DoPool or PoolDoReLU must be non null", Name);
	if (In_DataSize != Out_DataSize) GenTilingError("CNN_LargePoolReLU %s: Only homogeneous data types are supported (byte and half word)\n", Name);

	SelectKernels(1, DataSize, 1, 1, 1, 1, DoPool, FSpW,FSpH, PoolStrideW, PoolStrideH, 0, &PoolKerName, &ReLUKerName, 0, 0, 0, &NeedPoolDim, &NeedPoolStride);
	if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

	if (DoPool) {
		if (PoolDoPad) {
			PadTp = (FSpH-1)/2; PadBp = FSpH/2;
			PadLp = (FSpW-1)/2; PadRp = FSpW/2;
			PadInp = (v4s){PadLp,PadRp,PadTp,PadBp};
			Wo = (Width- FSpW+((FSpW-1)/2)+(FSpW/2))/PoolStrideW + 1;
			Ho = (Height-FSpH+((FSpH-1)/2)+(FSpH/2))/PoolStrideH + 1;
		} else {
			Wo = (Width-FSpW)/PoolStrideW + 1;
			Ho = (Height-FSpH)/PoolStrideH + 1;
		}
	} else {
		Wo = Width; Ho = Height;
	}

		UserKernel(Name,
		KernelDimensions(InFeat, Width, Height, OutFeat),
				KernelIterationOrder(KER_DIM2, KER_TILE),
				TILE_HOR,
				CArgs(2,
					  TCArg(DataType, "In"),
					  TCArg(DataType, "Out")
					 ),
				Calls(1,
			DoPool?
			Call(PoolKerName, LOC_INNER_LOOP,
				Bindings(12,
					K_Arg("In", KER_ARG_TILE),
					KerDim(K_W), // Imm(Width),
					KerDim(K_H), // Imm(Height),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					Imm((int)PadInp),
					NeedPoolDim?Imm(FSpW):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideW):AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING,
					Imm(((DoPool-1)<<1)|PoolDoReLU),
					NeedPoolDim?Imm(FSpH):AT_IGNORE_ARG_BINDING,
					NeedPoolStride?Imm(PoolStrideH):AT_IGNORE_ARG_BINDING
				)
			):
			Call(ReLUKerName, LOC_INNER_LOOP,
				Bindings(12,
					K_Arg("In", KER_ARG_TILE),
					KerDim(K_W), // Imm(Width),
					KerDim(K_H), // Imm(Height),
					K_Arg("In", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING,
					AT_IGNORE_ARG_BINDING, 
					AT_IGNORE_ARG_BINDING
				)
			)
					 ),
				KerArgs(2,
					KerArg("In",      OBJ_IN_DB|InL3,   Width*Height,  InFeat,  In_DataSize,  0, 0, 8, "In", 0),
					KerArg("Out",     OBJ_OUT_DB|OutL3, Wo*Ho,         OutFeat, Out_DataSize, 0, 0, 0, "Out", 0)
				)
		);

		AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, 0, 0, Out_DataSize},
			  (v4s) {InL3!=0,0,0,OutL3!=0},
			  InFeat, OutFeat, Width, Height,
			  0, 0, 0, 0, (v4s) 0, 0, 0,
			  FSpW, FSpH, PoolStrideW, PoolStrideH, PadInp, PoolDoReLU, DoPool,
			  0, (DoPool==0)&&(PoolDoReLU), 0, (DataSize==1)?7:15, 0);

}


/*********************************************************************************************************************************************************************
    Generators for Addition followed by an optional linear rectification (ReLU)

    Template:
        Name:       Name of the generated user kernel

        In_DataSize:    1: byte, 2: half word, 4: word
        Out_DataSize:   1: byte, 2: half word, 4: word

        In_InL3:    0: In is in L2, 1: In is in L3 memory
        Out_InL3:   0: Out is in L2, 1: Out is in L3 memory

        InOutFeat:      Number of input and output feature's maps (for add must be equal)
        Width:      Number of columns of a given feature map
        Height:     Number of lines of a given feature map
        
        DoReLU: 0: No linear rectification, 1: Linear rectification. 

    Currently only homegeneous data size are supported (bytes and hald words)

    CNN_LargeAddReLU
        Input feature's maps are evaluated one after the other. A given feature map is tiled and the evaluation of a tile is dispatched on all cores for
        parallel evaluation. Tiling is horizontal.
    CNN_LargeParOutFeatAddReLU
        Input feature's maps are processed as a group, a tile crosses all input feature maps and then each core is given a subset of input feature's maps.
        Tiling is horizontal.
    CNN_SmallParOutFeatPoolReLU:
        Input feature's maps are processed as a group but it is assumed than an entire feature maps can fit into cluster's L1 memory. A tile is a group of
        full input feature's maps, the size of the group is constrained to be a multiple of number of cores if possible.



*********************************************************************************************************************************************************************/

void CNN_LargeAddReLU(
            char         *Name,
            unsigned int In_DataSize,
            unsigned int Out_DataSize,

            unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
            unsigned int Out_InL3,

            unsigned int InOutFeat,
            unsigned int Width,
            unsigned int Height,

            int          DoReLU
            )
{

    unsigned int InL3  = In_InL3?O_L2DB:0;
    unsigned int OutL3 = Out_InL3?O_L2DB:0;
    
    char  *DataType, *AddKerName;
    int   DataSize = In_DataSize;

    if (In_DataSize != Out_DataSize) GenTilingError("CNN_LargePoolReLU %s: Only homogeneous data types are supported (byte and half word)\n", Name);

    if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";

    if (In_DataSize==2 && Out_DataSize==2 && DoReLU==0) {
        AddKerName = "KerAdd_fp";
        DataType = "short int * __restrict__";   DataType = "short int * __restrict__";
    }
    else if (In_DataSize==2 && Out_DataSize==2 && DoReLU==1) {
        AddKerName = "KerAddReLU_fp";
        DataType = "short int * __restrict__";   DataType = "short int * __restrict__";
    }
    else GenTilingError("CNN_LinearLayerReLU: %s Unsupported data size: In:%d, Out: %d\n", Name, In_DataSize, Out_DataSize);

        UserKernel(Name,
                KernelDimensions(InOutFeat, Width, Height, InOutFeat),
        KernelIterationOrder(KER_DIM3, KER_IN_OUT_PLANE, KER_TILE),
                TILE_HOR,
                CArgs(3,
                      TCArg(DataType, "In1"),
                      TCArg(DataType, "In2"),
                      TCArg(DataType, "Out")
                     ),
                Calls(1,
                    Call(AddKerName, LOC_INNER_LOOP,
                        Bindings(5,
                            K_Arg("In1", KER_ARG_TILE),
                            K_Arg("In2", KER_ARG_TILE),
                            K_Arg("In1", KER_ARG_TILE_W),
                            K_Arg("In1", KER_ARG_TILE_H),
                            K_Arg("Out", KER_ARG_TILE)
                )
            )
            ),
            KerArgs(3,
            KerArg ("In1", OBJ_IN_DB_3D  | InL3,   Width, Height,  In_DataSize,  0, 0, 8, "In1", 0),
            KerArg ("In2", OBJ_IN_DB_3D  | InL3,   Width, Height,  In_DataSize,  0, 0, 0, "In2", 0),
            KerArg ("Out", OBJ_OUT_DB_3D | OutL3,  Width, Height, Out_DataSize,  0, 0, 0, "Out", 0)
            )
        );
}


void CNN_LargeParOutFeatAddReLU(
            char         *Name,
            unsigned int In_DataSize,
            unsigned int Out_DataSize,

            unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
            unsigned int Out_InL3,

            unsigned int InOutFeat,
            unsigned int Width,
            unsigned int Height,

            int          DoReLU
            )

{
    unsigned int InL3 = In_InL3?O_L2DB:0;
    unsigned int OutL3 = Out_InL3?O_L2DB:0;

    char  *DataType, *AddKerName;
    int    DataSize = In_DataSize;

    if (DataSize==1) DataType = "signed char * __restrict__"; else DataType = "short int * __restrict__";
    if (In_DataSize==2 && Out_DataSize==2 && DoReLU==0) {
        AddKerName = "KerAdd_fp";
        DataType = "short int * __restrict__";   DataType = "short int * __restrict__";
    }
    else if (In_DataSize==2 && Out_DataSize==2 && DoReLU==1) {
        AddKerName = "KerAddReLU_fp";
        DataType = "short int * __restrict__";   DataType = "short int * __restrict__";
    }
    else GenTilingError("CNN_LinearLayerReLU: %s Unsupported data size: In:%d, Out: %d\n", Name, In_DataSize, Out_DataSize);

    if (In_DataSize != Out_DataSize) GenTilingError("CNN_LargePoolReLU %s: Only homogeneous data types are supported (byte and half word)\n", Name);

        UserKernel(Name,
        KernelDimensions(InOutFeat, Width, Height, InOutFeat),
        KernelIterationOrder(KER_DIM2, KER_TILE),
                TILE_HOR,
                CArgs(3,
                      TCArg(DataType, "In1"),
                      TCArg(DataType, "In2"),
                      TCArg(DataType, "Out")
                     ),
                Calls(1,
                Call(AddKerName, LOC_INNER_LOOP,
                        Bindings(5,
                            K_Arg("In1", KER_ARG_TILE),
                            K_Arg("In2", KER_ARG_TILE),
                            K_Arg("In1", KER_ARG_TILE_W),
                            K_Arg("In1", KER_ARG_TILE_H),
                            K_Arg("Out", KER_ARG_TILE)
                )
            )
                     ),
            KerArgs(3,
                KerArg2D ("In1", OBJ_IN_DB |InL3,   Width*InOutFeat,  Height, Width,    In_DataSize,  0,0,0, "In1", 0),
                KerArg2D ("In2", OBJ_IN_DB |InL3,   Width*InOutFeat,  Height, Width,    In_DataSize,  0,0,0, "In2", 0),
                KerArg2D ("Out", OBJ_OUT_DB|OutL3,  Width*InOutFeat,  Height, Width,    Out_DataSize, 0,0,0, "Out", 0)
            )
        );

}

void CNN_SmallParOutFeatAddReLU(
            char         *Name,
            unsigned int In_DataSize,
            unsigned int Out_DataSize,

            unsigned int In_InL3,       // 1 if In comes from L3, 0 if it comes from L2
            unsigned int Out_InL3,

            unsigned int InOutFeat,
            unsigned int Width,
            unsigned int Height,

            int          DoReLU
            )

{
    unsigned int InL3 = In_InL3 ? O_L2DB : 0;
    unsigned int OutL3 = Out_InL3 ? O_L2DB : 0;

    char  *DataType, *AddKerName;
    int    DataSize = In_DataSize;

    if (DataSize == 1) DataType = "signed char * __restrict__";
    else DataType = "short int * __restrict__";
    if (In_DataSize == 2 && Out_DataSize == 2 && DoReLU == 0)
    {
        AddKerName = "KerAdd_fp";
        DataType = "short int * __restrict__";
        DataType = "short int * __restrict__";
    }
    else if (In_DataSize == 2 && Out_DataSize == 2 && DoReLU == 1)
    {
        AddKerName = "KerAddReLU_fp";
        DataType = "short int * __restrict__";
        DataType = "short int * __restrict__";
    }
    else GenTilingError("CNN_LinearLayerReLU: %s Unsupported data size: In:%d, Out: %d\n", Name, In_DataSize, Out_DataSize);

    if (In_DataSize != Out_DataSize) GenTilingError("CNN_LargePoolReLU %s: Only homogeneous data types are supported (byte and half word)\n", Name);



    UserKernel(Name,
               KernelDimensions(InOutFeat, Width, Height, InOutFeat),
               KernelIterationOrder(KER_DIM2, KER_TILE),
               TILE_HOR,
               CArgs(3,
                     TCArg(DataType, "In1"),
                     TCArg(DataType, "In2"),
                     TCArg(DataType, "Out")
                    ),
               Calls(1,
                    Call(AddKerName, LOC_INNER_LOOP,
                        Bindings(5,
                        	K_Arg("In1", KER_ARG_TILE),
                        	K_Arg("In2", KER_ARG_TILE),
                        	K_Arg("In1", KER_ARG_TILE_W),
                        	K_Arg("In1", KER_ARG_TILE_H),
                        	K_Arg("Out", KER_ARG_TILE)
                        	)
                        )
                    ),
	KerArgs(3,
	    KerArg("In1", OBJ_IN_DB | InL3,  Width * Height, InOutFeat, In_DataSize,  0, 0, 8, "In1", 0),
	    KerArg("In2", OBJ_IN_DB | InL3,  Width * Height, InOutFeat, In_DataSize,  0, 0, 0, "In2", 0),
	    KerArg("Out", OBJ_OUT_DB | OutL3, Width * Height, InOutFeat, Out_DataSize, 0, 0, 0, "Out", 0)
	    )
	);

}




/*********************************************************************************************************************************************************************
	Generators for Linear layers followed by an optional linear rectification (ReLU)

	Template:
		Name:       Name of the generated user kernel

		In_DataSize:    1: byte, 2: half word, 4: word
		Filter_DataSize:1: byte, 2: half word, 4: word
		Bias_DataSize:  1: byte, 2: half word, 4: word
		Out_DataSize:   1: byte, 2: half word, 4: word

		In_InL3:    0: In is in L2, 1: In is in L3 memory
		Filter_InL3:    0: Filter is in L2, 1: Filter is in L3 memory
		Bias_InL3:  0: Bias is in L2, 1: Bias is in L3 memory
		Out_InL3:   0: Out is in L2, 1: Out is in L3 memory

		InDim:      Number of inputs
		OutDim:     Number of outputs

		DoReLU:     0: No linear rectification after linear layer, 1: Linear rectification applied after linear layer

	CNN_ParOutFeatLinearLayerReLU
		Input, Output, Bias and Filter are tiled. Outputs are evaluated in parallel, one per core, in case In does not fit entirely in share L1 the inner
		iterator will read for L2, in a tiled manner, for each group of evaluated ouput (outer loop)

	CNN_LinearLayerReLU
		Input, Bias and Output are assumed to fit into shared L1 (Buffer In or Out) and Filter is tiled.

*********************************************************************************************************************************************************************/

void CNN_ParOutFeatLinearLayerReLU(
	char *Name,

	unsigned int In_DataSize,
	unsigned int Filter_DataSize,
	unsigned int Bias_DataSize,
	unsigned int Out_DataSize,

	unsigned int In_InL3,
	unsigned int Filter_InL3,
	unsigned int Bias_InL3,
	unsigned int Out_InL3,

	unsigned int InDim,
	unsigned int OutDim,

	unsigned int DoReLU
	)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	int NormedBias = 0;

	char *SetBiasKerName, *LinearKerName, *ReLUKerName;
	char *InDataType, *FilterDataType, *BiasDataType, *OutDataType;

	if        (In_DataSize==2 && Filter_DataSize==2 && Bias_DataSize==2 && Out_DataSize==2) {

		SetBiasKerName = "KerParSetBias_fp"; LinearKerName = "KerParLinearLayerReLU_fp";
		InDataType = "short int * __restrict__"; FilterDataType = "short int * __restrict__"; BiasDataType = "short int * __restrict__"; OutDataType = "short int * __restrict__";
	} else if (In_DataSize==1 && Filter_DataSize==1 && Bias_DataSize==1 && Out_DataSize==1) {
		SetBiasKerName = "KerParSetBias_fps"; LinearKerName = "KerParLinearLayerReLU_fps";
		InDataType = "signed char * __restrict__"; FilterDataType = "signed char * __restrict__"; BiasDataType = "signed char * __restrict__"; OutDataType = "signed char * __restrict__";
	} else if (In_DataSize==2 && Filter_DataSize==1 && Bias_DataSize==2 && Out_DataSize==2) {
		SetBiasKerName = "KerParSetNormedBias_fp_fps"; LinearKerName = "KerParLinearLayerReLU_fp_fps_fp"; NormedBias = 1;
		InDataType = "short int * __restrict__"; FilterDataType = "signed char * __restrict__"; BiasDataType = "short int * __restrict__"; OutDataType = "short int * __restrict__";
	} else if (In_DataSize==2 && Filter_DataSize==2 && Bias_DataSize==2 && Out_DataSize==4) {
		SetBiasKerName = "KerParSetNormedBias_fpd_fp"; LinearKerName = "KerParLinearLayerReLU_fp_fp_fpd"; NormedBias = 1;
		InDataType = "short int * __restrict__"; FilterDataType = "short int * __restrict__"; BiasDataType = "short int * __restrict__"; OutDataType = "int * __restrict__";
	} else GenTilingError("CNN_ParOutFeatLinearLayerReLU: %s Unsupported data size mix: In:%d, Filter:%d, Bias:%d, Out: %d\n", Name, In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize);

		UserKernel(Name,
		KernelDimensions(1, InDim, OutDim, 1),
				KernelIterationOrder(KER_DIM3, KER_TILE, KER_TILE1),
				TILE_HOR,
				CArgs(6,
					  TCArg(InDataType,     "In"),
					  TCArg(FilterDataType, "Filter"),
					  TCArg(BiasDataType,   "Bias"),
					  TCArg(OutDataType,    "Out"),
					  TCArg("unsigned int", "Norm"),
					  TCArg("unsigned int", "NormBias")
					 ),
				Calls(2,
						Call(SetBiasKerName, LOC_INNER_LOOP1_PROLOG,
								Bindings(6, K_Arg("Out", KER_ARG_TILE), Imm(1), Imm(1), K_Arg("Out", KER_ARG_TILE_H),
						K_Arg("Bias", KER_ARG_TILE), NormedBias?C_Arg("NormBias"):AT_NO_ARG_BINDING)),
			Call(LinearKerName, LOC_INNER_LOOP1,
				Bindings(10,
					K_Arg("In", KER_ARG_TILE),  // In
					K_Arg("In", KER_ARG_TILE_H),    // InSize
					KerDim(K_W),            // TotalInSize
					K_Arg("In", KER_ARG_TILE_BASE), // InBase
					K_Arg("Filter", KER_ARG_TILE_H),// OutSize
					K_Arg("Filter", KER_ARG_TILE),  // Filter
					Imm(0),             // Bias
					K_Arg("Out", KER_ARG_TILE), // Out
					C_Arg("Norm"),          // Norm
					Imm(DoReLU)         // DoReLU
				)
			)
		),
				KerArgs(4,
						KerArg("In",      OBJ_IN_DB|InL3|O_TILE1,     1, InDim,  In_DataSize,     0, 0, 0, "In", 0),
						KerArg("Filter",  OBJ_IN_DB|FilterL3,     InDim, OutDim, Filter_DataSize, 0, 0, 0, "Filter", 0),
						KerArg("Bias",    OBJ_IN_DB|BiasL3,           1, OutDim, Bias_DataSize,   0, 0, 0, "Bias", 0),
						KerArg("Out",     OBJ_OUT_DB|OutL3,           1, OutDim, Out_DataSize,    0, 0, 8, "Out", 0)
						)
	);

	AT_PrepareForTest(Name,
			(v4s) {In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize},
			(v4s) {InL3!=0,FilterL3!=0,BiasL3!=0,OutL3!=0},
			InDim, OutDim, 1, 1,
			0, 0, 0, 0, (v4s) 0, 0, 0,
			0, 0, 0, 0, (v4s) 0, 0, 0,
			1, DoReLU, 0,
			(In_DataSize==1)?7:15,
			(Filter_DataSize==1)?7:15);

}

void CNN_LinearLayerReLU(
	char *Name,

	unsigned int In_DataSize,
	unsigned int Filter_DataSize,
	unsigned int Bias_DataSize,
	unsigned int Out_DataSize,

	unsigned int In_InL3,
	unsigned int Filter_InL3,
	unsigned int Bias_InL3,
	unsigned int Out_InL3,

	unsigned int InDim,
	unsigned int OutDim,

	unsigned int DoReLU
	)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int FilterL3 = Filter_InL3?O_L2DB:0;
	unsigned int BiasL3 = Bias_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	int NormedBias = 0;

	char *SetBiasKerName, *LinearKerName, *ReLUKerName;
	char *InDataType, *FilterDataType, *BiasDataType, *OutDataType;

	if        (In_DataSize==2 && Filter_DataSize==2 && Bias_DataSize==2 && Out_DataSize==2) {
		LinearKerName = "KerLinearLayerReLU_fp";
		InDataType = "short int * __restrict__"; FilterDataType = "short int * __restrict__"; BiasDataType = "short int * __restrict__"; OutDataType = "short int * __restrict__";
	} else if (In_DataSize==1 && Filter_DataSize==1 && Bias_DataSize==1 && Out_DataSize==1) {
		LinearKerName = "KerLinearLayerReLU_fps";
		InDataType = "signed char * __restrict__"; FilterDataType = "signed char * __restrict__"; BiasDataType = "signed char * __restrict__"; OutDataType = "signed char * __restrict__";
	} else if (In_DataSize==2 && Filter_DataSize==1 && Bias_DataSize==2 && Out_DataSize==2) {
		LinearKerName = "KerLinearLayerReLU_fp_fps_fp"; NormedBias = 1;
		InDataType = "short int * __restrict__"; FilterDataType = "signed char * __restrict__"; BiasDataType = "short int * __restrict__"; OutDataType = "short int * __restrict__";
	} else if (In_DataSize==2 && Filter_DataSize==2 && Bias_DataSize==2 && Out_DataSize==4) {
		LinearKerName = "KerLinearLayerReLU_fp_fp_fpd"; NormedBias = 1;
		InDataType = "short int * __restrict__"; FilterDataType = "short int * __restrict__"; BiasDataType = "short int * __restrict__"; OutDataType = "int * __restrict__";
	} else GenTilingError("CNN_LinearLayerReLU: %s Unsupported data size mix: In:%d, Filter:%d, Bias:%d, Out: %d\n", Name, In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize);

		UserKernel(Name,
		KernelDimensions(1, InDim, OutDim, 1),
				KernelIterationOrder(KER_DIM2, KER_TILE),
				TILE_HOR,
				CArgs(6,
					  TCArg(InDataType,     "In"),
					  TCArg(FilterDataType, "Filter"),
					  TCArg(BiasDataType,   "Bias"),
					  TCArg(OutDataType,    "Out"),
					  TCArg("unsigned int", "Norm"),
					  TCArg("unsigned int", "NormBias")
					 ),
				Calls(1,
			Call(LinearKerName, LOC_INNER_LOOP,
				Bindings(11,
					K_Arg("In", KER_ARG_TILE),  // In
					K_Arg("In", KER_ARG_TILE_H),    // InSize
					AT_IGNORE_ARG_BINDING,      // TotalInSize
					AT_IGNORE_ARG_BINDING,      // InBase
					K_Arg("Filter", KER_ARG_TILE_H),// OutSize
					K_Arg("Filter", KER_ARG_TILE),  // Filter
					K_Arg("Bias", KER_ARG_TILE),    // Bias
					K_Arg("Out", KER_ARG_TILE), // Out
					C_Arg("Norm"),          // Norm
					NormedBias?C_Arg("NormBias"):AT_NO_ARG_BINDING,
					Imm(DoReLU)         // DoReLU
				)
			)
		),
		KerArgs(4,
				KerArg("In",      OBJ_BUFFER_IN_NTILED|InL3,      1, InDim,  In_DataSize,     0, 0, 0, "In", 0),
				KerArg("Filter",  OBJ_IN_DB|FilterL3,         InDim, OutDim, Filter_DataSize, 0, 0, 0, "Filter", 0),
				KerArg("Bias",    OBJ_BUFFER_IN|BiasL3,           1, OutDim, Bias_DataSize,   0, 0, 0, "Bias", 0),
				KerArg("Out",     OBJ_BUFFER_OUT|OutL3,           1, OutDim, Out_DataSize,    0, 0, 8, "Out", 0)
				)
	);

    AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, Filter_DataSize, Bias_DataSize, Out_DataSize},
			  (v4s) {InL3!=0,FilterL3!=0,BiasL3!=0,OutL3!=0},
			  InDim, OutDim, 1, 1,
			  0, 0, 0, 0, (v4s) 0, 0, 0,
			  0, 0, 0, 0, (v4s) 0, 0, 0,
			  1, DoReLU, 0,
			  (In_DataSize==1)?7:15,
			  (Filter_DataSize==1)?7:15);

}



/*********************************************************************************************************************************************************************
	Generators for SoftMax layers

	Template:
		Name:       Name of the generated user kernel

		In_DataSize:    1: byte, 2: half word,
		Out_DataSize:   1: byte, 2: half word

		In_InL3:    0: In is in L2, 1: In is in L3 memory
		Out_InL3:   0: Out is in L2, 1: Out is in L3 memory

		Dim:        Number of inputs

	CNN_SoftMaxLayer
				Input and output are assumed to fit within given shared L1 memory. Dim is partitionned into subsets of inputs and each subset is given to
				a different code. By definition Output contains value is the [0.0 .. 1.0] range with sum(Output)=1.0. Results are always represented in Q15
				if DataSize is half word or in Q7 is DataSize is byte while for Input the point position must be provided (Norm)

*********************************************************************************************************************************************************************/

void CNN_SoftMaxLayer(
	char *Name,

	unsigned int In_DataSize,
	unsigned int Out_DataSize,

	unsigned int In_InL3,
	unsigned int Out_InL3,

	unsigned int Dim
	)

{
	unsigned int InL3 = In_InL3?O_L2DB:0;
	unsigned int OutL3 = Out_InL3?O_L2DB:0;
	char *SoftMaxKerName;
	char *InDataType, *OutDataType;

	if        (In_DataSize==2 && Out_DataSize==2) {
		SoftMaxKerName = "KerParSoftMax_fp";
		InDataType = "short int * __restrict__"; OutDataType = "short int * __restrict__";
	} else if (In_DataSize==1 && Out_DataSize==1) {
		SoftMaxKerName = "KerParSoftMax_fps";
		InDataType = "signed char * __restrict__"; OutDataType = "signed char * __restrict__";
	} else GenTilingError("CNN_SoftMaxLayer: %s Unsupported data size: In:%d, Out: %d\n", Name, In_DataSize, Out_DataSize);

		UserKernel(Name,
		KernelDimensions(1, 1, Dim, 1),
				KernelIterationOrder(KER_DIM2, KER_TILE),
				TILE_HOR,
				CArgs(3,
					  TCArg(InDataType,     "In"),
					  TCArg(OutDataType,    "Out"),
					  TCArg("unsigned int", "Norm")
					 ),
				Calls(1,
			Call(SoftMaxKerName, LOC_INNER_LOOP,
				Bindings(4,
					K_Arg("In", KER_ARG_TILE),  // In
					K_Arg("In", KER_ARG_TILE_H),    // N
					C_Arg("Norm"),          // Norm
					K_Arg("Out", KER_ARG_TILE)  // Out
				)
			)
		),
				KerArgs(2,
						KerArg("In",      OBJ_BUFFER_IN|InL3,   1, Dim,  In_DataSize, 0, 0, 8, "In", 0),
						KerArg("Out",     OBJ_BUFFER_OUT|OutL3, 1, Dim, Out_DataSize, 0, 0, 0, "Out", 0)
		)
	);

    AT_PrepareForTest(Name,
			  (v4s) {In_DataSize, 0, 0, Out_DataSize},
			  (v4s) {InL3!=0,0,0,OutL3!=0},
			  1, 1, 1, Dim,
			  0, 0, 0, 0, (v4s) 0, 0, 0,
			  0, 0, 0, 0, (v4s) 0, 0, 0,
			  0, 0, 1, (In_DataSize==1)?7:15, 0);

}

/***************************************************
 *
 *          CNN HWCE GENERATOR
 *
 * ************************************************/

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
						 TCArg("short int * __restrict__", "Out0"),
						 TCArg("short int * __restrict__", "Out1"),
						 TCArg("int", "W"),
						 TCArg("int", "H"),
						 TCArg("short int", "Bias0"),
						 TCArg("short int", "Bias1")
						),
				   "KerSetInBias2T"
		);
		LibKernel("KerSetInBias3", CALL_PARALLEL,
				   CArgs(8,
						 TCArg("short int * __restrict__", "Out0"),
						 TCArg("short int * __restrict__", "Out1"),
						 TCArg("short int * __restrict__", "Out2"),
						 TCArg("int", "W"),
						 TCArg("int", "H"),
						 TCArg("short int", "Bias0"),
						 TCArg("short int", "Bias1"),
						 TCArg("short int", "Bias2")
						),
				   "KerSetInBias3T"
		);
		LibKernel("HWCE_ProcessOneTile3x3_MultiOut", CALL_SEQUENTIAL,
				  CArgs(9,
						TCArg("short int * __restrict__", "In"),
						TCArg("short int * __restrict__", "Out0"),
						TCArg("short int * __restrict__", "Out1"),
						TCArg("short int * __restrict__", "Out2"),
						TCArg("short int * __restrict__", "Filter"),
						TCArg("short int", "Bias"),
						TCArg("unsigned int", "W"),
						TCArg("unsigned int", "H"),
						TCArg("unsigned int", "OutMask")
					   ),
				   ""
				);
		LibKernel("HWCE_ProcessOneTile5x5", CALL_SEQUENTIAL,
				  CArgs(6,
						TCArg("short int * __restrict__", "In"),
						TCArg("short int * __restrict__", "Out"),
						TCArg("short int * __restrict__", "Filter"),
						TCArg("short int", "Bias"),
						TCArg("unsigned int", "W"),
						TCArg("unsigned int", "H")
					   ),
				   ""
				);
		LibKernel("HWCE_ProcessOneTile7x7", CALL_SEQUENTIAL,
				  CArgs(6,
						TCArg("short int * __restrict__", "In"),
						TCArg("short int * __restrict__", "Out"),
						TCArg("short int * __restrict__", "Filter"),
						TCArg("short int", "Bias"),
						TCArg("unsigned int", "W"),
						TCArg("unsigned int", "H")
					   ),
				   ""
				);
		LibKernel("HWCE_ProcessOneTile7x4", CALL_SEQUENTIAL,
				  CArgs(6,
						TCArg("short int * __restrict__", "In"),
						TCArg("short int * __restrict__", "Out"),
						TCArg("short int * __restrict__", "Filter"),
						TCArg("short int", "Bias"),
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


/* HWCE enabled convolutions */

/* Pure convolutions, no bias setting and no accumulation */
void CNN_TiledPlainConvNxN_HWCE_fp(char *Name, unsigned int FS, unsigned int Width, unsigned int Height)

{
	char *ConvKerName;
	int Fw, Fh; /* Filter dimensions, Since FS*FS is odd and HwCE supports only 4 byte aligned accesses Fw * Fh = FS * FS + 1 */
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
			  TCArg("unsigned int",         "Norm")
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
			KerArg("In",     OBJ_IN_DB,            Width,      Height,  sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_BUFFER_IN_NTILED, Fw,     Fh,      sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out",    OBJ_OUT_DB,           Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out", 0)
			   )
	);
}

/*  Convolution layer, bias setting and accumulation */
void CNN_TiledConvNxN_HWCE_fp(char *Name, unsigned int FS, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height)

{
	char *ConvKerName;
	int Fw, Fh; /* Filter dimensions, Since FS*FS is odd and HwCE supports only 4 byte aligned accesses Fw * Fh = FS * FS + 1 */
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
			  TCArg("unsigned int",         "Norm")
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
			KerArg("In",     OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, Fw,     Fh,          sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out",    OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out", 0)
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
			  TCArg("unsigned int",         "Norm"),
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
			  TCArg("unsigned int",         "Norm"),
			  TCArg("short int * __restrict__", "Bias0"),
			  TCArg("short int * __restrict__", "Bias1")
			 ):
		CArgs(5,
			  TCArg("short int * __restrict__", "In"),
			  TCArg("short int * __restrict__", "Filter"),
			  TCArg("short int * __restrict__", "Out0"),
			  TCArg("unsigned int",         "Norm"),
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
			KerArg("In",     OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, 7,      4,           sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out0",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out0", 3),
			KerArg("Out1",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out1", 3),
			KerArg("Out2",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out2", 3)
			   ):
		((Nout == 2)?
		KerArgs(4,
			KerArg("In",     OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, 3*2,    3,           sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out0",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out0", 2),
			KerArg("Out1",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out1", 2)
			   ):
		KerArgs(3,
			KerArg("In",     OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter", OBJ_IN_DB_NTILED_4D, 5,      2,           sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("Out0",   OBJ_OUT_DB_3D,       Width-FS+1, Height-FS+1, sizeof(short int), 0,  0, 0,  "Out0", 0)
			   ))
	);
}


/* HWCE enabled composite kernels: NxN convolutions, ReLU and Max or Average Pooling 2x2 -> 1 */
void CNN_TiledConvNxNReLUPool2x2_HWCE_fp(char *Name, unsigned int FS, unsigned int InPlane, unsigned int OutPlane, unsigned int Width, unsigned int Height,
					 unsigned int PoolMax)

{
	char *ConvKerName, *KerReLUPoolName;
	int Fw, Fh; /* Filter dimensions, Since FS*FS is odd and HwCE supports only 4 byte aligned accesses Fw * Fh = FS * FS + 1 */
	int ConvMode, Mode3x3 = 0;
	v4s Pad = {0,0,0,0};
	unsigned char M=0, S=0, Orientation=0;
	unsigned short int DoReLU = 0;


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
		case 0: /* No Relu, PoolMax */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=2; S=2; Orientation=1; DoReLU=0;  break;
		case 1: /* Relu, PoolMax */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=2; S=2; Orientation=1; DoReLU=1;  break;
		case 2: /* No Relu, PoolAvg */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=2; S=2; Orientation=1; DoReLU=2;  break;
		case 3: /* Relu, PoolAvg */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=2; S=2; Orientation=1; DoReLU=3;  break;

		case 4: /* No Relu, PoolMax 3x3 */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=3; S=2; Orientation=1; DoReLU=0;  break;
		case 5: /* Relu, PoolMax 3x3 */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=3; S=2; Orientation=1; DoReLU=1;  break;
		case 6: /* No Relu, PoolAvg 3x3 */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=3; S=2; Orientation=1; DoReLU=2;  break;
		case 7: /* Relu, PoolAvg 2x2 */
			KerReLUPoolName = "KerPoolNxNStrideS_fp"; M=3; S=2; Orientation=1; DoReLU=3;  break;
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
			  TCArg("unsigned int",         "Norm"),
			  TCArg("short int * __restrict__", "Bias")
			 ),
		Calls(7,
			Call("HWCE_Enable", LOC_PROLOG, Bindings(0)),
			Call("HWCE_GenericInit", LOC_PROLOG, Bindings(3, Imm(0), Imm(0), C_Arg("Norm"))),
			Call("HwCE_SetYinMode", LOC_IN_PLANE_PROLOG, Bindings(1, Imm(1))),
			Mode3x3?
			Call(ConvKerName, LOC_IN_PLANE,
				Bindings(9, 
							K_Arg("In", KER_ARG_TILE), 
							K_Arg("Out", KER_ARG_TILE), 
							Imm(0), 
							Imm(0),
							K_Arg("Filter", KER_ARG_TILE), 
							C_ArgIndex("Bias", KER_OUT_PLANE, 1),
							K_Arg("In", KER_ARG_TILE_W), 
							K_Arg("In", KER_ARG_TILE_H), 
							Imm(0x7))):
			Call(ConvKerName, LOC_IN_PLANE,
				Bindings(6, 
						K_Arg("In", KER_ARG_TILE), 
						K_Arg("SetBiasOut", KER_ARG_TILE), 
						K_Arg("Filter", KER_ARG_TILE),
						C_ArgIndex("Bias", KER_OUT_PLANE, 1), 
						K_Arg("In", KER_ARG_TILE_W), 
						K_Arg("In", KER_ARG_TILE_H))
				),
			Call("HwCE_SetYinMode", LOC_IN_PLANE, Bindings(1, Imm(0))),
			Call(KerReLUPoolName, LOC_IN_PLANE_EPILOG,
				Bindings(13,
					K_Arg("SetBiasOut", KER_ARG_TILE),
					K_Arg("SetBiasOut", KER_ARG_TILE_W),
					K_Arg("SetBiasOut", KER_ARG_TILE_H),
					K_Arg("Out", KER_ARG_TILE),
					Imm((int)Pad), 
					Imm(M), 
					Imm(S),
					Imm(Orientation), 
					Imm(DoReLU),
					Imm(1), //Dilation 
					Imm(M), //Only square
					Imm(S), //Only square
					Imm(1)  //Dilation 
					)
				),
			Call("HWCE_Disable", LOC_EPILOG, Bindings(0))
			 ),
		KerArgs(4,
			KerArg("In",          OBJ_IN_DB_3D,        Width,      Height,      sizeof(short int), FS-1, 0, M, "In", 0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D, Fw,             Fh,              sizeof(short int), 0,    0, 0,    "Filter", 0),
			KerArg("SetBiasOut",  OBJ_BUFFER_ONETILE,  (Width-FS+1),   Height-FS+1,     sizeof(short int), 0,    0, 0,    "", 0),
			KerArg("Out",         OBJ_OUT_DB_3D,       (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,    0, 0,    "Out", 0)
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
			  TCArg("unsigned int",         "Norm"),
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
			  TCArg("unsigned int",         "Norm"),
			  TCArg("short int * __restrict__", "Bias0"),
			  TCArg("short int * __restrict__", "Bias1")
			 ):
		CArgs(5,
			  TCArg("short int * __restrict__", "In"),
			  TCArg("short int * __restrict__", "Filter"),
			  TCArg("short int * __restrict__", "Out0"),
			  TCArg("unsigned int",         "Norm"),
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
			KerArg("In",          OBJ_IN_DB_3D,         Width,          Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D,  7,          4,           sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("SetBiasOut0", OBJ_BUFFER_ONETILE,   (Width-FS+1),   (Height-FS+1),   sizeof(short int), 0,  0, 0,  "", 0),
			KerArg("SetBiasOut1", OBJ_BUFFER_ONETILE,   (Width-FS+1),   (Height-FS+1),   sizeof(short int), 0,  0, 0,  "", 0),
			KerArg("SetBiasOut2", OBJ_BUFFER_ONETILE,   (Width-FS+1),   (Height-FS+1),   sizeof(short int), 0,  0, 0,  "", 0),
			KerArg("Out0",        OBJ_OUT_DB_3D,        (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,  0, 0,  "Out0", 3),
			KerArg("Out1",        OBJ_OUT_DB_3D,        (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,  0, 0,  "Out1", 3),
			KerArg("Out2",        OBJ_OUT_DB_3D,        (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,  0, 0,  "Out2", 3)
			   ):
		((Nout == 2) ?
		KerArgs(6,
			KerArg("In",          OBJ_IN_DB_3D,         Width,          Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D,  3*2,            3,           sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("SetBiasOut0", OBJ_BUFFER_ONETILE,   (Width-FS+1),   (Height-FS+1),   sizeof(short int), 0,  0, 0,  "", 0),
			KerArg("SetBiasOut1", OBJ_BUFFER_ONETILE,   (Width-FS+1),   (Height-FS+1),   sizeof(short int), 0,  0, 0,  "", 0),
			KerArg("Out0",        OBJ_OUT_DB_3D,        (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,  0, 0,  "Out0", 2),
			KerArg("Out1",        OBJ_OUT_DB_3D,        (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,  0, 0,  "Out1", 2)
			   ):
		KerArgs(4,
			KerArg("In",          OBJ_IN_DB_3D,         Width,          Height,      sizeof(short int), FS-1, OBJ_CONSTRAINTS_EVEN, 0, "In", 0),
			KerArg("Filter",      OBJ_IN_DB_NTILED_4D,  5,          2,           sizeof(short int), 0,  0, 0,  "Filter", 0),
			KerArg("SetBiasOut0", OBJ_BUFFER_ONETILE,   (Width-FS+1),   Height-FS+1,     sizeof(short int), 0,  0, 0,  "", 0),
			KerArg("Out0",        OBJ_OUT_DB_3D,        (Width-FS+1)/2, (Height-FS+1)/2, sizeof(short int), 0,  0, 0,  "Out0", 0)
			   )
		)
	);
}
