#ifndef __CNN_BASIC_KERNELS_H__
#define __CNN_BASIC_KERNELS_H__
#include "Gap8.h"

/********************************************************************************************************************************************************************/
/****************** Bias setting. One output evaluated in parallel on all the cores *********************************************************************************/
/********************************************************************************************************************************************************************/

typedef struct
{
    signed char *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    unsigned short int OutFeatures;
    signed char *__restrict__ Bias;
} KerParSetBias_fps_T;

typedef struct
{
    short int *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    unsigned short int OutFeatures;
    short int *__restrict__ Bias;
} KerParSetBias_fp_T;

typedef struct
{
    short int *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    unsigned short int OutFeatures;
    unsigned short int Norm;
    signed char *__restrict__ Bias;
} KerParSetNormedBias_fp_fps_T;

typedef struct
{
    int *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    unsigned short int OutFeatures;
    unsigned short int Norm;
    short int *__restrict__ Bias;
} KerParSetNormedBias_fpd_fp_T;


/********************************************************************************************************************************************************************/
/****************** Bias setting. Several outputs evaluated in parallel, one on each core ***************************************************************************/
/********************************************************************************************************************************************************************/

typedef struct
{
    signed char *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    signed char *__restrict__ Bias;
} KerSetBias_fps_T;

typedef struct
{
    short int *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    short int *__restrict__ Bias;
} KerSetBias_fp_T;

typedef struct
{
    short int *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    short int Bias;
    unsigned short int Norm;
} KerSetNormedBias_fp_fps_T;

typedef struct
{
    int *__restrict__ Out;
    unsigned short int W;
    unsigned short int H;
    short int Bias;
    unsigned short int Norm;
} KerSetNormedBias_fpd_fp_T;


/********************************************************************************************************************************************************************/
/****************** Convolution. One output evaluated in parallel on all the cores **********************************************************************************/
/********************************************************************************************************************************************************************/

typedef struct
{
    short int *__restrict__ In;         /**< Pointer to input tile  */
    unsigned short int W;               /**< Width of the input tile */
    unsigned short int H;               /**< Height of the input tile */
    short int *__restrict__ Filter;     /**< Pointer to convolution coefficients. (Nx x Ny) coeffs in Q15 */
    short int *__restrict__ Out;        /**< Pointer to output tile, this tile can have up to N-1 lines and N-1 column than In depending on Pad */
    v4s Pad;                            /**< Paddding, 0: Left, 1: Right, 2: Top, 3: Bottom */
    unsigned char Norm;                 /**< Fixed point format, should be <= 15 */
    unsigned char N;                    /**< Dimension of the convolution: Nx, NxN, used only for general versions */
    unsigned char S;                            /**< Output stride, Sx, used only for general versions */
    unsigned char Orientation;          /**< Tile orientation: 1 => Horizontal, 0 => Vertical */
    unsigned char D;                    /**< Dilation Dx */
    unsigned char Ny;                   /**< Filter Ny, used only if Nx!=Ny */
    unsigned char Sy;                   /**< Stride Sy, used only if Sx!=Sy */
    unsigned char Dy;                   /**< Dilation Dy, used only if Dx!=Dy */
} KerConv_fp_T;

typedef struct
{
    signed char *__restrict__ In;       /**< Pointer to input tile  */
    unsigned short int W;               /**< Width of the input tile */
    unsigned short int H;               /**< Height of the input tile */
    signed char *__restrict__ Filter;   /**< Pointer to convolution coefficients. (Nx x Ny) coeffs in Q15 */
    signed char *__restrict__ Out;      /**< Pointer to output tile, this tile can have up to N-1 lines and N-1 column than In depending on Pad */
    v4s Pad;                            /**< Paddding, 0: Left, 1: Right, 2: Top, 3: Bottom */
    unsigned char Norm;                 /**< Fixed point format, should be <= 7 */
    unsigned char N;                    /**< Dimension of the convolution: Nx, NxN, used only for general versions */
    unsigned char S;                            /**< Output stride, S, used only for general versions */
    unsigned char Orientation;          /**< Tile orientation: 1 => Horizontal, 0 => Vertical */
    unsigned char D;                    /**< Dilation Dx */
    unsigned char Ny;                   /**< Filter Ny, used only if Nx!=Ny */
    unsigned char Sy;                   /**< Stride Sy, used only if Sx!=Sy */
    unsigned char Dy;                   /**< Dilation Dy, used only if Dx!=Dy */
} KerConv_fps_T;


/********************************************************************************************************************************************************************/
/****************** Convolution. Several outputs evaluated in parallel, one on each core ****************************************************************************/
/********************************************************************************************************************************************************************/

typedef struct
{
    short int *__restrict__ In;
    unsigned short int W;
    unsigned short int H;
    unsigned short int InFeatures;
    unsigned short int OutFeatures;
    unsigned short int BaseInFeature;
    unsigned short int BaseOutFeature;
    short int TotalInFeatures;          /**< If !=0 Input feature space is tiled, BaseFeature is the offset in input feature space. If <0 depth wise convolution */
    short int *__restrict__ Filter;
    short int *__restrict__ Out;
    v4s Pad;                            /**< Paddding, 0: Left, 1: Right, 2: Top, 3: Bottom */
    unsigned char Norm;
    unsigned char N;                    /**< Dimension of the convolution: Nx, NxN, used only for general versions */
    unsigned char S;                    /**< Output stride, Sx, used only for general versions */
    unsigned char D;                    /**< Dilation Dx */
    unsigned char Ny;                   /**< Filter Ny, used only if Nx!=Ny */
    unsigned char Sy;                   /**< Stride Sy, used only if Sx!=Sy */
    unsigned char Dy;                   /**< Dilation Dy, used only if Dx!=Dy */
} KerParConv_fp_T;

typedef struct
{
    signed char *__restrict__ In;
    unsigned short int W;
    unsigned short int H;
    unsigned short int InFeatures;
    unsigned short int OutFeatures;
    unsigned short int BaseInFeature;
    unsigned short int BaseOutFeature;
    short int TotalInFeatures;          /**< If !=0 Input feature space is tiled, BaseFeature is the offset in input feature space. If <0 depth wise convolution */
    signed char *__restrict__ Filter;
    signed char *__restrict__ Out;
    v4s Pad;                            /**< Paddding, 0: Left, 1: Right, 2: Top, 3: Bottom */
    unsigned char Norm;
    unsigned char N;                    /**< Dimension of the convolution: Nx, NxN, used only for general versions */
    unsigned char S;                    /**< Output stride, Sx, used only for general versions */
    unsigned char D;                    /**< Dilation Dx */
    unsigned char Ny;                   /**< Filter Ny, used only if Nx!=Ny */
    unsigned char Sy;                   /**< Stride Sy, used only if Sx!=Sy */
    unsigned char Dy;                   /**< Dilation Dy, used only if Dx!=Dy */
} KerParConv_fps_T;


/********************************************************************************************************************************************************************/
/****************** Pooling/Linear rectification. One output evaluated in parallel on all the cores *****************************************************************/
/********************************************************************************************************************************************************************/

typedef struct
{
    short int *__restrict__ In;     /**< Pointer to input tile  */
    unsigned short int W;           /**< Width of the input tile */
    unsigned short int H;           /**< Height of the input tile */
    short int *__restrict__ Out;    /**< Pointer to output tile */
    v4s Pad;
    unsigned char M;                /**< Pooling dimension */
    unsigned char S;                    /**< Stride */
    unsigned char Orientation;      /**< Tile orientation: 1 => Horizontal, 0 => Vertical */
    unsigned char DoReLU;               /**< Bit0: ReLU, Bit: MayPool/AvgPool. 0x0: MaxPool and NoReLU, 0x1: MaxPool and ReLU, 0x2: AvgPool, No ReLU, 0x3: AvgPool and ReLU */
    unsigned char D;            /**< Dilation, Dx */
    unsigned char My;           /**< Filter My, used only if Mx!=My */
    unsigned char Sy;           /**< Stride Sy, used only if Sx!=Sy */
    unsigned char Dy;           /**< Dilation Dy, used only if Dx!=Dy */
} KerReLUPool_fp_T;

typedef struct
{
    signed char *__restrict__ In;   /**< Pointer to input tile  */
    unsigned short int W;           /**< Width of the input tile */
    unsigned short int H;           /**< Height of the input tile */
    signed char *__restrict__ Out;  /**< Pointer to output tile */
    v4s Pad;
    unsigned char M;                /**< Pooling dimension */
    unsigned char S;                    /**< Stride */
    unsigned char Orientation;      /**< Tile orientation: 1 => Horizontal, 0 => Vertical */
    unsigned char DoReLU;               /**< Bit0: ReLU, Bit: MayPool/AvgPool. 0x0: MaxPool and NoReLU, 0x1: MaxPool and ReLU, 0x2: AvgPool, No ReLU, 0x3: AvgPool and ReLU */
    unsigned char D;            /**< Dilation, Dx */
    unsigned char My;           /**< Filter My, used only if Mx!=My */
    unsigned char Sy;           /**< Stride Sy, used only if Sx!=Sy */
    unsigned char Dy;           /**< Dilation Dy, used only if Dx!=Dy */
} KerReLUPool_fps_T;


/********************************************************************************************************************************************************************/
/****************** Pooling/Linear rectification. Several outputs evaluated in parallel, one on each core ***********************************************************/
/********************************************************************************************************************************************************************/

typedef struct
{
    short int *__restrict__ In;
    unsigned short int W;
    unsigned short int H;
    unsigned short int OutFeatures;
    short int *__restrict__ Out;
    v4s Pad;
    unsigned char M;                /**< Pooling dimension, Mx */
    unsigned char S;                    /**< Stride, Sx */
    unsigned char Orientation;      /**< Tile orientation: 1 => Horizontal, 0 => Vertical */
    unsigned char DoReLU;
    unsigned char My;               /**< Pooling dimension My, used only if Mx!=My */
    unsigned char Sy;                   /**< Stride, Sy, used only if Sx!=Sy */
} KerParReLUPool_fp_T;

typedef struct
{
    signed char *__restrict__ In;
    unsigned short int W;
    unsigned short int H;
    unsigned short int OutFeatures;
    signed char *__restrict__ Out;
    v4s Pad;
    unsigned char M;                /**< Pooling dimension, Mx */
    unsigned char S;                    /**< Stride, Sx */
    unsigned char Orientation;      /**< Tile orientation: 1 => Horizontal, 0 => Vertical */
    unsigned char DoReLU;
    unsigned char My;               /**< Pooling dimension My, used only if Mx!=My */
    unsigned char Sy;                   /**< Stride, Sy, used only if Sx!=Sy */
} KerParReLUPool_fps_T;

/********************************************************************************************************************************************************************/
/****************** Add plus optional Relu. *************************************************************************************************************************/
/********************************************************************************************************************************************************************/

typedef struct {
        short int * __restrict__ In1;            /**< Pointer to first input tile  */
        short int * __restrict__ In2;            /**< Pointer to second input tile  */
        unsigned short int W;                   /**< Width of the input tile */
        unsigned short int H;                   /**< Height of the input tile */
        short int * __restrict__ Out;           /**< Pointer to output tile*/
} KerAddReLU_fp_T;


/********************************************************************************************************************************************************************/
/****************** Linear layer/Linear rectification. For both one output in parallel and several outputs evaluated in parallel ************************************/
/********************************************************************************************************************************************************************/

typedef struct
{
    short int *__restrict__ In;         /**< Pointer to input tile */
    unsigned short int InSize;          /**< Size of the the tile */
    unsigned short int TotalInSize;             /**< Total input size in case parallelization is performed on outputs */
    unsigned short int InBase;          /**< Where does the current tile starts in the overall input group in case parallelization is performed on outputs */
    unsigned short int OutSize;         /**< Size of the output tile */
    short int *__restrict__ Filter;     /**< Pointer to filter tile, width is TotalInSize */
    short int *__restrict__ Bias;               /**< Pointer to bias tile, size is OutSize */
    short int *__restrict__ Out;                /**< Pointer to output tile, size if OutSize */
    unsigned char Norm;                 /**< Normalization factor */
    unsigned char DoReLU;                       /**< If 0: No linear rectification, if 1: Linear rectification after an output has been fully evaluated */
} KerLinearLayerReLU_fp_T;

typedef struct
{
    signed char *__restrict__ In;               /**< Pointer to input tile */
    unsigned short int InSize;          /**< Size of the the tile */
    unsigned short int TotalInSize;             /**< Total input size in case parallelization is performed on outputs */
    unsigned short int InBase;          /**< Where does the current tile starts in the overall input group in case parallelization is performed on outputs */
    unsigned short int OutSize;         /**< Size of the output tile */
    signed char *__restrict__ Filter;   /**< Pointer to filter tile, width is TotalInSize */
    signed char *__restrict__ Bias;     /**< Pointer to bias tile, size is OutSize */
    signed char *__restrict__ Out;              /**< Pointer to output tile, size if OutSize */
    unsigned char Norm;                 /**< Normalization factor */
    unsigned char DoReLU;                       /**< If 0: No linear rectification, if 1: Linear rectification after an output has been fully evaluated */
} KerLinearLayerReLU_fps_T;

typedef struct
{
    short int *__restrict__ In;         /**< Pointer to input tile */
    unsigned short int InSize;          /**< Size of the the tile */
    unsigned short int TotalInSize;             /**< Total input size in case parallelization is performed on outputs */
    unsigned short int InBase;          /**< Where does the current tile starts in the overall input group in case parallelization is performed on outputs */
    unsigned short int OutSize;         /**< Size of the output tile */
    signed char *__restrict__ Filter;   /**< Pointer to filter tile, width is TotalInSize */
    short int *__restrict__ Bias;               /**< Pointer to bias tile, size is OutSize */
    short int *__restrict__ Out;                /**< Pointer to output tile, size if OutSize */
    unsigned char Norm;                 /**< Normalization factor */
    unsigned char NormBias;                     /**< Normalization factor for the bias */
    unsigned char DoReLU;                       /**< If 0: No linear rectification, if 1: Linear rectification after an output has been fully evaluated */
} KerLinearLayerReLU_fp_fps_fp_T;

typedef struct
{
    short int *__restrict__ In;         /**< Pointer to input tile */
    unsigned short int InSize;          /**< Size of the the tile */
    unsigned short int TotalInSize;             /**< Total input size in case parallelization is performed on outputs */
    unsigned short int InBase;          /**< Where does the current tile starts in the overall input group in case parallelization is performed on outputs */
    unsigned short int OutSize;         /**< Size of the output tile */
    short int *__restrict__ Filter;     /**< Pointer to filter tile, width is TotalInSize */
    short int *__restrict__ Bias;               /**< Pointer to bias tile, size is OutSize */
    int *__restrict__ Out;                      /**< Pointer to output tile, size if OutSize */
    unsigned char Norm;                 /**< Normalization factor */
    unsigned char NormBias;                     /**< Normalization factor for the bias */
    unsigned char DoReLU;                       /**< If 0: No linear rectification, if 1: Linear rectification after an output has been fully evaluated */
} KerLinearLayerReLU_fp_fp_fpd_T;


/******************************************************************************************************************************/
/******************* SOFT MAX *************************************************************************************************/
/******************************************************************************************************************************/

typedef struct
{
    short int *__restrict__ In;         /**< Pointer to input tile */
    unsigned short int N;                       /**< Size of the tile */
    unsigned short int Norm;            /**< Normalization factor */
    short int *__restrict__ Out;                /**< Pointer to output tile */
} KerParSoftMax_fp_T;

typedef struct
{
    signed char *__restrict__ In;               /**< Pointer to input tile */
    unsigned short int N;                       /**< Size of the tile */
    unsigned short int Norm;            /**< Normalization factor */
    signed char *__restrict__ Out;              /**< Pointer to output tile */
} KerParSoftMax_fps_T;

/******************************************************************************************************************************/
/**************** BIAS SETTING ************************************************************************************************/
/******************************************************************************************************************************/

extern void KerParSetBias_fp(KerParSetBias_fp_T *Arg);
extern void KerParSetBias_fps(KerParSetBias_fps_T *Arg);
extern void KerParSetNormedBias_fp_fps(KerParSetNormedBias_fp_fps_T *Arg);
extern void KerParSetNormedBias_fpd_fp(KerParSetNormedBias_fpd_fp_T *Arg);

extern void KerSetBias_fp(KerSetBias_fp_T *Arg);
extern void KerSetBias_fps(KerSetBias_fps_T *Arg);
extern void KerSetNormedBias_fp_fps(KerSetNormedBias_fp_fps_T *Arg);
extern void KerSetNormedBias_fpd_fp(KerSetNormedBias_fpd_fp_T *Arg);


/******************************************************************************************************************************/
/**************** CONVOLUTIONS ************************************************************************************************/
/******************************************************************************************************************************/
/*
        1x1, 3x3, 5x5, NxN convolutions with Stride 1, 2 or S.
        Input, Output and filter are half words (_fp)
        Features are evaluated in parallel, one per core
*/
extern void KerParConv1x1Stride1_fp(KerParConv_fp_T *Arg);
extern void KerParConv1x1Stride2_fp(KerParConv_fp_T *Arg);
extern void KerParConv1x1StrideS_fp(KerParConv_fp_T *Arg);

extern void KerParConv3x1Stride1x1_fp(KerParConv_fp_T *Arg);
extern void KerParConv3x1Stride2x1_fp(KerParConv_fp_T *Arg);
extern void KerParConv1x3Stride1x1_fp(KerParConv_fp_T *Arg);
extern void KerParConv1x3Stride1x2_fp(KerParConv_fp_T *Arg);
extern void KerParConv3x3Stride1_fp(KerParConv_fp_T *Arg);
extern void KerParConv3x3Stride2_fp(KerParConv_fp_T *Arg);
extern void KerParConv3x3StrideS_fp(KerParConv_fp_T *Arg);

extern void KerParConv5x1Stride1x1_fp(KerParConv_fp_T *Arg);
extern void KerParConv5x1Stride2x1_fp(KerParConv_fp_T *Arg);
extern void KerParConv1x5Stride1x1_fp(KerParConv_fp_T *Arg);
extern void KerParConv1x5Stride1x2_fp(KerParConv_fp_T *Arg);
extern void KerParConv5x5Stride1_fp(KerParConv_fp_T *Arg);
extern void KerParConv5x5Stride2_fp(KerParConv_fp_T *Arg);
extern void KerParConv5x5StrideS_fp(KerParConv_fp_T *Arg);

extern void KerParConvNxNStrideS_fp(KerParConv_fp_T *Arg);
extern void KerParConvNxMStrideSxSy_fp(KerParConv_fp_T *Arg);

/*
        1x1, 3x3, 5x5, NxN convolutions with Stride 1, 2 or S.
        Input, Output and filter are bytes (_fps)
        Features are evaluated in parallel, one per core
*/
extern void KerParConv1x1Stride1_fps(KerParConv_fps_T *Arg);
extern void KerParConv1x1Stride2_fps(KerParConv_fps_T *Arg);
extern void KerParConv1x1StrideS_fps(KerParConv_fps_T *Arg);

extern void KerParConv3x1Stride1x1_fps(KerParConv_fps_T *Arg);
extern void KerParConv3x1Stride2x1_fps(KerParConv_fps_T *Arg);
extern void KerParConv1x3Stride1x1_fps(KerParConv_fps_T *Arg);
extern void KerParConv1x3Stride1x2_fps(KerParConv_fps_T *Arg);
extern void KerParConv3x3Stride1_fps(KerParConv_fps_T *Arg);
extern void KerParConv3x3Stride2_fps(KerParConv_fps_T *Arg);
extern void KerParConv3x3StrideS_fps(KerParConv_fps_T *Arg);

extern void KerParConv5x1Stride1x1_fps(KerParConv_fps_T *Arg);
extern void KerParConv5x1Stride2x1_fps(KerParConv_fps_T *Arg);
extern void KerParConv1x5Stride1x1_fps(KerParConv_fps_T *Arg);
extern void KerParConv1x5Stride1x2_fps(KerParConv_fps_T *Arg);
extern void KerParConv5x5Stride1_fps(KerParConv_fps_T *Arg);
extern void KerParConv5x5Stride2_fps(KerParConv_fps_T *Arg);
extern void KerParConv5x5StrideS_fps(KerParConv_fps_T *Arg);

extern void KerParConvNxNStrideS_fps(KerParConv_fps_T *Arg);
extern void KerParConvNxMStrideSxSy_fps(KerParConv_fps_T *Arg);

/*
        1x1, 3x3, 5x5, NxN convolutions with Stride 1, 2 or S.
        Input, Output and filter are half words (_fp)
        A single convolution is evaluated in parallel on all the cores
*/
extern void KerConv1x1Stride1_fp(KerConv_fp_T *Arg);
extern void KerConv1x1Stride2_fp(KerConv_fp_T *Arg);
extern void KerConv1x1StrideS_fp(KerConv_fp_T *Arg);

extern void KerConv3x1Stride1x1_fp(KerConv_fp_T *Arg);
extern void KerConv3x1Stride2x1_fp(KerConv_fp_T *Arg);
extern void KerConv1x3Stride1x1_fp(KerConv_fp_T *Arg);
extern void KerConv1x3Stride1x2_fp(KerConv_fp_T *Arg);
extern void KerConv3x3Stride1_fp(KerConv_fp_T *Arg);
extern void KerConv3x3Stride2_fp(KerConv_fp_T *Arg);
extern void KerConv3x3StrideS_fp(KerConv_fp_T *Arg);

extern void KerConv5x1Stride1x1_fp(KerConv_fp_T *Arg);
extern void KerConv5x1Stride2x1_fp(KerConv_fp_T *Arg);
extern void KerConv1x5Stride1x1_fp(KerConv_fp_T *Arg);
extern void KerConv1x5Stride1x2_fp(KerConv_fp_T *Arg);
extern void KerConv5x5Stride1_fp(KerConv_fp_T *Arg);
extern void KerConv5x5Stride2_fp(KerConv_fp_T *Arg);
extern void KerConv5x5StrideS_fp(KerConv_fp_T *Arg);

extern void KerConvNxNStrideS_fp(KerConv_fp_T *Arg);
extern void KerConvNxMStrideSxSy_fp(KerConv_fp_T *Arg);


/*
        1x1, 3x3, 5x5, NxN convolutions with Stride 1, 2 or S.
        Input, Output and filter are bytes(_fps)
        A single convolution is evaluated in parallel on all the cores
*/
extern void KerConv1x1Stride1_fps(KerConv_fps_T *Arg);
extern void KerConv1x1Stride2_fps(KerConv_fps_T *Arg);
extern void KerConv1x1StrideS_fps(KerConv_fps_T *Arg);

extern void KerConv3x1Stride1x1_fps(KerConv_fps_T *Arg);
extern void KerConv3x1Stride2x1_fps(KerConv_fps_T *Arg);
extern void KerConv1x3Stride1x1_fps(KerConv_fps_T *Arg);
extern void KerConv1x3Stride1x2_fps(KerConv_fps_T *Arg);
extern void KerConv3x3Stride1_fps(KerConv_fps_T *Arg);
extern void KerConv3x3Stride2_fps(KerConv_fps_T *Arg);
extern void KerConv3x3StrideS_fps(KerConv_fps_T *Arg);

extern void KerConv5x1Stride1x1_fps(KerConv_fps_T *Arg);
extern void KerConv5x1Stride2x1_fps(KerConv_fps_T *Arg);
extern void KerConv1x5Stride1x1_fps(KerConv_fps_T *Arg);
extern void KerConv1x5Stride1x2_fps(KerConv_fps_T *Arg);
extern void KerConv5x5Stride1_fps(KerConv_fps_T *Arg);
extern void KerConv5x5Stride2_fps(KerConv_fps_T *Arg);
extern void KerConv5x5StrideS_fps(KerConv_fps_T *Arg);

extern void KerConvNxNStrideS_fps(KerConv_fps_T *Arg);
extern void KerConvNxMStrideSxSy_fps(KerConv_fps_T *Arg);



/******************************************************************************************************************************/
/**************** LINEAR  RECTIFICATION ***************************************************************************************/
/******************************************************************************************************************************/

/* Feature maps are evaluated in parallel, one per core.
   Feature maps of bytes (_fps) or half words (_fp)
*/
extern void KerParReLU_fp(KerParReLUPool_fp_T *Arg);
extern void KerParReLU_fps(KerParReLUPool_fps_T *Arg);

/* One Feature map is evaluated in parallel on all cores.
   Feature maps of bytes (_fps) or half words (_fp)
*/
extern void KerReLU_fp(KerReLUPool_fp_T *Arg);
extern void KerReLU_fps(KerReLUPool_fps_T *Arg);


/******************************************************************************************************************************/
/**************** MAX/AVG POOLING WITH OPTIONAL ReLU **************************************************************************/
/******************************************************************************************************************************/
/*
        Performs Max or Average pooling followed by an optional linear rectification (ReLU).

        Zero padding is optional (Arg->Pad)

        Arg->DoReLU == 0        Max Pooling, No ReLU
        Arg->DoReLU == 1        Max Pooling then ReLU
        Arg->DoReLU == 2        Average Pooling, No ReLU
        Arg->DoReLU == 3        Average Pooling then ReLU
*/

/* Several output feature maps are evaluated in parallel, one output map per core
   Feature map is either half word (_fp) or byte (_fps)
   Pool 2x2 Stride 2 as a special case
*/
extern void KerParPool2x2Stride2_fp(KerParReLUPool_fp_T *Arg);
extern void KerParPoolNxNStrideS_fp(KerParReLUPool_fp_T *Arg);
extern void KerParPoolNxMStrideSxSy_fp(KerParReLUPool_fp_T *Arg);

extern void KerParPool2x2Stride2_fps(KerParReLUPool_fps_T *Arg);
extern void KerParPoolNxNStrideS_fps(KerParReLUPool_fps_T *Arg);
extern void KerParPoolNxMStrideSxSy_fps(KerParReLUPool_fps_T *Arg);

/* One output feature map is evaluated in parallel on all cores.
   Feature map is either half word (_fp) or byte (_fps)
   Pool 2x2 Stride 2 as a special case
*/
extern void KerPool2x2Stride2_fp(KerReLUPool_fp_T *Arg);
extern void KerPoolNxNStrideS_fp(KerReLUPool_fp_T *Arg);
extern void KerPoolNxMStrideSxSy_fp(KerReLUPool_fp_T *Arg);

extern void KerPool2x2Stride2_fps(KerReLUPool_fps_T *Arg);
extern void KerPoolNxNStrideS_fps(KerReLUPool_fps_T *Arg);
extern void KerPoolNxMStrideSxSy_fps(KerReLUPool_fps_T *Arg);

/******************************************************************************************************************************/
/**************** ADD + LINEAR  RECTIFICATION *********************************************************************************/
/******************************************************************************************************************************/
extern void KerAddReLU_fp(KerAddReLU_fp_T *Arg);
extern void KerAdd_fp    (KerAddReLU_fp_T *Arg);



/******************************************************************************************************************************/
/**************** FULLY CONNECTED *********************************************************************************************/
/******************************************************************************************************************************/

/*
        _fp             : Input, Bias, Filter and Output are half words
        _fps            : Input, Bias, Filter and Output are half bytes
        _fp_fps_fp      : Input, Bias and Output are half words, Filter is byte
        _fp_fp_fpd      : Input, Bias and Filter are half words, Output is word
*/

/* A single output is evaluated in parallel one all cores */
extern void KerLinearLayerReLU_fp(KerLinearLayerReLU_fp_T *Arg);
extern void KerLinearLayerReLU_fps(KerLinearLayerReLU_fps_T *Arg);
extern void KerLinearLayerReLU_fp_fps_fp(KerLinearLayerReLU_fp_fps_fp_T *Arg);
extern void KerLinearLayerReLU_fp_fp_fpd(KerLinearLayerReLU_fp_fp_fpd_T *Arg);

/* Several output are evaluated in parallel, one per core */
extern void KerParLinearLayerReLU_fp(KerLinearLayerReLU_fp_T *Arg);
extern void KerParLinearLayerReLU_fps(KerLinearLayerReLU_fps_T *Arg);
extern void KerParLinearLayerReLU_fp_fps_fp(KerLinearLayerReLU_fp_fps_fp_T *Arg);
extern void KerParLinearLayerReLU_fp_fp_fpd(KerLinearLayerReLU_fp_fp_fpd_T *Arg);


/******************************************************************************************************************************/
/******************* SOFT MAX *************************************************************************************************/
/******************************************************************************************************************************/

extern void KerParSoftMax_fp(KerParSoftMax_fp_T *Arg);
extern void KerParSoftMax_fps(KerParSoftMax_fps_T *Arg);


#endif

