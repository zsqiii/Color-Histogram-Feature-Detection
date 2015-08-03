/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**********************************************************************************************\
Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/
/***********************************************************************************************
									Information
Author: Siqi Zhang, Zhiyuan Xiao, Ken Ohyama, Sandy Chau
Purpose: Spring 2015 CSS 487 programm 4, University of Washington Bothell
Due Date: 3rd June 2015
Description:
			Based on the NEWSIFT code, we changed the decriptor from 8 orientation histogram to
			color histogram with 2 buckets on each of the RGB color axies (8 buckets in total).
***********************************************************************************************/

#include "NewDescriptorExtractor.h"
#include <iostream>
#include <stdarg.h>
using namespace cv::xfeatures2d;
//CHANGE MATCHING FUNCTION?

namespace cv
{

	//CV_INIT_ALGORITHM(NEWSIFT, "Feature2D.NEWSIFT",  
	//	obj.info()->addParam(obj, "nFeatures", obj.nfeatures);
	//	obj.info()->addParam(obj, "nOctaveLayers", obj.nOctaveLayers);
	//	obj.info()->addParam(obj, "contrastThreshold", obj.contrastThreshold);
	//	obj.info()->addParam(obj, "edgeThreshold", obj.edgeThreshold);
	//	obj.info()->addParam(obj, "sigma", obj.sigma));

	/******************************* Defs and macros *****************************/

	// default number of sampled intervals per octave
	static const int NEWSIFT_INTVLS = 3;

	// default sigma for initial gaussian smoothing
	static const float NEWSIFT_SIGMA = 1.6f;

	// default threshold on keypoint contrast |D(x)|
	static const float NEWSIFT_CONTR_THR = 0.04f;

	// default threshold on keypoint ratio of principle curvatures
	static const float NEWSIFT_CURV_THR = 10.f;

	// double image size before pyramid construction?
	static const bool NEWSIFT_IMG_DBL = true;

	// default width of descriptor histogram array
	static const int NEWSIFT_DESCR_WIDTH = 4;

	// default number of bins per histogram in descriptor array
	static const int NEWSIFT_DESCR_HIST_BINS = 8;

	// assumed gaussian blur for input image
	static const float NEWSIFT_INIT_SIGMA = 0.5f;

	// width of border in which to ignore keypoints
	static const int NEWSIFT_IMG_BORDER = 5;

	// maximum steps of keypoint interpolation before failure
	static const int NEWSIFT_MAX_INTERP_STEPS = 5;

	// default number of bins in histogram for orientation assignment
	static const int NEWSIFT_ORI_HIST_BINS = 36;

	// determines gaussian sigma for orientation assignment
	static const float NEWSIFT_ORI_SIG_FCTR = 1.5f;

	// determines the radius of the region used in orientation assignment
	static const float NEWSIFT_ORI_RADIUS = 3 * NEWSIFT_ORI_SIG_FCTR;

	// orientation magnitude relative to max that results in new feature
	static const float NEWSIFT_ORI_PEAK_RATIO = 0.8f;

	// determines the size of a single descriptor orientation histogram
	static const float NEWSIFT_DESCR_SCL_FCTR = 3.f;

	// threshold on magnitude of elements of descriptor vector
	static const float NEWSIFT_DESCR_MAG_THR = 0.2f;

	// factor used to convert floating-point descriptor to unsigned char
	static const float NEWSIFT_INT_DESCR_FCTR = 512.f;
	
	//changes: add constant
	// number of buckets in each dimension for R G B
	static const int SIZE = 2;

	// range of color value of each bucket
	static const int BUCKET_SIZE = 256 / SIZE;

#if 0
	// intermediate type used for DoG pyramids
	typedef short NEWSIFT_wt;
	static const int NEWSIFT_FIXPT_SCALE = 48;
#else
	// intermediate type used for DoG pyramids
	typedef float NEWSIFT_wt;
	static const int NEWSIFT_FIXPT_SCALE = 1;
#endif

	static inline void
		unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
	{
		octave = kpt.octave & 255;
		layer = (kpt.octave >> 8) & 255;
		octave = octave < 128 ? octave : (-128 | octave);
		scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
	}

	static Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma)
	{
		Mat gray, gray_fpt;
		if (img.channels() == 3 || img.channels() == 4)
			cvtColor(img, gray, COLOR_BGR2GRAY);
		else
			img.copyTo(gray);
		gray.convertTo(gray_fpt, DataType<NEWSIFT_wt>::type, NEWSIFT_FIXPT_SCALE, 0);

		float sig_diff;

		if (doubleImageSize)
		{
			sig_diff = sqrtf(std::max(sigma * sigma - NEWSIFT_INIT_SIGMA * NEWSIFT_INIT_SIGMA * 4, 0.01f));
			Mat dbl;
			resize(gray_fpt, dbl, Size(gray.cols * 2, gray.rows * 2), 0, 0, INTER_LINEAR);
			GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
			return dbl;
		}
		else
		{
			sig_diff = sqrtf(std::max(sigma * sigma - NEWSIFT_INIT_SIGMA * NEWSIFT_INIT_SIGMA, 0.01f));
			GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
			return gray_fpt;
		}
	}
	// initialize color base image for calculating color gaussian pyramid
	static Mat createInitialColorImage(const Mat& img, bool doubleImageSize, float sigma)
	{
		Mat colorImg = img, color_fpt;
		img.convertTo(color_fpt, DataType<NEWSIFT_wt>::type, NEWSIFT_FIXPT_SCALE, 0);

		float sig_diff;

		if (doubleImageSize)
		{
			sig_diff = sqrtf(std::max(sigma * sigma - NEWSIFT_INIT_SIGMA * NEWSIFT_INIT_SIGMA * 4, 0.01f));
			Mat dbl;
			resize(color_fpt, dbl, Size(colorImg.cols * 2, colorImg.rows * 2), 0, 0, INTER_LINEAR);
			GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
			return dbl;
		}
		else
		{
			sig_diff = sqrtf(std::max(sigma * sigma - NEWSIFT_INIT_SIGMA * NEWSIFT_INIT_SIGMA, 0.01f));
			GaussianBlur(color_fpt, color_fpt, Size(), sig_diff, sig_diff);
			return color_fpt;
		}
	}

	void NEWSIFT::buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves) const
	{
		vector<double> sig(nOctaveLayers + 3);
		pyr.resize(nOctaves*(nOctaveLayers + 3));

		// precompute Gaussian sigmas using the following formula:
		//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
		sig[0] = sigma;
		double k = pow(2., 1. / nOctaveLayers);
		for (int i = 1; i < nOctaveLayers + 3; i++)
		{
			double sig_prev = pow(k, (double)(i - 1))*sigma;
			double sig_total = sig_prev*k;
			sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
		}

		for (int o = 0; o < nOctaves; o++)
		{
			for (int i = 0; i < nOctaveLayers + 3; i++)
			{
				Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
				if (o == 0 && i == 0)
					dst = base;
				// base of new octave is halved image from end of previous octave
				else if (i == 0)
				{
					const Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
					resize(src, dst, Size(src.cols / 2, src.rows / 2),
						0, 0, INTER_NEAREST);
				}
				else
				{
					const Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
					GaussianBlur(src, dst, Size(), sig[i], sig[i]);
				}
			}
		}
	}


	void NEWSIFT::buildDoGPyramid(const vector<Mat>& gpyr, vector<Mat>& dogpyr) const
	{
		int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
		dogpyr.resize(nOctaves*(nOctaveLayers + 2));

		for (int o = 0; o < nOctaves; o++)
		{
			for (int i = 0; i < nOctaveLayers + 2; i++)
			{
				const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
				const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
				Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
				subtract(src2, src1, dst, noArray(), DataType<NEWSIFT_wt>::type);
			}
		}
	}


	// Computes a gradient orientation histogram at a specified pixel
	static float calcOrientationHist(const Mat& img, Point pt, int radius,
		float sigma, float* hist, int n)
	{
		int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);

		float expf_scale = -1.f / (2.f * sigma * sigma);
		AutoBuffer<float> buf(len * 4 + n + 4);
		float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
		float* temphist = W + len + 2;

		for (i = 0; i < n; i++)
			temphist[i] = 0.f;

		for (i = -radius, k = 0; i <= radius; i++)
		{
			int y = pt.y + i;
			if (y <= 0 || y >= img.rows - 1)
				continue;
			for (j = -radius; j <= radius; j++)
			{
				int x = pt.x + j;
				if (x <= 0 || x >= img.cols - 1)
					continue;

				float dx = (float)(img.at<NEWSIFT_wt>(y, x + 1) - img.at<NEWSIFT_wt>(y, x - 1));
				float dy = (float)(img.at<NEWSIFT_wt>(y - 1, x) - img.at<NEWSIFT_wt>(y + 1, x));

				X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
				k++;
			}
		}

		len = k;

		// compute gradient values, orientations and the weights over the pixel neighborhood
		hal::exp(W, W, len);
		hal::fastAtan2(Y, X, Ori, len, true);
		hal::magnitude(X, Y, Mag, len);

		for (k = 0; k < len; k++)
		{
			int bin = cvRound((n / 360.f)*Ori[k]);
			if (bin >= n)
				bin -= n;
			if (bin < 0)
				bin += n;
			temphist[bin] += W[k] * Mag[k];
		}

		// smooth the histogram
		temphist[-1] = temphist[n - 1];
		temphist[-2] = temphist[n - 2];
		temphist[n] = temphist[0];
		temphist[n + 1] = temphist[1];
		for (i = 0; i < n; i++)
		{
			hist[i] = (temphist[i - 2] + temphist[i + 2])*(1.f / 16.f) +
				(temphist[i - 1] + temphist[i + 1])*(4.f / 16.f) +
				temphist[i] * (6.f / 16.f);
		}

		float maxval = hist[0];
		for (i = 1; i < n; i++)
			maxval = std::max(maxval, hist[i]);

		return maxval;
	}


	//
	// Interpolates a scale-space extremum's location and scale to subpixel
	// accuracy to form an image feature. Rejects features with low contrast.
	// Based on Section 4 of Lowe's paper.
	static bool adjustLocalExtrema(const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
		int& layer, int& r, int& c, int nOctaveLayers,
		float contrastThreshold, float edgeThreshold, float sigma)
	{
		const float img_scale = 1.f / (255 * NEWSIFT_FIXPT_SCALE);
		const float deriv_scale = img_scale*0.5f;
		const float second_deriv_scale = img_scale;
		const float cross_deriv_scale = img_scale*0.25f;

		float xi = 0, xr = 0, xc = 0, contr = 0;
		int i = 0;

		for (; i < NEWSIFT_MAX_INTERP_STEPS; i++)
		{
			int idx = octv*(nOctaveLayers + 2) + layer;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];

			Vec3f dD((img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1))*deriv_scale,
				(img.at<NEWSIFT_wt>(r + 1, c) - img.at<NEWSIFT_wt>(r - 1, c))*deriv_scale,
				(next.at<NEWSIFT_wt>(r, c) - prev.at<NEWSIFT_wt>(r, c))*deriv_scale);

			float v2 = (float)img.at<NEWSIFT_wt>(r, c) * 2;
			float dxx = (img.at<NEWSIFT_wt>(r, c + 1) + img.at<NEWSIFT_wt>(r, c - 1) - v2)*second_deriv_scale;
			float dyy = (img.at<NEWSIFT_wt>(r + 1, c) + img.at<NEWSIFT_wt>(r - 1, c) - v2)*second_deriv_scale;
			float dss = (next.at<NEWSIFT_wt>(r, c) + prev.at<NEWSIFT_wt>(r, c) - v2)*second_deriv_scale;
			float dxy = (img.at<NEWSIFT_wt>(r + 1, c + 1) - img.at<NEWSIFT_wt>(r + 1, c - 1) -
				img.at<NEWSIFT_wt>(r - 1, c + 1) + img.at<NEWSIFT_wt>(r - 1, c - 1))*cross_deriv_scale;
			float dxs = (next.at<NEWSIFT_wt>(r, c + 1) - next.at<NEWSIFT_wt>(r, c - 1) -
				prev.at<NEWSIFT_wt>(r, c + 1) + prev.at<NEWSIFT_wt>(r, c - 1))*cross_deriv_scale;
			float dys = (next.at<NEWSIFT_wt>(r + 1, c) - next.at<NEWSIFT_wt>(r - 1, c) -
				prev.at<NEWSIFT_wt>(r + 1, c) + prev.at<NEWSIFT_wt>(r - 1, c))*cross_deriv_scale;

			Matx33f H(dxx, dxy, dxs,
				dxy, dyy, dys,
				dxs, dys, dss);

			Vec3f X = H.solve(dD, DECOMP_LU);

			xi = -X[2];
			xr = -X[1];
			xc = -X[0];

			if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
				break;

			if (std::abs(xi) > (float)(INT_MAX / 3) ||
				std::abs(xr) > (float)(INT_MAX / 3) ||
				std::abs(xc) > (float)(INT_MAX / 3))
				return false;

			c += cvRound(xc);
			r += cvRound(xr);
			layer += cvRound(xi);

			if (layer < 1 || layer > nOctaveLayers ||
				c < NEWSIFT_IMG_BORDER || c >= img.cols - NEWSIFT_IMG_BORDER ||
				r < NEWSIFT_IMG_BORDER || r >= img.rows - NEWSIFT_IMG_BORDER)
				return false;
		}

		// ensure convergence of interpolation
		if (i >= NEWSIFT_MAX_INTERP_STEPS)
			return false;

		{
			int idx = octv*(nOctaveLayers + 2) + layer;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];
			Matx31f dD((img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1))*deriv_scale,
				(img.at<NEWSIFT_wt>(r + 1, c) - img.at<NEWSIFT_wt>(r - 1, c))*deriv_scale,
				(next.at<NEWSIFT_wt>(r, c) - prev.at<NEWSIFT_wt>(r, c))*deriv_scale);
			float t = dD.dot(Matx31f(xc, xr, xi));

			contr = img.at<NEWSIFT_wt>(r, c)*img_scale + t * 0.5f;
			if (std::abs(contr) * nOctaveLayers < contrastThreshold)
				return false;

			// principal curvatures are computed using the trace and det of Hessian
			float v2 = img.at<NEWSIFT_wt>(r, c)*2.f;
			float dxx = (img.at<NEWSIFT_wt>(r, c + 1) + img.at<NEWSIFT_wt>(r, c - 1) - v2)*second_deriv_scale;
			float dyy = (img.at<NEWSIFT_wt>(r + 1, c) + img.at<NEWSIFT_wt>(r - 1, c) - v2)*second_deriv_scale;
			float dxy = (img.at<NEWSIFT_wt>(r + 1, c + 1) - img.at<NEWSIFT_wt>(r + 1, c - 1) -
				img.at<NEWSIFT_wt>(r - 1, c + 1) + img.at<NEWSIFT_wt>(r - 1, c - 1)) * cross_deriv_scale;
			float tr = dxx + dyy;
			float det = dxx * dyy - dxy * dxy;

			if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
				return false;
		}

		kpt.pt.x = (c + xc) * (1 << octv);
		kpt.pt.y = (r + xr) * (1 << octv);
		kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
		kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
		kpt.response = std::abs(contr);

		return true;
	}


	//
	// Detects features at extrema in DoG scale space.  Bad features are discarded
	// based on contrast and ratio of principal curvatures.
	void NEWSIFT::findScaleSpaceExtrema(const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
		vector<KeyPoint>& keypoints) const
	{
		int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
		int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * NEWSIFT_FIXPT_SCALE);
		const int n = NEWSIFT_ORI_HIST_BINS;
		float hist[n];
		KeyPoint kpt;

		keypoints.clear();

		for (int o = 0; o < nOctaves; o++)
			for (int i = 1; i <= nOctaveLayers; i++)
			{
			int idx = o*(nOctaveLayers + 2) + i;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];
			int step = (int)img.step1();
			int rows = img.rows, cols = img.cols;

			for (int r = NEWSIFT_IMG_BORDER; r < rows - NEWSIFT_IMG_BORDER; r++)
			{
				const NEWSIFT_wt* currptr = img.ptr<NEWSIFT_wt>(r);
				const NEWSIFT_wt* prevptr = prev.ptr<NEWSIFT_wt>(r);
				const NEWSIFT_wt* nextptr = next.ptr<NEWSIFT_wt>(r);

				for (int c = NEWSIFT_IMG_BORDER; c < cols - NEWSIFT_IMG_BORDER; c++)
				{
					NEWSIFT_wt val = currptr[c];

					// find local extrema with pixel accuracy
					if (std::abs(val) > threshold &&
						((val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
						val >= currptr[c - step - 1] && val >= currptr[c - step] && val >= currptr[c - step + 1] &&
						val >= currptr[c + step - 1] && val >= currptr[c + step] && val >= currptr[c + step + 1] &&
						val >= nextptr[c] && val >= nextptr[c - 1] && val >= nextptr[c + 1] &&
						val >= nextptr[c - step - 1] && val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
						val >= nextptr[c + step - 1] && val >= nextptr[c + step] && val >= nextptr[c + step + 1] &&
						val >= prevptr[c] && val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
						val >= prevptr[c - step - 1] && val >= prevptr[c - step] && val >= prevptr[c - step + 1] &&
						val >= prevptr[c + step - 1] && val >= prevptr[c + step] && val >= prevptr[c + step + 1]) ||
						(val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
						val <= currptr[c - step - 1] && val <= currptr[c - step] && val <= currptr[c - step + 1] &&
						val <= currptr[c + step - 1] && val <= currptr[c + step] && val <= currptr[c + step + 1] &&
						val <= nextptr[c] && val <= nextptr[c - 1] && val <= nextptr[c + 1] &&
						val <= nextptr[c - step - 1] && val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
						val <= nextptr[c + step - 1] && val <= nextptr[c + step] && val <= nextptr[c + step + 1] &&
						val <= prevptr[c] && val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
						val <= prevptr[c - step - 1] && val <= prevptr[c - step] && val <= prevptr[c - step + 1] &&
						val <= prevptr[c + step - 1] && val <= prevptr[c + step] && val <= prevptr[c + step + 1])))
					{
						int r1 = r, c1 = c, layer = i;
						if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
							nOctaveLayers, (float)contrastThreshold,
							(float)edgeThreshold, (float)sigma))
							continue;
						float scl_octv = kpt.size*0.5f / (1 << o);
						float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers + 3) + layer],
							Point(c1, r1),
							cvRound(NEWSIFT_ORI_RADIUS * scl_octv),
							NEWSIFT_ORI_SIG_FCTR * scl_octv,
							hist, n);
						float mag_thr = (float)(omax * NEWSIFT_ORI_PEAK_RATIO);
						for (int j = 0; j < n; j++)
						{
							int l = j > 0 ? j - 1 : n - 1;
							int r2 = j < n - 1 ? j + 1 : 0;

							if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
							{
								float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
								bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
								kpt.angle = 360.f - (float)((360.f / n) * bin);
								if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
									kpt.angle = 0.f;
								keypoints.push_back(kpt);
							}
						}
					}
				}
			}
			}
	}
//-------------------------------------------------------------------------------------
	//img: color image
	//ptf: keypoint
	//ori: angle(degree) of the keypoint relative to the coordinates, clockwise
	//scl: radius of meaningful neighborhood around the keypoint 
	//d: newsift descr_width, 4 in this case
	//n: newsift_descr_hist_bins, 8 in this case
	//dst: descriptor array to pass in
	//changes: 1. img now is a color image
	//         2. change variable name from bins_per_rad to bins_per_degree
	static void calcNEWSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
		int d, int n, float* dst)
	{
		Mat greyImg;
		cvtColor(img, greyImg, CV_BGR2GRAY);
		Point pt(cvRound(ptf.x), cvRound(ptf.y));	//point object
		float cos_t = cosf(ori*(float)(CV_PI / 180));	
		float sin_t = sinf(ori*(float)(CV_PI / 180));
		float exp_scale = -1.f / (d * d * 0.5f);
		float hist_width = NEWSIFT_DESCR_SCL_FCTR * scl;
		int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
		// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
		radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));
		cos_t /= hist_width;
		sin_t /= hist_width;

		//Need to change length and histogram length
		int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
		int rows = img.rows, cols = img.cols;

		AutoBuffer<float> buf(len * 7 + histlen);
		float *RBin = buf, *CBin = RBin + len, *hist = CBin + len;
		//reserve memory for RGB value of all inclosed pixels
		float *RedBin = hist + len, *GreenBin = RedBin + len, *BlueBin = GreenBin + len;
		// reserver memory for all inclosed pixels
		float *colorBucketsIndex = BlueBin + len;
		//Vote for 8 color buckets
		for (i = 0; i < d + 2; i++)
		{
			for (j = 0; j < d + 2; j++)
				for (k = 0; k < n + 2; k++)
					hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
		}
		for (i = -radius, k = 0; i <= radius; i++)
			for (j = -radius; j <= radius; j++)
			{
			// Calculate sample's histogram array coords rotated relative to ori.
			// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			// r_rot = 1.5) have full weight placed in row 1 after interpolation.
			float c_rot = j * cos_t - i * sin_t;  // column after "rotation"
			float r_rot = j * sin_t + i * cos_t;  // row after "rotation"
			float rbin = r_rot + d / 2 - 0.5f;   // row index of 4x4 bin
			float cbin = c_rot + d / 2 - 0.5f;   // col index of 4x4 bin
			int r = pt.y + i, c = pt.x + j;      // row and column index in actual image
			//need a r array, g array , b array instead of X and Y.
			if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
				r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
				{
					RBin[k] = rbin; CBin[k] = cbin;
					//changes: color histogram
					//red
					float red = img.at<Vec3f>(r, c)[2];
					//green
					float green = img.at<Vec3f>(r, c)[1];
					//blue
					float blue = img.at<Vec3f>(r, c)[0];
					//stores RGB info
					RedBin[k] = red;
					GreenBin[k] = green;
					BlueBin[k] = blue;
					k++;
				}
			}

		len = k;
		//calculate which bucket the inclosed pixels belong to, and assign the index number
		//to colorBuckets
		// Bucket index:
					// 0 : 0 <= red <= 127; 0 <= green <= 127; 0 <= blue <= 127
					// 1 : 0 <= red <= 127; 0 <= green <= 127; 128 <= blue <= 255
					// 2 : 0 <= red <= 127; 128 <= green <= 255; 0 <= blue <= 127
					// 3 : 0 <= red <= 127; 128 <= green <= 255; 128 <= blue <= 255
					// 4 : 128 <= red <= 255; 0 <= green <= 127; 0 <= blue <= 127
					// 5 : 128 <= red <= 255; 0 <= green <= 127; 128 <= blue <= 255
					// 6 : 128 <= red <= 255; 128 <= green <= 255; 0 <= blue <= 127
					// 7 : 128 <= red <= 255; 128 <= green <= 255; 128 <= blue <= 255
		/*
			How this works:
				we have RGB(three dimensions) and each dimensions are divided into 2 parts.
				Thus, we can think about this in an abstract way: we represent R G B as a 3bits
				binary number(with R at most significantbit and B at least significant bit). 
				When a color value(any one of the RGB) falls into 0<= value <=127,we consider that as a 0 bit, 
				When a color value(any one of the RGB) falls into 127<= value <=255, we consider that as a 1 bit. 
				Now 3 bits (0-7) can be fully mapped to our 8 color buckets 
		*/
		for (int pixel = 0; pixel < k + 1; pixel++)
		{
			colorBucketsIndex[pixel] = 4 * (RedBin[pixel] / BUCKET_SIZE)
								     + 2 * (GreenBin[pixel] / BUCKET_SIZE)
									 + 1 * (BlueBin[pixel] / BUCKET_SIZE);
		}
		
		// going through all enclosed pixels and vote for bucket
		for (k = 0; k < len; k++)
		{
			float rbin = RBin[k], cbin = CBin[k];
			int binIndex = int(colorBucketsIndex[k]);
			// RGV color value for keypoint pixel k
			int red = RedBin[k];
			int blue = BlueBin[k];
			int green = GreenBin[k];
			// weight
			float redWeight = 1.0 - std::min(std::max((red - 63.5) / 128, 0.0), 1.0);
			float blueWeight = 1.0 - std::min(std::max((blue - 63.5) / 128, 0.0), 1.0);
			float greenWeight = 1.0 - std::min(std::max((green - 63.5) / 128, 0.0), 1.0);
			// weight container
			float rWeight[2] = {redWeight, (1.0 - redWeight)};
			float bWeight[2] = {blueWeight, (1.0 - blueWeight) };
			float gWeight[2] = {greenWeight, (1.0 - greenWeight) };

			int r0 = cvFloor(rbin);
			int c0 = cvFloor(cbin);
			rbin -= r0;
			cbin -= c0;

			//// histogram update using tri-linear interpolation
			// vote is weighted by 1 in colo histogram
			// (while vote is weighted by gradient magnitude in grediant orientation histogram)

			float v_r1 = 1*rbin, v_r0 = 1 - v_r1;
			float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
			float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01; 
			float v_rco110 = v_rc11;
			float v_rco100 = v_rc10;
			float v_rco010 = v_rc01;
			float v_rco000 = v_rc00;
			for (int red = 0; red < 2; red++) 
			{
				for (int green = 0; green < 2; green++) 
				{
					for (int blue = 0; blue < 2; blue++) 
					{
						//Voting
						int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + binIndex;
						float temp = v_rco000 * rWeight[red] * gWeight[green] * bWeight[blue];
						hist[idx] += v_rco000 * rWeight[red] * gWeight[green] * bWeight[blue];
						hist[idx + (n + 2)] += v_rco010 * rWeight[red] * gWeight[green] * bWeight[blue];
						hist[idx + (d + 2)*(n + 2)] += v_rco100 * rWeight[red] * gWeight[green] * bWeight[blue];
						hist[idx + (d + 3)*(n + 2)] += v_rco110 * rWeight[red] * gWeight[green] * bWeight[blue];
					}
				}
			}
		}
//-------------------------------------------------------------------------------------
		// finalize histogram, since the orientation histograms are circular fixes things 
		for (i = 0; i < d; i++)
			for (j = 0; j < d; j++)
			{
			int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
			hist[idx] += hist[idx + n];
			hist[idx + 1] += hist[idx + n + 1];
			for (k = 0; k < n; k++)
				dst[(i*d + j)*n + k] = hist[idx + k];
			}
		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array
		float nrm2 = 0;
		len = d*d*n;
		for (k = 0; k < len; k++)
			nrm2 += dst[k] * dst[k];
		float thr = std::sqrt(nrm2)*NEWSIFT_DESCR_MAG_THR;  //we didnt change the threshold
		for (i = 0, nrm2 = 0; i < k; i++)
		{
			float val = std::min(dst[i], thr);
			dst[i] = val;
			nrm2 += val*val;
		}
		nrm2 = NEWSIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
		for (k = 0; k < len; k++)
		{
			dst[k] = saturate_cast<uchar>(dst[k] * nrm2);
		}
#else
		float nrm1 = 0;
		for (k = 0; k < len; k++)
		{
			dst[k] *= nrm2;
			nrm1 += dst[k];
		}
		nrm1 = 1.f / std::max(nrm1, FLT_EPSILON);
		for (k = 0; k < len; k++)
		{
			dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*NEWSIFT_INT_DESCR_FCTR);
		}
#endif
	}

	//changes: change grey gaussian pyramid to a colorful one
	static void calcDescriptors(const vector<Mat>& colorGpyr, const vector<KeyPoint>& keypoints,
		Mat& descriptors, int nOctaveLayers, int firstOctave)
	{
		int d = NEWSIFT_DESCR_WIDTH, n = NEWSIFT_DESCR_HIST_BINS;

		for (size_t i = 0; i < keypoints.size(); i++)
		{
			KeyPoint kpt = keypoints[i];
			int octave, layer;
			float scale;
			unpackOctave(kpt, octave, layer, scale);
			CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
			float size = kpt.size*scale;        //
			Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
			const Mat& colorImg = colorGpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];
			float angle = 360.f - kpt.angle;
			if (std::abs(angle - 360.f) < FLT_EPSILON)
				angle = 0.f; 
			//changes: pass in the color image rather than grey image
			calcNEWSIFTDescriptor(colorImg, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
			//image, point being calculated, angle, Size, d = newsift descr_width, n = newsift_descr_hist_bins, all the descriptors
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////

	NEWSIFT::NEWSIFT(int _nfeatures, int _nOctaveLayers,
		double _contrastThreshold, double _edgeThreshold, double _sigma)
		: nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
		contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
	{
	}

	int NEWSIFT::descriptorSize() const
	{
		return NEWSIFT_DESCR_WIDTH*NEWSIFT_DESCR_WIDTH*NEWSIFT_DESCR_HIST_BINS;
	}

	int NEWSIFT::descriptorType() const
	{
		return CV_32F;
	}


	void NEWSIFT::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& keypoints) const
	{
		(*this)(_image, _mask, keypoints, noArray());
	}


	void NEWSIFT::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& keypoints,
		OutputArray _descriptors,
		bool useProvidedKeypoints) const
	{
		int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
		Mat image = _image.getMat(), mask = _mask.getMat();

		if (image.empty() || image.depth() != CV_8U)
			CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

		if (!mask.empty() && mask.type() != CV_8UC1)
			CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");

		if (useProvidedKeypoints)
		{
			firstOctave = 0;
			int maxOctave = INT_MIN;
			for (size_t i = 0; i < keypoints.size(); i++)
			{
				int octave, layer;
				float scale;
				unpackOctave(keypoints[i], octave, layer, scale);
				firstOctave = std::min(firstOctave, octave);
				maxOctave = std::max(maxOctave, octave);
				actualNLayers = std::max(actualNLayers, layer - 2);
			}

			firstOctave = std::min(firstOctave, 0);
			CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
			actualNOctaves = maxOctave - firstOctave + 1;
		}
		// base is a grey image
		Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
		//initialize color image
		Mat colorBase = createInitialColorImage(image, firstOctave < 0, (float)sigma);
		vector<Mat> gpyr, dogpyr, colorGpyr; // colorGpyr is a gaussian pyramid for color image
		int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

		//double t, tf = getTickFrequency();
		//t = (double)getTickCount();
		buildGaussianPyramid(base, gpyr, nOctaves);
		buildDoGPyramid(gpyr, dogpyr);
		// build color gaussian pyramid
		buildGaussianPyramid(colorBase, colorGpyr, nOctaves);
		//t = (double)getTickCount() - t;
		//printf("pyramid construction time: %g\n", t*1000./tf);

		if (!useProvidedKeypoints)
		{
			//t = (double)getTickCount();
			findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
			KeyPointsFilter::removeDuplicated(keypoints);

			if (nfeatures > 0)
				KeyPointsFilter::retainBest(keypoints, nfeatures);
			//t = (double)getTickCount() - t;
			//printf("keypoint detection time: %g\n", t*1000./tf);

			if (firstOctave < 0)
				for (size_t i = 0; i < keypoints.size(); i++)
				{
				KeyPoint& kpt = keypoints[i];
				float scale = 1.f / (float)(1 << -firstOctave);
				kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
				kpt.pt *= scale;
				kpt.size *= scale;
				}

			if (!mask.empty())
				KeyPointsFilter::runByPixelsMask(keypoints, mask);
		}
		else
		{
			// filter keypoints by mask
			//KeyPointsFilter::runByPixelsMask( keypoints, mask );
		}

		if (_descriptors.needed())
		{
			//t = (double)getTickCount();
			int dsize = descriptorSize();
			_descriptors.create((int)keypoints.size(), dsize, CV_32F);
			Mat descriptors = _descriptors.getMat();

			//Need to change this 
			//change: add color image
			calcDescriptors(colorGpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
			//t = (double)getTickCount() - t;
			//printf("descriptor extraction time: %g\n", t*1000./tf);
		}
	}

	void NEWSIFT::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
	{
		(*this)(image, mask, keypoints, noArray());
	}

	void NEWSIFT::computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		(*this)(image, Mat(), keypoints, descriptors, true);
	}

//------------------------------ compute --------------------------------------
// Compute the descriptor with a given color image and array of keypoints
// Preconditions:  1. keypoints and images are correctly formatted
//				   2. images and keypoints are at the same level of smotthing
// Postconditions: descritops are filled
//-----------------------------------------------------------------------------
	void NEWSIFT::compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors)
	{
		this->computeImpl(image, keypoints, descriptors);
	}
}
