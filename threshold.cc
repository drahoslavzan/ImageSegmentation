//===================================================================
// File:        threshold.cc
// Author:      Drahoslav Zan
// Email:       izan@fit.vutbr.cz
// Affiliation: Brno University of Technology,
//              Faculty of Information Technology
// Date:        Sun Apr 21 15:51:17 CET 2013
// Project:     Image Segmentation using Histogram Analysis (ISHA)
//-------------------------------------------------------------------
// Copyright (C) 2013 Drahoslav Zan
//
// This file is part of ISHA.
//
// ISHA is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ISHA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ISHA. If not, see <http://www.gnu.org/licenses/>.
//===================================================================
// vim: set nowrap sw=2 ts=2


#include "threshold.h"

#include <cassert>
#include <limits>
#include <cmath>
#include <map>


using namespace cv;


static Mat histogram(Mat gray)
{
	Mat h = Mat::zeros(1, NGRAY, CV_32FC1);

	for(int r = 0; r < gray.rows; ++r)
		for(int c = 0; c < gray.cols; ++c)
			h.at<float>(gray.at<unsigned char>(r, c))++;

	return h;
}

static Mat histogramNormalized(Mat gray)
{
	Mat h = histogram(gray);

	float sz = gray.rows * gray.cols;

	for(size_t i = 0; i < NGRAY; ++i)
		h.at<float>(i) /= sz;

	return h;
}


// =====================================================================
// Algorithms
// =====================================================================

// Otsu

float otsuRecursive(Mat lh, size_t n, Threshold &th,
		size_t s = 1, float tm = 0, float sum = 0)
{
	size_t m = n + 1;

	if(n == 1)
	{
		for(size_t i = s; i < NGRAY - m; ++i)
		{
			float t = sum + lh.at<float>(s, i) + lh.at<float>(i + 1, NGRAY - 1);

			if(t > tm)
			{
				tm = t;
				th[n - 1] = i;
			}
		}
	}
	else
	{
		for(size_t i = s; i < NGRAY - m; ++i)
		{
			float t = otsuRecursive(lh, n - 1, th, i + 1, tm, sum + lh.at<float>(s, i));

			if(t > tm)
			{
				tm = t;
				th[n - 1] = i;
			}
		}
	}

	return tm;
}

void otsu(Mat gray, size_t n, Threshold &th)
{
	assert(gray.channels() == 1);

	Mat h = histogramNormalized(gray);

	Mat lp = Mat::zeros(NGRAY, NGRAY, CV_32FC1);
	Mat ls = Mat::zeros(NGRAY, NGRAY, CV_32FC1);
	Mat lh = Mat::zeros(NGRAY, NGRAY, CV_32FC1);

	// Build lookup tables

	for(size_t i = 0; i < NGRAY; ++i)
	{
		lp.at<float>(i, i) = h.at<float>(i);
		ls.at<float>(i, i) = i * h.at<float>(i);
	}

	for(size_t i = 1; i < NGRAY - 1; ++i)
	{
		lp.at<float>(1, i + 1) = lp.at<float>(1, i) + h.at<float>(i + 1);
		ls.at<float>(1, i + 1) = ls.at<float>(1, i) + (i + 1) * h.at<float>(i + 1);
	}

	for(size_t i = 2; i < NGRAY; ++i)
		for(size_t j = i + 1; j < NGRAY; ++j)
		{
			lp.at<float>(i, j) = lp.at<float>(1, j) - lp.at<float>(1, i - 1);
			ls.at<float>(i, j) = ls.at<float>(1, j) - ls.at<float>(1, i - 1);
		}

	for(size_t i = 1; i < NGRAY; ++i)
		for(size_t j = i + 1; j < NGRAY; ++j)
			if(lp.at<float>(i, j))
			{
				float u = ls.at<float>(i, j);
				float v = lp.at<float>(i, j);

				lh.at<float>(i, j) = u * u / v;
			}

	// Compute

	th.clear();

	th.resize(n);

	otsuRecursive(lh, n, th);

	// Reverse vector

	sort(th.begin(), th.end());
}


// MTBC

void findThresholdRanges(size_t n,
		std::vector<size_t> &minTj, std::vector<size_t> &maxTj)
{
	size_t m = n + 1;

	minTj.resize(m + 1);
	maxTj.resize(m + 1);

	minTj[0] = 0;
	minTj[m] = (NGRAY - 1);
	maxTj[0] = 0;
	maxTj[m] = (NGRAY - 1);

	for(size_t i = 1; i < m; ++i)
	{
		minTj[i] = i;
		maxTj[i] = (NGRAY - 1) - n + i - 1;
	}
}

void findThresholds(Mat ld, size_t n, Threshold &th)
{
	th.resize(n + 1);

	th[n] = NGRAY - 1;

	for(size_t j = n; j > 0; --j)
		th[j - 1] = ld.at<unsigned char>(th[j], j + 1);

	th.pop_back();
}

void mtbc(Mat gray, size_t n, Threshold &th)
{
	assert(gray.channels() == 1);

	Mat h = histogram(gray);

	float hsum = 0;
	for(size_t i = 0; i < NGRAY; ++i)
		hsum += h.at<float>(i);

	for(size_t i = 0; i < NGRAY; ++i)
		h.at<float>(i) /= hsum;

	std::vector<size_t> minTj, maxTj;

	findThresholdRanges(n, minTj, maxTj);

	size_t m = n + 1;

	Mat lc = Mat::zeros(NGRAY, m + 1, CV_32FC1);
	Mat ld = Mat::zeros(NGRAY, m + 1, CV_8UC1);

	lc.at<float>(0, 0) = 0;
	ld.at<unsigned char>(0, 0) = 0;

	for(size_t j = 1; j <= m; ++j)
		for(size_t t = minTj[j]; t <= maxTj[j]; ++t)
		{
			lc.at<float>(t, j) = 0;

			float u = 0;
			float w = 0;
			size_t hi = j;

			for(size_t i = j; i <= t; ++i)
			{
				w += h.at<float>(i);
				u += i * h.at<float>(i);
			}

			float psi = (1 / w) * u * u;

			for(size_t i = minTj[j - 1]; i <= MIN(maxTj[j - 1], t - 1); ++i)
			{
				if(lc.at<float>(i, j - 1) + psi > lc.at<float>(t, j))
				{
					lc.at<float>(t, j) = lc.at<float>(i, j - 1) + psi;
					ld.at<unsigned char>(t, j) = i;
				}

				w -= h.at<float>(hi);
				u -= hi * h.at<float>(hi);

				psi = (1 / w) * u * u;

				++hi;
			}
		}

	findThresholds(ld, n, th);
}

// Kittler

unsigned optimize(Mat h, float mu)
{
	size_t f = NGRAY - 1, l = NGRAY - 1;

	for(size_t i = 1; i < NGRAY - 1; ++i)
		if(h.at<float>(i) && h.at<float>(i + 1))
		{
			if(f > i)
				f = i;
			l = i;
		}

	double q1[NGRAY - 1], q2[NGRAY - 1], mu1[NGRAY - 1], mu2[NGRAY - 1], var1[NGRAY - 1], var2[NGRAY - 1];

	q1[f] = h.at<float>(f);
	q2[f] = 1 - q1[f];
	mu1[f] = f;
	mu2[f] = (mu - mu1[f] * q1[f]) / q2[f];
	var1[f] = (f - mu1[f]) * (f - mu1[f]) * h.at<float>(f) / q1[f];

	var2[f] = 0;
	for(size_t i = f; i < l; ++i)
		var2[f] += (i - mu2[f]) * (i - mu2[f]) * h.at<float>(i) / q2[f];

	double H[NGRAY - 1];

	for(size_t i = f + 1; i < l; ++i)
	{
		q1[i] = q1[i - 1] + h.at<float>(i);
		q2[i] = 1- q1[i];
		mu1[i] = (q1[i - 1] * mu1[i - 1] + (double)i * h.at<float>(i)) / q1[i];
		mu2[i] = (mu - q1[i] * mu1[i]) / q2[i];

		var1[i] =(q1[i - 1] * (var1[i - 1] + (mu1[i - 1] - mu1[i]) * (mu1[i - 1] - mu1[i]))
				+ h.at<float>(i) * (i - mu1[i]) * (i - mu1[i])) / q1[i];
		var2[i] =(q2[i - 1] * (var2[i - 1] + (mu2[i - 1] - mu2[i]) * (mu2[i - 1] - mu2[i]))
				- h.at<float>(i) * (i - mu2[i]) * (i - mu2[i])) / q2[i];
		
		H[i] = (q1[i] * log(var1[i]) + q2[i] * log(var2[i])) / 2 - q1[i] * log(q1[i])
				- q2[i] * log(q2[i]);
	}

	size_t t = 0;
	double min = std::numeric_limits<double>::max();

	for(size_t i = f + 1; i < l - 1; ++i)
		if(H[i] < min)
		{
			min = H[i];
			t = i;
		}

	return t;
}

void kittler(Mat gray, size_t, Threshold &th)
{
	assert(gray.channels() == 1);

	Mat h = histogramNormalized(gray);

	h.at<float>(NGRAY - 1) = 0;

	float mu = 0;
	for(size_t i = 0; i < NGRAY - 1; ++i)
		mu += i * h.at<float>(i);

	th.clear();

	th.push_back(optimize(h, mu));
}

// =====================================================================
// Segmentation
// =====================================================================

void segment(Mat gray, Threshold &th, bool semi)
{
	assert(gray.channels() == 1);

	//sort(th.begin(), th.end());

	if(!semi)
		th.push_back(NGRAY - 1);

	for(int r = 0; r < gray.rows; ++r)
		for(int c = 0; c < gray.cols; ++c)
			for(size_t i = 0; i < th.size(); ++i)
				if(gray.at<unsigned char>(r, c) < th[i])
				{
					gray.at<unsigned char>(r, c) = i * float(NGRAY - 1) / (th.size() - 1);
					break;
				}

	if(!semi)
		th.pop_back();
}

Mat colorize(Mat gray)
{
	assert(gray.channels() == 1);

	Mat img = Mat(gray.rows, gray.cols, CV_8UC3);
	cvtColor(gray, img, CV_GRAY2BGR);

	Mat h = histogram(gray);

	std::map<unsigned char, Vec3b> clr;

	RNG rng(time(NULL));

	for(size_t i = 0; i < NGRAY; ++i)
	{
		Vec3b v(rng.uniform(0., 1.) * NGRAY,
				rng.uniform(0., 1.) * NGRAY, rng.uniform(0., 1.) * NGRAY);

		if(h.at<float>(i))
			clr[i] = v;
	}

	for(int r = 0; r < img.rows; ++r)
		for(int c = 0; c < img.cols; ++c)
		{
			Vec3b &ic = img.at<Vec3b>(r, c);

			ic = clr[ic[0]];
		}

	return img;
}
