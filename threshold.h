//===================================================================
// File:        threshold.h
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


#ifndef _THRESHOLD_H_
#define _THRESHOLD_H_


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>


const size_t NGRAY = 256;

typedef std::vector<unsigned> Threshold;


// ==================================================================
// Algorithms
// ==================================================================

// ------------------------------------------------------------------
// A Fast Algorithm for Multilevel Thresholding
// PING-SUNG LIAO, TSE-SHENG CHEN AND PAU-CHOO CHUNG
// URL: http://www.iis.sinica.edu.tw/page/jise/2001/200109_01.pdf
// ------------------------------------------------------------------
// Multi Otsu Threshold
// URL: http://fiji.sc/wiki/index.php/Multi_Otsu_Threshold
// ------------------------------------------------------------------
void otsu(cv::Mat gray, size_t n, Threshold &th);

// ------------------------------------------------------------------
// An Efficient Algorithm for Optimal Multilevel
// Thresholding of Irregularly Sampled Histograms
// Luis Rueda
// URL: http://cs.uwindsor.ca/~lrueda/papers/PolyThresSPR2008.pdf
// ------------------------------------------------------------------
void mtbc(cv::Mat gray, size_t n, Threshold &th);

// ------------------------------------------------------------------
// Minimum Error Thresholding
// J. Kittler, J. Illingworth
// URL: http://en.pudn.com/downloads159/sourcecode/graph/texture_mapping/detail715575_en.html
// ------------------------------------------------------------------
void kittler(cv::Mat gray, size_t n, Threshold &th);


// ==================================================================
// Additional
// ==================================================================

void segment(cv::Mat gray, Threshold &th, bool semi = false);

cv::Mat colorize(cv::Mat gray);


#endif /* _THRESHOLD_H_ */
