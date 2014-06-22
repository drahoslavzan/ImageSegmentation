//===================================================================
// File:        main.cc
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


#include "generic.h"
#include "threshold.h"

#include <iostream>
#include <cstdlib>

#include <unistd.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#define TM_OTSU     1
#define TM_MTBC     2
#define TM_KITTLER  3


using namespace std;
using namespace cv;


extern char *optarg;
extern int optind, opterr, optopt;


static const char *progName;


void usage(ostream &stream, int ecode)
{
	stream << "USAGE: " << progName << " [OPTIONS] [METHOD] <image>" << endl;
	exit(ecode);
}

void help(ostream &stream, int ecode)
{
	stream << "USAGE: " << progName << " [OPTIONS] [METHOD] <image>" << endl;
	stream << endl;
	stream << "OPTIONS:" << endl;
	stream << "  -t n   Number of thresholds" << endl;
	stream << "  -o i   Segmented image output" << endl;
	stream << "  -c     Pseudo-colorize output" << endl;
	stream << "  -s     Semi-thresholding" << endl;
	stream << "  -h     Show this help and exit" << endl;
	stream << endl;
	stream << "METHODS:" << endl;
	stream << "  -" << TM_OTSU       << "     Otsu algorithm" << endl;
	stream << "  -" << TM_MTBC       << "     MTBC algorithm" << endl;
	stream << "  -" << TM_KITTLER    << "     Kittler algorithm" << endl;
	exit(ecode);
}

int main(int argc, char **argv)
{
	progName = argv[0];

	if(argc < 2)
		usage(cout, 0);

	size_t oT = 1;
	bool oC = false;
	bool oS = false;
	const char *oO = NULL;
	int oM = TM_OTSU;

	int opt;
	while((opt = getopt(argc, argv,
					TO_STR_REF1(TM_OTSU) TO_STR_REF1(TM_MTBC) TO_STR_REF1(TM_KITTLER)
					"t:cso:h")) != -1)
		switch(opt)
		{
			case TM_OTSU + '0':
				oM = TM_OTSU;
				break;
			case TM_MTBC + '0':
				oM = TM_MTBC;
				break;
			case TM_KITTLER + '0':
				oM = TM_KITTLER;
				break;
			case 't':
				oT = (atoi(optarg) < 1) ? 1 : atoi(optarg);
				break;
			case 'c':
				oC = true;
				break;
			case 's':
				oS = true;
				break;
			case 'o':
				oO = optarg;
				break;
			case 'h':
				help(cout, 0);
			default:
			case '?':
				usage(cerr, 1);
		}

	Mat img = imread(argv[optind], CV_LOAD_IMAGE_GRAYSCALE);

	if(img.empty())
	{
		cerr << "ERROR: Cannot open file " << argv[optind] << "'" << endl;
		return 1;
	}

	Threshold th;
 
	switch(oM)
	{
		case TM_MTBC:
			mtbc(img, oT, th);
			break;
		case TM_KITTLER:
			kittler(img, oT, th);
			break;
		case TM_OTSU:
		default:
			otsu(img, oT, th);
			break;
	}

	if(oO != NULL)
	{
		segment(img, th, oS);

		Mat imo = img;

		if(oC)
			imo = colorize(img);

		imwrite(oO, imo);
	}

	// Output

	for(size_t i = 0; i < th.size(); ++i)
		cout << th[i] << " ";

	cout << endl;

	return 0;
}

