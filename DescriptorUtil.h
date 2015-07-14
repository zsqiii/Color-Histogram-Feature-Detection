/*
Authors: Nick Huebner, Clark Olson
DescriptorUtil.h

This class provides utilities for computing key points and different types of descriptors.
*/

#ifndef SIFTPROGRAM_H
#define SIFTPROGRAM_H

#include "DescriptorType.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class DescriptorUtil
{
public:
    // Constructor, initializes parameters to be used for the keypoint detectors and descriptor extractors
    DescriptorUtil();
    // Destructor
    ~DescriptorUtil();

    // Detect features in an image using the SIFT feature detector. The keyPoints parameter will contain the key points detected
    void detectFeatures(const Mat& img, vector<KeyPoint> &keyPoints);

    // Reads key points from a file
    vector<KeyPoint> readKeyPoints(string filePath, string imgName);

    // Writes key points to a file (.xml or .yml)
    void writeKeyPoints(vector<KeyPoint> *kpts, string *imgNames, int numImgs, string filename);

    // Computes the descriptors of a specified type for an image, given a set of keypoints
    Mat computeDescriptors(Mat& img, vector<KeyPoint> &kpts, DESC_TYPES type);

    // Merge two descriptor types. There should be an equal number of descriptors in the matrices
    Mat mergeDescriptors(Mat& descr1, Mat& descr2);

    // Convert an image from BGR color space to opponent color space
    vector<Mat> convertToOpponentColor(const Mat &bgrImage);

    // Reads descriptors from a file
    Mat readDescriptors(string filePath, string imgName);

    // Writes descriptors to a file (.xml or .yml)
    void writeDescriptors(Mat *&descriptors, string *imgNames, int numImgs, string filename);

    // Matches descriptors from two different images, evaluates the matches using the provided homography, and writes the results out to a file
    void match(const Mat &descr1, Mat &descr2, const vector<KeyPoint> &kpts1, const vector<KeyPoint> &kpts2, const Mat &img1, const Mat &img2, const Mat &homography, const string outFilename, bool drawMatches = false);

};

#endif
