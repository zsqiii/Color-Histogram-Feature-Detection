/*
Authors: Nick Huebner, Clark Olson

This class provides utilities for computing key points and different types of descriptors.
*/

#include "DescriptorUtil.h"
#include "NewDescriptorExtractor.h"
//#include <opencv2\nonfree\features2d.hpp>
#include "opencv2\xfeatures2d\nonfree.hpp"  //3.0 version
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv::xfeatures2d;

// Constructor, initializes parameters to be used for the keypoint detectors and descriptor extractors
DescriptorUtil::DescriptorUtil()
{
}

// Desctructor
DescriptorUtil::~DescriptorUtil()
{
}

// Detect features in an image using the SIFT feature detector. The keyPoints parameter will contain the key points detected
void DescriptorUtil::detectFeatures(const Mat& img, vector<KeyPoint> &keyPoints)
{
    // Initialize SIFT feature detector
	// opencv 2.x version
	//SiftFeatureDetector siftDetector;
    //siftDetector.detect(img, keyPoints);
	// opencv 3.0 version
	Ptr<SIFT> sift = SIFT::create();
	sift->detect(img, keyPoints);
}

// Reads key points from a file
vector<KeyPoint> DescriptorUtil::readKeyPoints(string filePath, string imgName)
{
    vector<KeyPoint> keyPoints;
    // Read features from file
    FileStorage fs(filePath, FileStorage::READ);
    read(fs[imgName], keyPoints);
    fs.release();
    return keyPoints;
}

// Writes key points to a file (.xml or .yml)
void DescriptorUtil::writeKeyPoints(vector<KeyPoint> *kpts, string *imgNames, int numImgs, string filename)
{
    // Write keypoints out to file
    FileStorage fs(filename, FileStorage::WRITE);
    for (int i = 0; i < numImgs; ++i) {
        size_t dotLocation = imgNames[i].rfind('.');
        write(fs, imgNames[i].substr(0, dotLocation), kpts[i]);
    }
    fs.release();
}

// Computes the descriptors of a specified type for an image, given a set of keypoints
Mat DescriptorUtil::computeDescriptors(Mat& img, vector<KeyPoint> &keypoints, DESC_TYPES type)
{
    Mat descriptors;
    vector<KeyPoint> kpts(keypoints.begin(), keypoints.end());
    

    // Lowe's SIFT Descriptor: descriptor size = 128
    if (type == GRAY_SIFT) {
        //descParams.recalculateAngles = true;
		//opencv 2.x version
        //SiftDescriptorExtractor siftExtractor;
		//siftExtractor.compute(img, kpts, descriptors);
		//opencv 3.0 version
		Ptr<SIFT> sift = SIFT::create();
		sift->compute(img, kpts, descriptors);
    }
	// New Descriptor: descriptor size = 128
	else if (type == NEW_DESCRIPTOR) {
		//opencv 2.x version
		//NewSiftDescriptorExtractor newSiftExtractor;
		//newSiftExtractor.compute(img, kpts, descriptors);
		//3.0 version
		Ptr<NEWSIFT> newsift = NEWSIFT::create();
		newsift->compute(img, kpts, descriptors);
	}
	// SURF Descriptor: descriptor size = 64
	else if (type == GRAY_SURF) {
		//SurfDescriptorExtractor surfExtractor;
		//surfExtractor.compute(img, kpts, descriptors);
	}
    // Opponent SIFT: descriptor size = 384	... WHY IS THIS CRASHING?
    else if (type == OPPONENT_SIFT) {
		Ptr<DescriptorExtractor> siftExtractor = new SiftDescriptorExtractor;
        //OpponentColorDescriptorExtractor opponentExtractor(siftExtractor);
        //opponentExtractor.compute(img, kpts, descriptors);
    }
	else if (type == NONE) { }

    return descriptors;
}

// Merge two descriptor types. There should be an equal number of descriptors in the matrices
Mat DescriptorUtil::mergeDescriptors(Mat& descr1, Mat& descr2)
{
    Mat descriptors(0, descr1.cols + descr2.cols, descr1.type());

    if (descr1.rows == descr2.rows) {
        vector<float> mergedRow;
        for (int i = 0; i < descr1.rows; ++i) {
            // Get the corresponding rows
            Mat m1 = descr1.row(i);
            Mat m2 = descr2.row(i);
            for (int j = 0; j < descr1.cols; ++j) {
                mergedRow.push_back(m1.at<float>(j));
            }
            for (int j = 0; j < descr2.cols; ++j) {
                mergedRow.push_back(m2.at<float>(j));
            }
            // Create a matrix
            Mat x(mergedRow);
            Mat y = x.t();
            descriptors.push_back(y);
            mergedRow.clear();
        }
    }

    return descriptors;
}

// Convert an image from BGR color space to opponent color space
vector<Mat> DescriptorUtil::convertToOpponentColor(const Mat &bgrImage)
{
    vector<Mat> opponentChannels;
    if( bgrImage.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "input image must be an BGR image of type CV_8UC3" );

    // Split image into BGR to allow conversion to Opponent Color Space.
    vector<Mat> bgrChannels(3);
    split(bgrImage, bgrChannels);

    // Prepare opponent color space storage matrices.
    opponentChannels.resize(3);
    opponentChannels[0] = cv::Mat(bgrImage.size(), CV_8UC1); // R-G RED-GREEN
    opponentChannels[1] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G-2B YELLOW-BLUE
    opponentChannels[2] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G+B

    // Calculate the channels of the opponent color space
    {
        // (R - G + 255) / 2
        MatConstIterator_<unsigned char> rIt = bgrChannels[2].begin<unsigned char>();
        MatConstIterator_<unsigned char> gIt = bgrChannels[1].begin<unsigned char>();
        MatConstIterator_<unsigned char> bIt = bgrChannels[0].begin<unsigned char>();
        MatIterator_<unsigned char> dstIt = opponentChannels[0].begin<unsigned char>();
        float factor = 1.0f / 2.0f;
        for( ; dstIt != opponentChannels[0].end<unsigned char>(); ++rIt, ++gIt, ++dstIt ) {
            int value = static_cast<int>((static_cast<float>(static_cast<int>(*rIt)-static_cast<int>(*gIt)) + 255) * factor);
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
    {
        // (R + G - 2B + 510) / 4
        MatConstIterator_<unsigned char> rIt = bgrChannels[2].begin<unsigned char>();
        MatConstIterator_<unsigned char> gIt = bgrChannels[1].begin<unsigned char>();
        MatConstIterator_<unsigned char> bIt = bgrChannels[0].begin<unsigned char>();
        MatIterator_<unsigned char> dstIt = opponentChannels[1].begin<unsigned char>();
        float factor = 1.0f / 4.0f;
        for( ; dstIt != opponentChannels[1].end<unsigned char>(); ++rIt, ++gIt, ++bIt, ++dstIt ) {
            int value = static_cast<int>( static_cast<float>((static_cast<int>(*rIt) + static_cast<int>(*gIt) - 2*static_cast<int>(*bIt)) + 510) * factor );
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
    {
        // (R + G + B) / 3
        MatConstIterator_<unsigned char> rIt = bgrChannels[2].begin<unsigned char>();
        MatConstIterator_<unsigned char> gIt = bgrChannels[1].begin<unsigned char>();
        MatConstIterator_<unsigned char> bIt = bgrChannels[0].begin<unsigned char>();
        MatIterator_<unsigned char> dstIt = opponentChannels[2].begin<unsigned char>();
        float factor = 1.0f / 3.0f;
        for( ; dstIt != opponentChannels[2].end<unsigned char>(); ++rIt, ++gIt, ++bIt, ++dstIt ) {
            int value = static_cast<int>((static_cast<float>(static_cast<int>(*rIt) + 
			  static_cast<int>(*gIt) + static_cast<int>(*bIt))) * factor);
            // The colors could be weighted differently. Below is how OpenCV weights RGB when it converts from color to grayscale:
            // int value = static_cast<int>(0.299f * static_cast<float>(*rIt) + 0.587f * static_cast<float>(*gIt) + 0.115f * static_cast<float>(*bIt));
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
    return opponentChannels;
}

// Reads descriptors from a file
Mat DescriptorUtil::readDescriptors(string filePath, string imgName)
{
    Mat descriptors;
    // Read descriptors from file
    FileStorage fs(filePath, FileStorage::READ);
    read(fs[imgName], descriptors);
    fs.release();
    return descriptors;
}

// Writes descriptors to a file (.xml or .yml)
void DescriptorUtil::writeDescriptors(Mat *&descriptors, string *imgNames, int numImgs, string filename)
{
    // Write descriptors out to file
    FileStorage fs(filename, FileStorage::WRITE);
    for (int i = 0; i < numImgs; ++i) {
        size_t dotLocation = imgNames[i].rfind('.');
        write(fs, imgNames[i].substr(0, dotLocation), descriptors[i]);
    }
    fs.release();
}

// Matches descriptors from two different images, evaluates the matches using the provided homography, and writes the results out to a file
void DescriptorUtil::match(const Mat &descr1, Mat &descr2, 
					  const vector<KeyPoint> &kpts1, const vector<KeyPoint> &kpts2, const Mat &img1, const Mat &img2, 
					  const Mat &homography, const string outFilename, bool drawMatches)
{
    // matching descriptors
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descr1, descr2, matches);
    {
        Point p1 = kpts1[matches[0].queryIdx].pt; // image 1 point
        Point p2 = kpts2[matches[0].trainIdx].pt; // image 2 point
    }
    int totalMatches = matches.size();
    sort(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2) {
        return m1.distance < m2.distance;
    });
    bool *correct = new bool[totalMatches];
    memset(correct, 0, totalMatches);

    int outBounds = 0;
    vector<char> matchesMask( totalMatches, 0 );
    for (int i = 0; i < totalMatches; ++i) {
        Point p1 = kpts1[matches[i].queryIdx].pt; // image 1 point
        Point p2 = kpts2[matches[i].trainIdx].pt; // image 2 point
        Mat p1Mat = (Mat_<double>(3, 1) << (double)p1.x, (double)p1.y, 1.0);
        Mat p2Mat = (Mat_<double>(3, 1) << (double)p2.x, (double)p2.y, 1.0);

        // if (norm(p2 - H * p1 / H.z)) < 2 * p2.size
        p1Mat = homography * p1Mat;
        p1Mat /= p1Mat.at<double>(2, 0);
        double x = p1Mat.at<double>(0, 0);
        double y = p1Mat.at<double>(1, 0);
        // Check for out of bounds
        if (x < 0 || x > img2.cols || y < 0 || y > img2.rows) {
            outBounds++;
        } else {
            Mat res = p2Mat - p1Mat;
            //cout << p1Mat << endl;
            //cout << p2Mat << endl;
            if (norm(res) < kpts2[matches[i].trainIdx].size) {
                correct[i] = true;
            }
            //cout << (correct[i] ? "good" : "bad") << endl;
            if (matches[i].distance < 275 && correct[i]) matchesMask[i] = 1;
        }
    }

    ofstream outFile(outFilename.c_str());

    stringstream s;
    s << totalMatches << "\t"  << (kpts1.size() - outBounds) << endl;

    const int numTiers = 28;
    int DISTANCES[numTiers] = {10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 
	    150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 700, 
	    800, 900, 1000};

    // Find precision and recall values at matches above various tiers of quality
    int numCorrect = 0;
    int j = 0;
    for (int i = 0; i < numTiers; ++i) {
        while (j < totalMatches && matches[j].distance < DISTANCES[i]) {
            if (correct[j]) {
                ++numCorrect;
            }
            ++j;
        }

        s << numCorrect << "\t" << j << endl;
        outFile << s.str().c_str();
        s.str("");
    }
    outFile.close();

    // drawing the results
    if (drawMatches) {
        namedWindow("Match Results", 1);
        Mat img_matches;

        cv::drawMatches(img1, kpts1, img2, kpts2, matches, img_matches,
                        Scalar::all(-1), Scalar::all(-1), matchesMask);
        imshow("Match Results", img_matches);
        imwrite("matches.jpg", img_matches);
        waitKey(0);
        destroyWindow("Match Results");
    }

    delete [] correct;
}
