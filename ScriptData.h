/*
Authors: Nick Huebner, Clark Olson
A data structure that stores specifications for the program execution.
This structure can only be created from a file with the appropriate parameters.
*/

#ifndef SCRIPTDATA_H
#define SCRIPTDATA_H

#include "DescriptorType.h"

// C++ std includes
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

struct ScriptData {
    bool failed;
    string relativePath;
    int numImgs;
    bool saveData;
    string *imageNames;
    int numTypes;
    DescriptorType *types;
    bool homographyFlag;
    cv::Mat *homographies;

    // Construct the script data by parsing a file
    ScriptData(char *args[])
    { 
        failed = false;
        
		// Get relative path for images
		relativePath = args[1];
		cout << ">> Image path: " << relativePath << endl;

        // Get number of images to read
		numImgs = atoi(args[2]);
		imageNames = new string[numImgs];
		cout << ">> Num images: " << numImgs << endl;

		int argc = 3;
		for (int i = 0; i < numImgs; i++) {
			imageNames[i] = args[argc++];
			cout << ">> Image " << i << ": " << imageNames[i] << endl;
		}
		
        // Get the number of different types of descriptors to compute
        numTypes = atoi(args[argc++]);
		cout << ">> Num descriptors: " << numTypes << endl;

        types = new DescriptorType[numTypes];
        homographies = new cv::Mat[numImgs - 1];

        // Get the descriptor types
        for (int i = 0; i < numTypes; ++i) {
            // Construct descriptors types from the string code
			cout << ">> Descriptor " << i << ": " << args[argc] << endl; 
			types[i] = DescriptorType(args[argc++]);	
        }
     
		// read the homographies
		homographyFlag = true;
        if (homographyFlag) {
			for (int i = 0; i < numImgs - 1; i++) {
				// Read and construct the homography matrix for each image pair
				string homographyFile = args[argc++];
				cout << ">> Homography file: " << homographyFile << endl;
                double tmpArray[9];
				ifstream hFile(relativePath + homographyFile);
				if (!hFile.is_open()) {
					cout << "Unable to open homography file." << endl;
					failed = true;
					return;
				}
                for (int j = 0; j < 9; ++j) {
                    hFile >> tmpArray[j];
                }
				printf(">> Homography: %6.1f %6.1f %6.1f\n", tmpArray[0], tmpArray[1], tmpArray[2]);
				printf(">> Homography: %6.1f %6.1f %6.1f\n", tmpArray[3], tmpArray[4], tmpArray[5]);
				printf(">> Homography: %6.1f %6.1f %6.1f\n", tmpArray[6], tmpArray[7], tmpArray[8]);
                homographies[i] = (cv::Mat_<double>(3,3) << tmpArray[0], tmpArray[1], tmpArray[2], tmpArray[3], tmpArray[4], tmpArray[5], tmpArray[6], tmpArray[7], tmpArray[8]);
            }
        }
    }

    // Clean up allocated memory
    ~ScriptData()
    {
        if (!failed) {
            delete [] imageNames;
            delete [] types;
            delete [] homographies;
        }
    }
};

#endif
