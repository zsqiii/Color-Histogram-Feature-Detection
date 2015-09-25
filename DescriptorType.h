/*
Authors: Nick Huebner, Clark Olson, Siqi Zhang
DescriptorType.h

Enum and structure for managing the types of descriptors in a clean way
History log: 
	1. Descriptor name NEWSIFT was changed to COLOR_HIST_SIFT on 8/27/2015
	2. Adde a new type of descriptor: HUE_SAT_SIFT on 8/27/2015
*/

#ifndef DESCRIPTOR_TYPE_H
#define DESCRIPTOR_TYPE_H

#include <string>
using namespace std;

// Enum of the four basic descriptor types that have been implemented so far
// change NEW_DESCRIPTOR to COLOR_HIST_SIFT
enum DESC_TYPES {
	GRAY_SIFT, GRAY_SURF, OPPONENT_SIFT, COLOR_HIST_SIFT, HUE_SAT_SIFT, NONE
};

// A struct that holds the type of a descriptor
struct DescriptorType {
    bool doubleDescriptor;
    DESC_TYPES first;
    DESC_TYPES second;
    // Helper to convert from string types to enums types
    static DESC_TYPES convertHelper(string code)
    {
        if (code == "SIFT") {
            return GRAY_SIFT;
		}
		else if (code == "SURF") {
			return GRAY_SURF;
		}
		else if (code == "OPSIFT") {
            return OPPONENT_SIFT;
		} 
		else if (code == "CHSIFT") {
			return COLOR_HIST_SIFT;
		} 
		else if (code == "HSSIFT") {
			return HUE_SAT_SIFT;
		}
		else {
            return NONE;
        }
    }

    // Construct a descriptor type from a string code
    DescriptorType(string code)
    {
        first = convertHelper(code);
        doubleDescriptor = false;
		second = GRAY_SIFT;
    }
    // Default constructor for a descriptor type is grayscale sift
    DescriptorType() : doubleDescriptor(false), first(GRAY_SIFT), second(NONE) { }

    // Construct a descriptor type by explicitly specifying the enums
    DescriptorType(bool isDouble, DESC_TYPES firstType, DESC_TYPES secondType) : doubleDescriptor(isDouble), first(firstType), second(secondType) {	}

};

#endif
