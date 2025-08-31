// פאיכ Vision.h
#pragma once
#include <windows.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>


#include <opencv2/opencv.hpp>

#include <baseapi.h>
#include <leptonica/allheaders.h>    


using namespace std;
using namespace cv;


class Vision 
{
public:

    Vision();
    ~Vision();

    Mat CaptureScreen(int resol);
    void Show_Window(const Mat& screen);



    string RecognizeText(const Mat& tooltip);


    


};

