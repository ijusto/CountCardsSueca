/*
 * Adapted from http://sweet.ua.pt/jmadeira/OpenCV/OpenCV_Guiao_03.zip J. Madeira - Dec 2012 + Nov 2017
 *
 */


// Visual Studio ONLY - Allowing for pre-compiled header files

// This has to be the first #include

// Remove it, if you are not using Windows and Visual Studio

//#include "stdafx.h"


#include <iostream>


#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"


// If you want to "simplify" code writing, you might want to use:

// using namespace cv;

// using namespace std;


// AUXILIARY  FUNCTION

void printImageFeatures( const cv::Mat &image )
{
    std::cout << std::endl;

    std::cout << "Number of rows : " << image.size().height << std::endl;

    std::cout << "Number of columns : " << image.size().width << std::endl;

    std::cout << "Number of channels : " << image.channels() << std::endl;

    std::cout << "Number of bytes per pixel : " << image.elemSize() << std::endl;

    std::cout << std::endl;
}

// machine learning
void ml(){
    //bool train(const Ptr<TrainData>& trainData, int flags=0);
    //bool train(InputArray samples, int layout, InputArray responses);
    //PTR<_Tp> train(const Ptr<TrainData>& data, const _Tp::Params& p, int flags=0);
    //Ptr<_Tp> train(InputArray samples, int layout, Input Array responses, const _Tp::Params& p, int flags=0);
}


// MAIN

int main( int argc, char** argv )
{
    if( argc != 2 )
    {
        std::cout << "The name of the image file is missing !!" << std::endl;

        return -1;
    }

    cv::Mat originalImage;

    originalImage = cv::imread( argv[1], cv::IMREAD_UNCHANGED );

    if( originalImage.empty() )
    {
        // NOT SUCCESSFUL : the data attribute is empty

        std::cout << "Image file could not be open !!" << std::endl;

        return -1;
    }

    if( originalImage.channels() > 1 )
    {
        // Convert to a single-channel, intensity image

        cv::cvtColor( originalImage, originalImage, cv::COLOR_BGR2GRAY, 1 );
    }

    // Create window

    cv::namedWindow( "Original Image", cv::WINDOW_AUTOSIZE );

    // Display image

    cv::imshow( "Original Image", originalImage );

    // Print some image features

    std::cout << "ORIGINAL IMAGE" << std::endl;

    printImageFeatures( originalImage );

    cv::Mat image_clone;
    image_clone = originalImage.clone();

    cv::Scalar colorImageClone;

    if (image_clone.channels() == 1) { // gray image
        colorImageClone = cv::Scalar(0, 255, 0);
    } else { // colored image
        colorImageClone = cv::Scalar(0, 255, 0);
    }

    int startPointH = 0;
    int startPointW = 0;
    bool startHorLine = false;
    for(int h = 0; h < image_clone.size().height; h++){
        for(int w = 0; w < image_clone.size().width; w++) {
            int r = image_clone.at<cv::Vec3b>(h,w)[0];
            int g = image_clone.at<cv::Vec3b>(h,w)[1];
            int b = image_clone.at<cv::Vec3b>(h,w)[2];
            if (r == 0 && g == 0 && b == 0){
                if(!startHorLine){
                    startHorLine = true;
                    startPointH = h;
                    startPointW = w;
                }
            }
            else {
                if(startHorLine){
                    startHorLine = !startHorLine;
                    int endPointH = h;
                    int endPointW = w;
                    cv::line(image_clone, cv::Point(startPointW, startPointH), cv::Point(endPointW, endPointH), colorImageClone);
                }
            }
        }
    }

    for(int w = 20; w < image_clone.size().width+20; w += 20){
        for(int h = 20; h < image_clone.size().height+20; h += 20){
            cv::line(image_clone, cv::Point(w - 20, h), cv::Point(w, h), colorImageClone);
            cv::line(image_clone, cv::Point(w, h - 20), cv::Point(w, h), colorImageClone);
        }
    }

    //cv::floodFill(originalImage, cv::Scalar(3, 3, 3), cv::Scalar(3, 3, 3 ));

    cv::namedWindow( "new image", cv::WINDOW_AUTOSIZE );

    cv::imshow( "new image", image_clone );

    // Waiting

    cv::waitKey( 0 );

    // Destroy the windows

    cv::destroyAllWindows();

    return 0;
}