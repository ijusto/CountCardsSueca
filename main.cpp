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

void filterImage(cv::Mat originalImage){
    // Learning from https://www.youtube.com/watch?v=bSeFrPrqZ2A
    // object tracking... the easy way
    //      -> color filtering
    //          - cvtColor() -> from bgr to hsv
    //          - inRange() -> between a max and a min
    //      -> contour finding
    //          - findContours()
    //          - moments method
    //
    // set 1:
    //      -> convert image from bgr to hsv
    //          (blue, green, red) -> (hue, saturation, value)
    //
    //initial min and max HSV filter values. these will be changed using trackbars
    int H_MIN = 0; int H_MAX = 256; int S_MIN = 0; int S_MAX = 256; int V_MIN = 0; int V_MAX = 256;
    //matrix storage for HSV image
    cv::Mat HSV;
    //matrix storage for binary threshold image
    cv::Mat threshold;
    // void cv::cvtColor(InputArray	src, OutputArray dst, int code, int	dstCn = 0)
    //convert img from BGR to HSV colorspace
    cv::cvtColor(originalImage, HSV, cv::COLOR_BGR2HSV);
    //filter HSV image between values and store filtered image to threshold matrix
    // set 2:
    //      -> filter the colors of interest between a min and max threshold

    H_MIN = 0; H_MAX = 495; S_MIN = 150; S_MAX = 491; V_MIN = 210; V_MAX = 400;
    cv::inRange(HSV, cv::Scalar(H_MIN,S_MIN,V_MIN), cv::Scalar(H_MAX,S_MAX,V_MAX), threshold);
    cv::namedWindow( "hsv image", cv::WINDOW_AUTOSIZE );
    cv::imshow( "hsv image", HSV );
    cv::namedWindow( "threshold image", cv::WINDOW_AUTOSIZE );
    cv::imshow( "threshold image", threshold );

}

cv::Mat originalImage;

void myChoice( int event, int x, int y, int flags, void *userdata )
{
    if( event == cv::EVENT_LBUTTONDOWN ) {
        filterImage(originalImage);
    }
}

int CHOICE; int PREVCHOICE; int c; int d; int u; int num;
int H_MIN = 0; int H_MAX = 256; int S_MIN = 0; int S_MAX = 256; int V_MIN = 0; int V_MAX = 256;

// MAIN

int main( int argc, char** argv ) {
    if (argc != 2) {
        std::cout << "The name of the image file is missing !!" << std::endl;

        return -1;
    }

    std::cout << std::endl;

    std::cout << "select number and click to process" << std::endl;

    std::cout << std::endl;

    originalImage = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    resize(originalImage, originalImage,
           cv::Size(originalImage.cols / 6, originalImage.rows / 6)); // to half size or even smaller

    if (originalImage.empty()) {
        // NOT SUCCESSFUL : the data attribute is empty

        std::cout << "Image file could not be open !!" << std::endl;

        return -1;
    }

    filterImage(originalImage);

    // Create window
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);

    // Display image
    cv::imshow("Original Image", originalImage);

    // Print some image features
    std::cout << "ORIGINAL IMAGE" << std::endl;

    printImageFeatures(originalImage);

    cv::Mat image_clone;
    image_clone = originalImage.clone();

    cv::Scalar colorImageClone;

    int startPointH = 0;
    int startPointW = 0;
    bool startHorLine = false;
    for (int h = 0; h < image_clone.size().height; h++) {
        for (int w = 0; w < image_clone.size().width; w++) {
            int r = image_clone.at<cv::Vec3b>(h, w)[0];
            int g = image_clone.at<cv::Vec3b>(h, w)[1];
            int b = image_clone.at<cv::Vec3b>(h, w)[2];
            if (r == 0 && g == 0 && b == 0) {
                if (!startHorLine) {
                    startHorLine = true;
                    startPointH = h;
                    startPointW = w;
                }
            } else {
                if (startHorLine) {
                    startHorLine = !startHorLine;
                    int endPointH = h;
                    int endPointW = w;
                    cv::line(image_clone, cv::Point(startPointW, startPointH), cv::Point(endPointW, endPointH),
                             colorImageClone);
                }
            }
        }
    }

    for (int w = 20; w < image_clone.size().width + 20; w += 20) {
        for (int h = 20; h < image_clone.size().height + 20; h += 20) {
            cv::line(image_clone, cv::Point(w - 20, h), cv::Point(w, h), colorImageClone);
            cv::line(image_clone, cv::Point(w, h - 20), cv::Point(w, h), colorImageClone);
        }
    }

    //cv::floodFill(originalImage, cv::Scalar(3, 3, 3), cv::Scalar(3, 3, 3 ));

    cv::namedWindow("new image", cv::WINDOW_AUTOSIZE);

    cv::imshow("new image", image_clone);

    cv::setMouseCallback( "threshold image", myChoice );

    // Processing keyboard events

    for( ; ; ){
        PREVCHOICE = cv::waitKey(0);
        CHOICE = cv::waitKey(0);
        c = cv::waitKey(0);
        d = cv::waitKey(0);
        u = cv::waitKey(0);
        num = c * 100 + d * 10 + u;

        if (((char) CHOICE == 'Q') || ((char) CHOICE == 'q')) {
            break;
        }

        if ((((char) PREVCHOICE == 'H') || ((char) PREVCHOICE == 'h')) &&
            (((char) CHOICE == 'I') || ((char) CHOICE == 'i'))) {
            H_MIN = num;
        } else if ((((char) PREVCHOICE == 'H') || ((char) PREVCHOICE == 'h')) &&
                   (((char) CHOICE == 'A') || ((char) CHOICE == 'a'))) {
            H_MAX = num;
        } else if ((((char) PREVCHOICE == 'S') || ((char) PREVCHOICE == 's')) &&
                   (((char) CHOICE == 'I') || ((char) CHOICE == 'i'))) {
            S_MIN = num;
        } else if ((((char) PREVCHOICE == 'S') || ((char) PREVCHOICE == 's')) &&
                   (((char) CHOICE == 'A') || ((char) CHOICE == 'a'))) {
            S_MAX = num;
        } else if ((((char) PREVCHOICE == 'V') || ((char) PREVCHOICE == 'v')) &&
                   (((char) CHOICE == 'I') || ((char) CHOICE == 'i'))) {
            V_MIN = num;
        } else if ((((char) PREVCHOICE == 'V') || ((char) PREVCHOICE == 'v')) &&
                   (((char) CHOICE == 'A') || ((char) CHOICE == 'a'))) {
            V_MAX = num;
        }


    }
    // Destroy the windows --- Actually not needed in such a simple program

    cv::destroyAllWindows( );

    return 0;
}