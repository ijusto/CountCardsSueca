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

int H_MIN; int H_MAX; int S_MIN; int S_MAX; int V_MIN; int V_MAX;
cv::Mat originalImage;
int CHOICE; int PREVCHOICE; int c; int d; int u; int num;

void filterImage(){
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

    //H_MIN = 0; H_MAX = 495; S_MIN = 150; S_MAX = 491; V_MIN = 210; V_MAX = 400;
    cv::inRange(HSV, cv::Scalar(H_MIN,S_MIN,V_MIN), cv::Scalar(H_MAX,S_MAX,V_MAX), threshold);
    cv::namedWindow( "hsv image", cv::WINDOW_AUTOSIZE );
    cv::imshow( "hsv image", HSV );
    cv::namedWindow( "threshold image", cv::WINDOW_AUTOSIZE );
    cv::imshow( "threshold image", threshold );

}

void myChoice( int event, int x, int y, int flags, void *userdata )
{
    if( event == cv::EVENT_LBUTTONDOWN ) {
        //std::cout << "key: ";
        //std::cout << CHOICE << std::endl;
        //std::cout << "(char)key: ";
        //std::cout << (char)CHOICE << std::endl;
        /*
        switch( CHOICE )
        {
        case '0': case '1': case '2': case '3': case 4: case 5: case 6: case 7: case 8: case 9: H_MIN = CHOICE; break;
        default: H_MIN = 0; break;
        }
         */

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

        filterImage();
        std::cout << "H_MIN: ";
        std::cout << H_MIN;
        std::cout << ", H_MAX: ";
        std::cout << H_MAX;
        std::cout << ", S_MIN: ";
        std::cout << S_MIN;
        std::cout << ", S_MAX: ";
        std::cout << S_MAX;
        std::cout << ", V_MIN: ";
        std::cout << V_MIN;
        std::cout << ", V_MAX: ";
        std::cout << V_MAX;
        std::cout << "num: ";
        std::cout << num << std::endl;
    }
}

int charToInt(char c){
    int i;
    switch(c)
    {
        case '0' : i = 0; break; case '1' : i = 1; break; case '2' : i = 2; break; case '3' : i = 3; break;
        case '4' : i = 4; break; case '5' : i = 5; break; case '6' : i = 6; break; case '7' : i = 7; break;
        case '8' : i = 8; break; case '9' : i = 9; break; default: i = 0; break;
    }
    return i;
}

// MAIN

int main( int argc, char** argv ) {
    std::string fn;
    if (argc != 2) {
        std::cout << "The name of the image file is missing !!" << std::endl;
        fn = "IMG_20191219_184809.jpg";
    } else {
        fn = argv[1];
    }

    std::cout << std::endl;

    std::cout << "select number and click to process" << std::endl;

    std::cout << std::endl;

    originalImage = cv::imread(fn, cv::IMREAD_UNCHANGED);
    if (fn == "IMG_20191219_184809.jpg"){
        resize(originalImage, originalImage,
               cv::Size(originalImage.cols / 6, originalImage.rows / 6)); // to half size or even smaller
    }

    if (originalImage.empty()) {
        // NOT SUCCESSFUL : the data attribute is empty

        std::cout << "Image file could not be open !!" << std::endl;

        return -1;
    }

    H_MIN = 0; H_MAX = 256; S_MIN = 0; S_MAX = 256; V_MIN = 0; V_MAX = 256;
    filterImage();

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

        std::cout << std::endl;
        std::cout << "Q --- Quit" << std::endl;
        std::cout << "hi -- H_MIN; ha -- H_MAX; si -- S_MIN; sa -- S_MAX; i -- V_MIN; va -- V_MAX; " << std::endl;
        std::cout << "then write 3 digits for the value" << std::endl;

        PREVCHOICE = cv::waitKeyEx(0);
        std::cout << (char) PREVCHOICE;

        if( ((char)PREVCHOICE == 'Q') || ((char)PREVCHOICE == 'q') )
        {
            break;
        }

        CHOICE = cv::waitKeyEx(0);
        std::cout << (char) CHOICE << std::endl;

        if( ((char)CHOICE == 'Q') || ((char)CHOICE == 'q') )
        {
            break;
        }

        int c_tmp; int d_tmp; int u_tmp;
        c_tmp = cv::waitKeyEx(0);
        c = charToInt((char)c_tmp);
        std::cout << "c : " << c << std::endl;

        d_tmp = cv::waitKeyEx(0);
        d = charToInt((char)d_tmp);
        std::cout << "d : " << d << std::endl;

        u_tmp = cv::waitKeyEx(0);
        u = charToInt((char)u_tmp);
        std::cout << "u : " << u << std::endl;

        num = c * 100 + d * 10 + u;
        std::cout << "num : " << num << std::endl;

        std::cout << "confirm with mouse LEFT BUTTON" << std::endl;
    }
    // Destroy the windows --- Actually not needed in such a simple program

    cv::destroyAllWindows( );

    return 0;
}