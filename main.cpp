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

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 1*1;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH;
int H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX;
cv::Mat originalImage;
cv::Mat image_clone;

int CHOICE, PREVCHOICE, c, d, u, num;

// from https://www.youtube.com/watch?v=bSeFrPrqZ2A
std::string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

// adapted from https://www.youtube.com/watch?v=bSeFrPrqZ2A
void drawObject(int x, int y, cv::Mat &frame){

    //use some of the openCV drawing functions to draw crosshairs
    //on your tracked image!

    //UPDATE:JUNE 18TH, 2013
    //added 'if' and 'else' statements to prevent
    //memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

    cv::circle(frame, cv::Point(x,y), 20, cv::Scalar(0,255,0),2);
    if(y - 25 > 0)
        cv::line(frame, cv::Point(x,y), cv::Point(x,y-25), cv::Scalar(0,255,0),2);
    else cv::line(frame, cv::Point(x,y), cv::Point(x,0), cv::Scalar(0,255,0),2);
    if(y + 25 < FRAME_HEIGHT)
        cv::line(frame, cv::Point(x,y), cv::Point(x,y + 25), cv::Scalar(0,255,0),2);
    else cv::line(frame, cv::Point(x,y), cv::Point(x, FRAME_HEIGHT), cv::Scalar(0,255,0),2);
    if(x - 25 > 0)
        cv::line(frame, cv::Point(x,y), cv::Point(x - 25, y), cv::Scalar(0,255,0),2);
    else cv::line(frame, cv::Point(x,y), cv::Point(0, y), cv::Scalar(0,255,0),2);
    if(x + 25 < FRAME_WIDTH)
        cv::line(frame, cv::Point(x,y), cv::Point(x + 25, y), cv::Scalar(0,255,0),2);
    else cv::line(frame, cv::Point(x,y), cv::Point(FRAME_WIDTH, y), cv::Scalar(0,255,0),2);

    putText(frame, intToString(x) + "," + intToString(y), cv::Point(x,y + 30), 1, 1, cv::Scalar(0,255,0),2);
}

// adapted from https://www.youtube.com/watch?v=bSeFrPrqZ2A
void trackFilteredObject(int &x, int &y, cv::Mat& threshold, cv::Mat &oi){
    cv::Mat temp;
    threshold.copyTo(temp);
    //these two vectors needed for output of findContours
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    cv::findContours(temp, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );

    //use moments method to find our filtered object
    double refArea = 0;
    bool objectFound = false;
    if (!hierarchy.empty()) {
        int numObjects = hierarchy.size();
        //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
        if(numObjects < MAX_NUM_OBJECTS){
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {

                cv::Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;

                //if the area is less than 20 px by 20px then it is probably just noise
                //if the area is the same as the 3/2 of the image size, probably just a bad filter
                //we only want the object with the largest area so we safe a reference area each
                //iteration and compare it to the area in the next iteration.
                x = (int)(moment.m10/area);
                y = (int)(moment.m01/area);
                putText(oi,"Tracking Object", cv::Point(0,50),2,1, cv::Scalar(0,255,0),2);
                //draw object location on screen
                drawObject(x, y, oi);
            }

        }else putText(oi,"TOO MUCH NOISE! ADJUST FILTER", cv::Point(0,50),1,2, cv::Scalar(0,0,255),2);
    }
}

// adapted from https://www.youtube.com/watch?v=bSeFrPrqZ2A
void morphOps(cv::Mat &thresh){
    // create structuring element that will be used to "dilate" and "erode" image.
    // the element choosen here is a 3px by 3px rectangle
    cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

    cv::erode(thresh, thresh, erodeElement);
    cv::erode(thresh, thresh, erodeElement);
    cv::erode(thresh, thresh, erodeElement);
    cv::erode(thresh, thresh, erodeElement);

    cv::dilate(thresh, thresh, erodeElement);
    cv::dilate(thresh, thresh, erodeElement);
}

void filterImage();

void on_trackbar( int, void* )
{//This function gets called whenever a
    // trackbar position is changed
    filterImage();
}

// adapted from https://www.youtube.com/watch?v=bSeFrPrqZ2A
void createTrackbars(){
    //create window for trackbars
    cv::namedWindow("Trackbars",cv::WINDOW_AUTOSIZE);

    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf( TrackbarName, "H_MIN");
    sprintf( TrackbarName, "H_MAX");
    sprintf( TrackbarName, "S_MIN");
    sprintf( TrackbarName, "S_MAX");
    sprintf( TrackbarName, "V_MIN");
    sprintf( TrackbarName, "V_MAX");
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH),
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->
    //create Trackbars
    cv::createTrackbar( "H_MIN", "Trackbars", &H_MIN, 500, on_trackbar );
    cv::createTrackbar( "H_MAX", "Trackbars", &H_MAX, 500, on_trackbar );
    cv::createTrackbar( "S_MIN", "Trackbars", &S_MIN, 500, on_trackbar );
    cv::createTrackbar( "S_MAX", "Trackbars", &S_MAX, 500, on_trackbar );
    cv::createTrackbar( "V_MIN", "Trackbars", &V_MIN, 500, on_trackbar );
    cv::createTrackbar( "V_MAX", "Trackbars", &V_MAX, 500, on_trackbar );
}

void filterImage(){
    // Learning and adapting from https://www.youtube.com/watch?v=bSeFrPrqZ2A
    // object tracking... the easy way
    //      -> color filtering
    //          - cvtColor() -> from bgr to hsv
    //          - inRange() -> between a max and a min
    //      -> contour finding
    //          - findContours()
    //          - moments method
    //
    // step 1:
    //      -> convert image from bgr to hsv
    //          (blue, green, red) -> (hue, saturation, value)
    //
    //initial min and max HSV filter values. these will be changed using trackbars
    //matrix storage for HSV image
    cv::Mat HSV;
    //matrix storage for binary threshold image
    cv::Mat threshold;
    //x and y values for the location of the object
    int x = 0, y = 0;
    // void cv::cvtColor(InputArray	src, OutputArray dst, int code, int	dstCn = 0)
    //convert img from BGR to HSV colorspace
    cv::cvtColor(originalImage, HSV, cv::COLOR_BGR2HSV);
    //filter HSV image between values and store filtered image to threshold matrix
    // step 2:
    //      -> filter the colors of interest between a min and max threshold

    //H_MIN = 0; H_MAX = 495; S_MIN = 150; S_MAX = 491; V_MIN = 210; V_MAX = 400;
    cv::inRange(HSV, cv::Scalar(H_MIN,S_MIN,V_MIN), cv::Scalar(H_MAX,S_MAX,V_MAX), threshold);

    // step 3:
    //      -> Morphological Operations
    //          - "Dilate" and "Erode" functions
    //          - ERODE (remove noise)
    //              * "erodes" into white space. Making it smaller or non existent
    //          - DILATE (definitive object without noise)
    //              * "dilates" white space. Making it larger.
    // perform morphological operations on thresholded image to eliminate noise and emphasize the filtered object(s)
    morphOps(threshold);

    // final step:
    //      -> FindContours
    //          - Input: binary image
    //          - Output: a vector of contours. The outline of all objects found in binary image
    //      -> "Moments" method
    //          - Input: vector of contours
    //          - Output: x, y coordinates of largest contour. Defined by its inner area.
    // pass in thresholded frame to our object tracking function
    // this function will return the x and y coordinates of the filtered object
    trackFilteredObject(x, y, threshold,image_clone);

    cv::namedWindow( "hsv image", cv::WINDOW_AUTOSIZE );
    cv::imshow( "hsv image", HSV );
    cv::namedWindow( "threshold image", cv::WINDOW_AUTOSIZE );
    cv::imshow( "threshold image", threshold );
    cv::namedWindow("new image", cv::WINDOW_AUTOSIZE);
    cv::imshow("new image", image_clone);

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

int charToInt(char arg){
    int i;
    switch(arg)
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
    resize(originalImage, originalImage,
           cv::Size(originalImage.cols / 2, originalImage.rows / 2)); // to half size or even smaller
    if (originalImage.empty()) {
        // NOT SUCCESSFUL : the data attribute is empty

        std::cout << "Image file could not be open !!" << std::endl;

        return -1;
    }

    H_MIN = 0; H_MAX = 12; S_MIN = 75; S_MAX = 256; V_MIN = 139; V_MAX = 210; // RED
    //H_MIN = 0; H_MAX = 129; S_MIN = 30; S_MAX = 124; V_MIN = 0; V_MAX = 83; // BLACK
    image_clone = originalImage.clone();

    // Create window
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    // Display image
    cv::imshow("Original Image", originalImage);

    // Print some image features
    std::cout << "ORIGINAL IMAGE" << std::endl;
    printImageFeatures(originalImage);

    //create slider bars for HSV filtering
    createTrackbars();

    while(true){
        filterImage();

        //delay 30ms so that screen can refresh.
        //image will not appear without this waitKey() command
        cv::waitKey(30);
    }
    /*
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
    */

    cv::destroyAllWindows();

    return 0;

}