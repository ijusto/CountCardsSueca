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

#include "stdlib.h"


//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
cv::Mat image;
cv::Mat train_ranks[18];
std::string ranks[18] = {"2", "2", "2", "2", "3","3", "4", "5", "6", "6","7", "a", "a", "j","k", "k", "k", "q"};

std::string best_rank_match_name = "Unknown";
// Dimensions of rank train images
int RANK_WIDTH = 70;
int RANK_HEIGHT = 125;
void swap(double *xp, double *yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void swapOther(std::vector<cv::Point> *xp, std::vector<cv::Point> *yp)
{
    std::vector<cv::Point> temp = *xp;
    *xp = *yp;
    *yp = temp;
}

// A function to implement bubble sort
void bubbleSort(double arr[], std::vector<std::vector<cv::Point>> other, int n)
{
    int i, j;
    for (i = 0; i < n-1; i++) {
        // Last i elements are already in place
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] < arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
                swapOther(&other[j], &other[j + 1]);
            }
        }
    }
}

void filterImage(){

    // Returns a grayed, blurred, and adaptively thresholded camera image.
    cv::Mat gray, blur, thresh;
    // Converts an image from one color space to another.
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);


    // The best threshold level depends on the ambient lighting conditions.
    // For bright lighting, a high threshold must be used to isolate the cards
    // from the background. For dim lighting, a low threshold must be used.
    // To make the card detector independent of lighting conditions, the
    // following adaptive threshold method is used.

    // A background pixel in the center top of the image is sampled to determine
    // its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    // than that. This allows the threshold to adapt to the lighting conditions.
    int img_w = image.size().width;
    int img_h = image.size().height;
    double bkg_level = gray.at<uchar>(int(img_h / 100), int(img_w / 2));

    // Adaptive threshold levels
    int BKG_THRESH = 60;

    double thresh_level = bkg_level + BKG_THRESH;
    cv::threshold(blur, image, thresh_level, 255, cv::THRESH_BINARY);

    // Find and sort the contours of all cards in the image (query cards)
    std::vector<std::vector<cv::Point>> cnts;
    std::vector<cv::Vec4i> hier;
    // Find contours and sort their indices by contour size
    cv::findContours(image, cnts, hier, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    double cntarea[cnts.size()];
    for(int i = 0; i < cnts.size(); i++){
        cntarea[i] = cv::contourArea(cnts[i]);
    }

    bubbleSort(cntarea, cnts, cnts.size());

    std::vector<std::vector<cv::Point>> cnts_sort;
    std::vector<cv::Vec4i> hier_sort;
    int cnt_is_card[cnts.size()];

    // If there are no contours, do nothing
    if (!cnts.empty()){
        // Otherwise, initialize empty sorted contour and hierarchy lists
        for (int i = 0; i < cnts.size(); i++) {
            cnt_is_card[i] = 0;
        }

        // Fill empty lists with sorted contour and sorted hierarchy. Now,
        // the indices of the contour list still correspond with those of
        // the hierarchy list. The hierarchy array can be used to check if
        // the contours have parents or not.
        for (int i = 0; i < cnts.size(); i++) {
            cnts_sort[i] = cnts[i];
            hier_sort[0][i] = hier[0][i];
        }

        // Determine which of the contours are cards by applying the following criteria:
        // 1) Smaller area than the maximum card size,
        // 2), bigger area than the minimum card size,
        // 3) have no parents, and
        // 4) have four corners

        double CARD_MAX_AREA = 120000;
        double CARD_MIN_AREA = 25000;

        for (int i = 0; i < cnts_sort.size(); i++) {
            double size = cv::contourArea(cnts_sort[i]);
            double peri = cv::arcLength(cnts_sort[i], true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cnts_sort[i], approx, 0.02 * peri, true);

            if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and (hier_sort[i][3] == -1) and (approx.size() == 4)){
                cnt_is_card[i] = 1;
            }
        }
    }

    /*


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
     */

}
const char* wndname = "Square Detection Demo";

std::string filename = "7_clubs";

static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// from https://www.youtube.com/watch?v=bSeFrPrqZ2A
std::string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}
/* 	Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
*/
cv::Mat flattener(cv::Mat image, cv::Point2f cornerPoints[], cv::RotatedRect rotr){

    int maxWidth = 200;
    int maxHeight = 300;

    //cv::Mat rot = cv::getRotationMatrix2D(rotr.center, 90 - rotr.angle, 1.0);
    //cv::Mat dstimg;
    //cv::warpAffine(image, image, rot, rotr.boundingRect2f().size());

    // Create destination array, calculate perspective transform matrix, and warp card image
    cv::Point2f dst[4];
    dst[0] = cv::Point2f(0,0);
    dst[1] = cv::Point2f(maxWidth - 1,0);
    dst[2] = cv::Point2f(maxWidth - 1,maxHeight - 1);
    dst[3] = cv::Point2f(0,maxHeight - 1);

    cv::Mat warp;
    cv::Point2f cpts[4];
    rotr.points(cpts);
    cv::Mat m = cv::getPerspectiveTransform(cpts, dst); //Calculates a perspective transform from four pairs of the corresponding points.
    //cv::Mat m = cv::getPerspectiveTransform(cpts, dst); //Calculates a perspective transform from four pairs of the corresponding points.
    cv::warpPerspective(image, warp, m, cv::Size(maxWidth, maxHeight)); // Applies a perspective transformation to an image.
    //cv::warpPerspective(dstimg, warp, m, cv::Size(maxWidth, maxHeight)); // Applies a perspective transformation to an image.

    // Convert to greysacle
    cv::cvtColor(warp, warp, cv::COLOR_BGR2GRAY);
    if(warp.data) {
        cv::namedWindow("count2 cards", cv::WINDOW_AUTOSIZE);
        cv::imshow("count2 cards", warp);
    }

    // Width and height of card corner, where rank and suit are
    int CORNER_WIDTH = 32;
    int CORNER_HEIGHT = 84;

    // Grab corner of warped card image and do a 4x zoom
    cv::Mat Qcorner = warp(cv::Rect(0,0,CORNER_WIDTH, CORNER_HEIGHT));
    cv::Mat Qcorner_zoom;
    cv::resize(Qcorner, Qcorner_zoom, cv::Size(0, 0), 4, 4);
    if(Qcorner_zoom.data) {
        cv::namedWindow("count3 cards", cv::WINDOW_AUTOSIZE);
        cv::imshow("count3 cards", Qcorner_zoom);
    }


    // Adaptive threshold levels
    int BKG_THRESH = 60;
    int CARD_THRESH = 30;

    // Sample known white pixel intensity to determine good threshold level
    int white_level = Qcorner_zoom.at<uchar>(15, (int)((CORNER_WIDTH * 4) / 2));
    int thresh_level = white_level - CARD_THRESH;
    if (thresh_level <= 0){
        thresh_level = 1;
    }
    cv::Mat query_thresh;
    cv::threshold(Qcorner_zoom, query_thresh, thresh_level, 255, cv::THRESH_BINARY_INV);


    // Split in to top and bottom half (top shows rank, bottom shows suit)
    //cv::Mat Qrank = query_thresh(cv::Rect(20, 0, 165, 128));
    cv::Mat Qrank = query_thresh(cv::Rect(query_thresh.size().width/7, query_thresh.size().height/8, 6*query_thresh.size().width/7, 4*query_thresh.size().height/9));

    //Extract the contours so that
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours0;
    findContours( Qrank, contours0, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    contours.resize(contours0.size());
    for( size_t k = 0; k < contours0.size(); k++ )
        cv::approxPolyDP(cv::Mat(contours0[k]), contours[k], 3, true);

    //cv::drawContours( Qrank, contours, -1, cv::Scalar(128,255,255), 3, cv::LINE_AA, hierarchy);
    if(Qrank.data) {
        cv::namedWindow("count5 cards", cv::WINDOW_AUTOSIZE);
        cv::imshow("count5 cards", Qrank);
    }


    cv::Rect rank_rect = cv::boundingRect(Qrank);
    //rectangle( Qrank, rank_rect.tl(), rank_rect.br(), cv::Scalar(255,255,0), 2 );
    cv::Mat Qrank_crop = Qrank(cv::Rect(rank_rect.x, rank_rect.y,
                                            rank_rect.width, rank_rect.height));

    if(Qrank.data) {
        cv::namedWindow("count6 cards", cv::WINDOW_AUTOSIZE);
        cv::imshow("count6 cards", Qrank);
    }
    if (Qrank_crop.data) {
        cv::resize(Qrank_crop, Qrank_crop, cv::Size(RANK_WIDTH,RANK_HEIGHT));
        cv::namedWindow("count7 cards", cv::WINDOW_AUTOSIZE);
        cv::imshow("count7 cards", Qrank_crop);
        cv::imwrite( filename + "_train.jpg", Qrank_crop );
    }
    return Qrank_crop;
}

std::string match_card(cv::Mat qCard, int size) {
    // Finds best rank and suit matches for the query card. Differences
    // the query card rank and suit images with the train rank and suit images.
    // The best match is the rank or suit image that has the least difference."""

    int best_rank_match_diff = 10000;
    cv::Mat best_rank_diff_img;
    int i = 0;

    // If no contours were found in query card in preprocess_card function,
    // the img size is zero, so skip the differencing process
    // (card will be left as Unknown)
    if (qCard.data) {

        // Difference the query card rank image from each of the train rank images,
        // and store the result with the least difference
        for (int img_i = 0; img_i < size; img_i++) {

            cv::Mat qCard_res, train_res;
            cv::resize(qCard, qCard_res, cv::Size(RANK_WIDTH,RANK_HEIGHT));
            cv::resize(train_ranks[img_i], train_res, cv::Size(RANK_WIDTH,RANK_HEIGHT));

            if(qCard_res.size == train_res.size) {
                cv::Mat diff_img;
                cv::absdiff(qCard_res, train_res, diff_img);
                int rank_diff;
                for (int ri = 0; ri < diff_img.rows; ri++) {
                    for (int ci = 0; ci < diff_img.cols; ci++) {
                        rank_diff += diff_img.at<uchar>(ri, ci);
                    }
                }
                rank_diff = rank_diff/255;
                if (rank_diff < best_rank_match_diff) {
                    best_rank_diff_img = diff_img;
                    best_rank_match_diff = rank_diff;
                    best_rank_match_name = ranks[img_i];
                    i = img_i;
                    cv::Mat tmp_img = qCard_res - train_res;
                    cv::namedWindow("query img", cv::WINDOW_AUTOSIZE);
                    cv::imshow("query img", qCard_res);
                    cv::namedWindow("train img", cv::WINDOW_AUTOSIZE);
                    cv::imshow("train img", train_res);
                }
            }
        }
    }
    std::cout<<"indice: " + intToString(i)<<std::endl;

    return best_rank_match_name;
}
class Card{
    public:
        cv::Mat image;
        cv::Point2f cornerPoints[4];
        cv::RotatedRect rotr;
        cv::Mat cropRank;
        std::string rank;
};

// returns sequence of squares detected on the image.
static void findSquares( const cv::Mat& image, std::vector<std::vector<cv::Point> >& squares, std::vector<Card> & cards)
{
    int thresh = 50, N = 11;
    squares.clear();
    cards.clear();
    cv::Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    // down-scale and upscale the image to filter out the noise
    if(image.data){
        cv::pyrDown(image, pyr, cv::Size(image.cols/2, image.rows/2));
        cv::pyrUp(pyr, timg, image.size());
        std::vector<std::vector<cv::Point> > contours;
        // find squares in every color plane of the image
        for( int c = 0; c < 3; c++ ) {
            int ch[] = {c, 0};
            cv::mixChannels(&timg, 1, &gray0, 1, ch, 1);
            // try several threshold levels
            for (int l = 0; l < N; l++) {
                // hack: use Canny instead of zero threshold level.
                // Canny helps to catch squares with gradient shading
                if (l == 0) {
                    // apply Canny. Take the upper threshold from slider
                    // and set the lower to 0 (which forces edges merging)
                    cv::Canny(gray0, gray, 0, thresh, 5);
                    // dilate canny output to remove potential
                    // holes between edge segments
                    cv::dilate(gray, gray, cv::Mat(), cv::Point(-1, -1));
                } else {
                    // apply threshold if l!=0:
                    //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                    gray = gray0 >= (l + 1) * 255 / N;
                }
                // find contours and store them all as a list
                cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                std::vector<cv::Point> approx;
                // test each contour
                for (size_t i = 0; i < contours.size(); i++) {
                    // Find perimeter of card and use it to approximate corner points
                    double perimeter = cv::arcLength(contours[i],
                                                     true); // Calculates a contour perimeter or a curve length.
                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    cv::approxPolyDP(contours[i], approx, perimeter * 0.01,
                                     true); // Approximates a polygonal curve(s) with the specified precision.

                    // Find width and height of card's bounding rectangle
                    cv::RotatedRect rotr = minAreaRect(cv::Mat(contours[i]));
                    cv::Rect r = cv::boundingRect(
                            contours[i]); // Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
                    double min = r.width;
                    double max = r.height;
                    if (min > max) {
                        double tmp = min;
                        min = max;
                        max = tmp;
                    }
                    double ratio = max / min;

                    cv::Point2f cornerPoints[4];
                    rotr.points(cornerPoints); // The order is bottomLeft, topLeft, topRight, bottomRight.
                    cv::Point2f cp_tmp[4];
                    // The order is topLeft, topRight, bottomRight, bottomLeft
                    cp_tmp[0] = cornerPoints[1];
                    cp_tmp[1] = cornerPoints[0];
                    cp_tmp[2] = cornerPoints[3];
                    cp_tmp[3] = cornerPoints[2];
                    // Sort corner point clockwise starting top left
                    for (int index = 0; index < 4; index++) {
                        cornerPoints[index] = cp_tmp[index];
                    }

                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because area may be positive or negative
                    // - in accordance with the contour orientation
                    double CARD_MAX_AREA = 120000;
                    double CARD_MIN_AREA = 1000;
                    if (approx.size() == 4 && fabs(cv::contourArea(approx)) > CARD_MIN_AREA
                        /*&& fabs(cv::contourArea(approx)) < CARD_MAX_AREA*/ && cv::isContourConvex(approx)
                        && (ratio > 1, 2 && ratio < 1, 7)
                            ) {
                        std::cout << "w: " + intToString(r.width) + ", h: " + intToString(r.height) << std::endl;
                        std::cout << ratio << std::endl;
                        double maxCosine = 0;
                        for (int j = 2; j < 5; j++) {
                            // find the maximum cosine of the angle between joint edges
                            double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                            maxCosine = MAX(maxCosine, cosine);
                        }
                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if (maxCosine < 0.3) {

                            std::cout << "rotr ang: " << std::endl;
                            std::cout << intToString((int) rotr.angle) << std::endl;

                            squares.push_back(approx);
                            for (int index = 0; index < 4; index++) {
                                cv::circle(image, cornerPoints[index], 4, cv::Scalar(0, 255, 0), 2);
                            }

                            Card c;
                            c.image = image;
                            for(int ci = 0; ci < 4; ci++){
                                c.cornerPoints[ci] = cornerPoints[ci];
                            }
                            c.rotr = rotr;
                            cv::Mat cropRank = flattener(image, cornerPoints, rotr);
                            c.cropRank = cropRank;
                            c.rank = match_card(cropRank, 18);
                            std::cout << c.rank << std::endl;
                            cards.push_back(c);
                        }
                    }
                }
            }
        }
    }
}

void calcPoints(std::vector<Card> & cards){
    int p = 0;
    for (auto& card:cards) {
        //if(card.rank.compare("2") || card.rank.compare("3") || card.rank.compare("4") || card.rank.compare("5") || card.rank.compare("6") {
        //}
        if(card.rank.compare("7")) {
            p += 10;
        } else if(card.rank.compare("a")) {
            p += 11;
        } else if(card.rank.compare("k")) {
            p += 4;
        } else if(card.rank.compare("j")) {
            p += 3;
        } else if(card.rank.compare("q")) {
            p += 2;
        }
    }
    if(p != 0){
        putText(image, intToString(p), cv::Point(10, 10), 1, 1, cv::Scalar(0,255,0),2);
    }

}

// the function draws all the squares in the image
static void drawSquares( cv::Mat& image, const std::vector<std::vector<cv::Point> >& squares, std::vector<Card> & cards)
{

    for (size_t i = 0; i < squares.size(); i++) {
        const cv::Point *p = &squares[i][0];
        int n = (int) squares[i].size();
        cv::polylines(image, &p, &n, 1, true, cv::Scalar(255,255,0/*std::rand()%255,std::rand()%255,std::rand()%255*/), 3, cv::LINE_AA);
    }
    calcPoints(cards);
    if(image.data) {
        //cv::imshow(wndname, image);
    }
}

int main( int argc, char** argv ) {

    //std::string fn[] = {"queen_clubs"};

    //filename = fn[0];
    //image = cv::imread(filename + ".jpg", cv::IMREAD_UNCHANGED);
    //resize(image, image,cv::Size(image.cols / 10, image.rows / 10)); // to half size or even smaller

    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop
    std::vector<std::vector<cv::Point> > squares;
    std::vector<Card> cards;

    std::string fns[] = {"2_clubs_train", "2_diamonds_train", "2_hearts_train", "2_spades_train", "3_hearts_train",
                         "3_spades_train", "4_diamonds_train", "5_hearts_train", "6_hearts_train", "6_spades_train",
                         "7_hearts_train", "ace_diamonds_train", "ace_hearts_train", "jack_hearts_train",
                         "king_clubs_train", "king_hearts_train", "king_spades_train", "queen_clubs_train"};
    for(int ti = 0; ti < 18; ti++){
        train_ranks[ti] = cv::imread(fns[ti] + ".jpg", cv::IMREAD_UNCHANGED);
    }

    // Processing keyboard events
    //video capture object to acquire webcam feed
    cv::VideoCapture capture;
    //open capture object at location zero (default location for webcam)
    capture.open(0);
    //set height and width of capture frame
    capture.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop

    while(true){
        capture>>image;
        findSquares(image, squares, cards);
        drawSquares(image, squares, cards);

        if(image.data) {
            cv::namedWindow("count cards", cv::WINDOW_AUTOSIZE);
            cv::imshow("count cards", image);
        }

        //delay 30ms so that screen can refresh.
        //image will not appear without this waitKey() command
        cv::waitKey(100);
    }


    cv::destroyAllWindows();
    return 0;
}