/*
 *
 * Adapted from
 * **** https://docs.opencv.org/4.2.0/de/dc0/samples_2tapi_2squares_8cpp-example.html#a18
 * **** https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector/tree/1f8365779f88f7f46634114bf2e35427bc1c00d0
 * **** https://raw.githubusercontent.com/kylehounslow/opencv-tuts/master/object-tracking-tut/objectTrackingTut.cpp
 *
 */

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
std::string ranks[18] = {"2", "2", "2", "3","3", "4", "4", "5", "6", "6","7", "a", "a", "j","k", "k", "k", "q"};

std::string best_rank_match_name = "Unknown";
// Dimensions of rank train images
int RANK_WIDTH = 70;
int RANK_HEIGHT = 125;

const char* wndname = "Square Detection Demo";

std::string filename = "7d";

/* Adapted from https://docs.opencv.org/4.2.0/de/dc0/samples_2tapi_2squares_8cpp-example.html#a18*/
static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// https://raw.githubusercontent.com/kylehounslow/opencv-tuts/master/object-tracking-tut/objectTrackingTut.cpp
std::string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

/* Adapted from https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector/tree/1f8365779f88f7f46634114bf2e35427bc1c00d0*/
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
        cv::namedWindow("warp", cv::WINDOW_AUTOSIZE);
        cv::imshow("warp", warp);
    }

    // Width and height of card corner, where rank and suit are
    int CORNER_WIDTH = 32;
    int CORNER_HEIGHT = 84;

    // Grab corner of warped card image and do a 4x zoom
    cv::Mat Qcorner = warp(cv::Rect(0,0,CORNER_WIDTH, CORNER_HEIGHT));
    cv::Mat Qcorner_zoom;
    cv::resize(Qcorner, Qcorner_zoom, cv::Size(0, 0), 4, 4);
    if(Qcorner_zoom.data) {
        cv::namedWindow("Qcorner_zoom", cv::WINDOW_AUTOSIZE);
        cv::imshow("Qcorner_zoom", Qcorner_zoom);
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
    cv::Mat Qrank = query_thresh(cv::Rect(2*query_thresh.size().width/7, query_thresh.size().height/8, 5*query_thresh.size().width/7, 4*query_thresh.size().height/9));

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
        cv::namedWindow("Qrank", cv::WINDOW_AUTOSIZE);
        cv::imshow("Qrank", Qrank);
    }


    cv::Rect rank_rect = cv::boundingRect(Qrank);
    //rectangle( Qrank, rank_rect.tl(), rank_rect.br(), cv::Scalar(255,255,0), 2 );
    cv::Mat Qrank_crop = Qrank(cv::Rect(rank_rect.x, rank_rect.y, rank_rect.width, rank_rect.height));

    if(Qrank_crop.data) {
        cv::namedWindow("Qrank_crop", cv::WINDOW_AUTOSIZE);
        cv::imshow("Qrank_crop", Qrank_crop);
    }
    if (Qrank_crop.data) {
        cv::resize(Qrank_crop, Qrank_crop, cv::Size(RANK_WIDTH,RANK_HEIGHT));
        cv::namedWindow("Qrank_crop resize", cv::WINDOW_AUTOSIZE);
        cv::imshow("Qrank_crop resize", Qrank_crop);
        //cv::imwrite( filename + "_train.jpg", Qrank_crop );
    }
    return Qrank_crop;
}

/* Adapted from https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector/tree/1f8365779f88f7f46634114bf2e35427bc1c00d0*/
std::string match_card(cv::Mat qCard, int size) {
    // Finds best rank matches for the query card. Differences
    // the query card rank image with the train rank image.
    // The best match is the rank image that has the least difference.

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
            qCard_res = qCard;
            train_res = train_ranks[img_i];
            //cv::resize(qCard, qCard_res, cv::Size(RANK_WIDTH,RANK_HEIGHT));
            //cv::resize(train_ranks[img_i], train_res, cv::Size(RANK_WIDTH,RANK_HEIGHT));

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
                    cv::Mat tmp_img = qCard_res - train_res;
                    cv::namedWindow("tmp_img", cv::WINDOW_AUTOSIZE);
                    cv::imshow("tmp_img", tmp_img);
                    cv::namedWindow("qCard_res", cv::WINDOW_AUTOSIZE);
                    cv::imshow("qCard_res", qCard_res);
                    cv::namedWindow("train img", cv::WINDOW_AUTOSIZE);
                    cv::imshow("train img", train_res);
                }
            }
        }
    }

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

/* Adapted from https://docs.opencv.org/4.2.0/de/dc0/samples_2tapi_2squares_8cpp-example.html#a18*/
// finds sequence of squares detected on the image.
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
                    double perimeter = cv::arcLength(contours[i], true); // Calculates a contour perimeter or a curve length.
                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    cv::approxPolyDP(contours[i], approx, perimeter * 0.01,
                                     true); // Approximates a polygonal curve(s) with the specified precision.

                    // Find width and height of card's bounding rectangle
                    cv::RotatedRect rotr = minAreaRect(cv::Mat(contours[i]));
                    cv::Rect r = cv::boundingRect(contours[i]); // Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
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
                        //std::cout << "w: " + intToString(r.width) + ", h: " + intToString(r.height) << std::endl;
                        //std::cout << ratio << std::endl;
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
                            bool accept_card = true;
                            if(!cards.empty()){
                                for(Card c_tmp:cards){
                                    for(int cin_card = 0; cin_card < 4; cin_card++) {
                                        if(!accept_card){
                                            break;
                                        }
                                        for (int cin = 0; cin < 4; cin++) {
                                            //std::cout<<"fabs x : ";
                                            //std::cout<<fabs(cornerPoints[0].x - c_tmp.cornerPoints[0].x)<<std::endl;
                                            //std::cout<<"fabs y: ";
                                            //std::cout<<fabs(cornerPoints[0].y - c_tmp.cornerPoints[0].y)<<std::endl;
                                            if (fabs(cornerPoints[0].x - c_tmp.cornerPoints[0].x)<100 &&
                                                       fabs(cornerPoints[0].y - c_tmp.cornerPoints[0].y)<100){
                                                accept_card = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if(accept_card) {
                                //std::cout << "rotr ang: " << std::endl;
                                //std::cout << intToString((int) rotr.angle) << std::endl;

                                squares.push_back(approx);
                                for (int index = 0; index < 4; index++) {
                                    cv::circle(image, cornerPoints[index], 4, cv::Scalar(0, 255, 0), 2);
                                }

                                Card c;
                                c.image = image;
                                for (int ci = 0; ci < 4; ci++) {
                                    c.cornerPoints[ci] = cornerPoints[ci];
                                }
                                c.rotr = rotr;
                                cv::Mat cropRank = flattener(image, cornerPoints, rotr);
                                c.cropRank = cropRank;
                                c.rank = match_card(cropRank, 18);


                                int p = 0;
                                if ((c.rank).compare("7") == 0) {
                                    p = 10;
                                } else if ((c.rank).compare("a") == 0) {
                                    p = 11;
                                } else if ((c.rank).compare("k") == 0) {
                                    p = 4;
                                } else if ((c.rank).compare("j") == 0) {
                                    p = 3;
                                } else if ((c.rank).compare("q") == 0) {
                                    p = 2;
                                }
                                std::cout << "PONTOS CARTA " + c.rank + ": " + intToString(p) << std::endl;
                                putText(image, intToString(p), cv::Point(cornerPoints[0].x + 5, cornerPoints[0].y + 5),
                                        1, 1, cv::Scalar(0, 255, 255), 2);

                                //std::cout << c.rank << std::endl;
                                cards.push_back(c);
                            }
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
        if((card.rank).compare("7") == 0) {
            p += 10;
        } else if((card.rank).compare("a") == 0) {
            p += 11;
        } else if((card.rank).compare("k") == 0) {
            p += 4;
        } else if((card.rank).compare("j") == 0) {
            p += 3;
        } else if((card.rank).compare("q") == 0) {
            p += 2;
        }
    }
    if(p != 0){
        putText(image, intToString(p), cv::Point(10, 10), 1, 1, cv::Scalar(0,255,0),2);
    }
}

/* Adapted from https://docs.opencv.org/4.2.0/de/dc0/samples_2tapi_2squares_8cpp-example.html#a18*/
// the function draws all the squares in the image
static void drawSquares( cv::Mat& image, const std::vector<std::vector<cv::Point> >& squares, std::vector<Card> & cards)
{
    resize(image, image,cv::Size(image.cols / 10, image.rows / 10)); // to half size or even smaller

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

    filename = "ex_3";
    image = cv::imread(filename + ".jpg", cv::IMREAD_UNCHANGED);

    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop
    std::vector<std::vector<cv::Point> > squares;
    std::vector<Card> cards;

    std::string fns[] = {"2_diamonds_train", "2_hearts_train", "2_spades_train", "3_hearts_train",
                         "3_spades_train", "4_diamonds_train", "4_spades_train", "5_hearts_train", "6_hearts_train",
                         "6_spades_train", "7e_train", "ace_diamonds_train", "ace_hearts_train", "jack2_train",
                         "king_clubs_train", "king_hearts_train", "king_spades_train", "queen_clubs_train"};
    for(int ti = 0; ti < 18; ti++){
        train_ranks[ti] = cv::imread(fns[ti] + ".jpg", cv::IMREAD_UNCHANGED);
    }

    /*
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
*/
    //while(true){
        //capture>>image;
        findSquares(image, squares, cards);
        drawSquares(image, squares, cards);

        if(image.data) {
            cv::namedWindow("count cards", cv::WINDOW_AUTOSIZE);
            cv::imshow("count cards", image);
        }

        //delay 30ms so that screen can refresh.
        //image will not appear without this waitKey() command
        cv::waitKey(0);
        //cv::waitKey(30);
    //}


    cv::destroyAllWindows();
    return 0;
}