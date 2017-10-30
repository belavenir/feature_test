#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <dirent.h>
#include <ctime>
using namespace cv;
using namespace std;



void ransacTest(const std::vector<cv::DMatch> matches, 
                const std::vector<cv::KeyPoint>& kp1,
                const std::vector<cv::KeyPoint>& kp2,
                std::vector<cv::DMatch>& ransacMatches,
                double distance, 
                double confidence);


int main(int argc, const char *argv[]){

    string path = "../data/match/";
    string scenario = "scenario4/";

    Mat image1 = cv::imread( path+scenario+"frame.png");
    Mat image2 = cv::imread( path+scenario+"pano.png", CV_LOAD_IMAGE_GRAYSCALE);

    if(image1.empty() || image2.empty() )                              
    {
        cerr <<  "Could not open pano or vedecom image" << std::endl ;
        return -1;
    }


    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;

    initModule_nonfree();
    /* 
     * SIFT,SURF, ORB
    */

    string dect = "HARRIS";
    string desp = "SURF";

    detector = FeatureDetector::create(dect);
    extractor = DescriptorExtractor::create(desp);
   

    clock_t begin = clock();

    vector<KeyPoint> keypoints1, keypoints2;
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    cout << "# keypoints of image1 :" << keypoints1.size() << endl;
    cout << "# keypoints of image2 :" << keypoints2.size() << endl;
   
    Mat descriptors1,descriptors2;
    extractor->compute(image1,keypoints1,descriptors1);
    extractor->compute(image2,keypoints2,descriptors2);


    cout << "Descriptors size :" << descriptors1.cols << " : "<< descriptors1.rows << endl;

    vector< vector<DMatch> > matches12, matches21;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased"); //FlannBased or BruteForce
    matcher->knnMatch( descriptors1, descriptors2, matches12, 60 );
    matcher->knnMatch( descriptors2, descriptors1, matches21, 60 );
    
    // BFMatcher bfmatcher(NORM_L2, true);
    // vector<DMatch> matches;
    // bfmatcher.match(descriptors1, descriptors2, matches);
    cout << "Matches1-2:" << matches12.size() << endl;
    cout << "Matches2-1:" << matches21.size() << endl;

    // ratio test proposed by David Lowe paper = 0.8
    std::vector<DMatch> good_matches1, good_matches2;

    for(int i=0; i < matches12.size(); i++){
        const float ratio = 0.8;
        if(matches12[i][0].distance < ratio * matches12[i][1].distance)
            good_matches1.push_back(matches12[i][0]);
    }

    for(int i=0; i < matches21.size(); i++){
        const float ratio = 0.8;
        if(matches21[i][0].distance < ratio * matches21[i][1].distance)
            good_matches2.push_back(matches21[i][0]);
    }

    cout << "Good matches1:" << good_matches1.size() << endl;
    cout << "Good matches2:" << good_matches2.size() << endl;

    // Symmetric Test
    std::vector<DMatch> better_matches;
    for(int i=0; i<good_matches1.size(); i++){
        for(int j=0; j<good_matches2.size(); j++){
            if(good_matches1[i].queryIdx == good_matches2[j].trainIdx && good_matches2[j].queryIdx == good_matches1[i].trainIdx){
                better_matches.push_back(DMatch(good_matches1[i].queryIdx, good_matches1[i].trainIdx, good_matches1[i].distance));
                break;
            }
        }
    }


    cout << "Better matches:" << better_matches.size() << endl;

    std::vector<cv::DMatch> ransac_matches;
    ransacTest(better_matches, keypoints1, keypoints2, ransac_matches, 2, 0.99);

    cout<<"ransac matches="<<ransac_matches.size()<<endl;


    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time Costs : " << elapsed_secs << endl;


    // show it on an image


    Mat output;
    drawMatches(image1, keypoints1, image2, keypoints2, matches12, output);
    imshow("Matches result 0",output);
    string filename = "../result/"+scenario+dect+desp+"0.png";
    cv::imwrite( filename, output );
    waitKey(0);

    Mat output1;
    drawMatches(image1, keypoints1, image2, keypoints2, better_matches, output1);
    imshow("Matches result 1",output1);
    string filename1 = "../result/"+scenario+dect+desp+"1.png";
    cv::imwrite( filename1, output1 );
    waitKey(0);


    Mat output2;
    drawMatches(image1, keypoints1, image2, keypoints2, ransac_matches, output2);
    imshow("Matches result 2",output2);
    string filename2 = "../result/" +scenario+dect+desp+"2.png";
    cv::imwrite( filename2, output2 );
    waitKey(0);

    return 0;
}



void ransacTest(const std::vector<cv::DMatch> matches, 
                const std::vector<cv::KeyPoint>& kp1,
                const std::vector<cv::KeyPoint>& kp2,
                std::vector<cv::DMatch>& ransacMatches,
                double distance, 
                double confidence){ 
        
    ransacMatches.clear();
    //convert keypoint into point2f
    std::vector<cv::Point2f> pts1, pts2;

    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); //const_iterator ??
            it != matches.end(); ++it){
        //get the position of reference kpts
        float x = kp1[it->trainIdx].pt.x;
        float y = kp1[it->trainIdx].pt.y;
        pts1.push_back(cv::Point2f(x,y));

        //get the position of current kpts
        x = kp2[it->queryIdx].pt.x;
        y = kp2[it->queryIdx].pt.y;
        pts2.push_back(cv::Point2f(x,y));   
    }

    //compute F matrix using RANSAC
    std::vector<uchar> inliers(pts1.size(),0);
    cv::Mat F = cv::findFundamentalMat(cv::Mat(pts1), cv::Mat(pts2),inliers, CV_FM_RANSAC, distance,confidence);
    //extract the surviving matches(inliers)
    std::vector<uchar> :: const_iterator itIn = inliers.begin();
    std::vector<cv::DMatch> ::const_iterator itM = matches.begin();
    //for all matches
    for ( ; itIn != inliers.end(); ++itIn, ++itM)
    {
        if (*itIn)
            ransacMatches.push_back(*itM);
    }
}