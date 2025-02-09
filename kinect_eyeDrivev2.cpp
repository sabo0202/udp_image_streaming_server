#include <list>
#include <stdexcept>
#include <OpenNI.h>
#include <opencv.hpp>
#include <tracking.hpp>
#include <imgcodecs.hpp>

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "PracticalSocket.h"      // For UDPSocket and SocketException
#include <iostream>               // For cout and cerr
#include <cstdlib>                // For atoi()

#include <opencv2/rgbd.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/viz.hpp>

#include "config.h"
#include"matplotlibcpp.h"

#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <fcntl.h>
#include <termios.h>
#include <time.h>

#define DEV_NAME "/dev/ttyACM0"
#define BAUD_RATE B115200
#define BUFF_SIZE 4096

#define GAZEX_BIAS -50
#define GAZEY_BIAS -50
#define BILATERAL_FILTER 0// if 1 then bilateral filter will be used for the depth

using namespace std;
using namespace cv;
using namespace cv::rgbd;
namespace plt = matplotlibcpp;

struct depthIndex {
  short pointIndex[1561];
};

template<typename T_n>
string NumToString(T_n number)
{
	stringstream ss;
	ss << number;
	return ss.str();
}

class MyTickMeter
{
public:
    MyTickMeter() { reset(); }
    void start() { startTime = getTickCount(); }
    void stop()
    {
        int64 time = getTickCount();
        if ( startTime == 0 )
            return;
        ++counter;
        sumTime += ( time - startTime );
        startTime = 0;
    }

    int64 getTimeTicks() const { return sumTime; }
    double getTimeSec()   const { return (double)getTimeTicks()/getTickFrequency(); }
    int64 getCounter() const { return counter; }

    void reset() { startTime = sumTime = 0; counter = 0; }
private:
    int64 counter;
    int64 sumTime;
    int64 startTime;
};


Point2d writeResults( const vector<Mat>& Rt )
{
    cout.precision(4);
    Point2d point;

    const Mat& Rt_curr = Rt[0];

    CV_Assert( Rt_curr.type() == CV_64FC1 );

    Mat R = Rt_curr(Rect(0,0,3,3)), rvec;
    Rodrigues(R, rvec);
    double alpha = norm( rvec );
    if(alpha > DBL_MIN)
        rvec = rvec / alpha;

    double cos_alpha2 = std::cos(0.5 * alpha);
    double sin_alpha2 = std::sin(0.5 * alpha);

    rvec *= sin_alpha2;

    CV_Assert( rvec.type() == CV_64FC1 );
    // timestamp tx ty tz qx qy qz qw
    //cout << fixed
    //     << Rt_curr.at<double>(0,3) << " " << Rt_curr.at<double>(1,3) << " " << Rt_curr.at<double>(2,3) << " "
    //     << rvec.at<double>(0) << " " << rvec.at<double>(1) << " " << rvec.at<double>(2) << " " << cos_alpha2 << endl;

    point.x = Rt_curr.at<double>(0,3);
    point.y = Rt_curr.at<double>(2,3);
    
    return point;
}


Point2d writeResults1( const vector<Mat>& Rt )
{
    cout.precision(4);
    Point2d point;

    const Mat& Rt_curr = Rt[0];

    CV_Assert( Rt_curr.type() == CV_64FC1 );

    Mat R = Rt_curr(Rect(0,0,3,3)), rvec;
    Rodrigues(R, rvec);
    double alpha = norm( rvec );
    if(alpha > DBL_MIN)
        rvec = rvec / alpha;

    double cos_alpha2 = std::cos(0.5 * alpha);
    double sin_alpha2 = std::sin(0.5 * alpha);

    rvec *= sin_alpha2;

    CV_Assert( rvec.type() == CV_64FC1 );
    // timestamp tx ty tz qx qy qz qw
    //cout << fixed
    //     << Rt_curr.at<double>(0,3) << " " << Rt_curr.at<double>(1,3) << " " << Rt_curr.at<double>(2,3) << " "
    //     << rvec.at<double>(0) << " " << rvec.at<double>(1) << " " << rvec.at<double>(2) << " " << cos_alpha2 << endl;

    point.x = Rt_curr.at<double>(0,3);
    point.y = Rt_curr.at<double>(2,3);
    
    return point;
}


void setCameraMatrixFreiburg1(float& fx, float& fy, float& cx, float& cy)
{
    fx = 517.3f; fy = 516.5f; cx = 318.6f; cy = 255.3f;
}



class GrabDetectorSample
{
public:

  /* Initialize */
  void initialize()
  {
    initOpenNI();
  }

  /* OpenNI's initialize */
  void initOpenNI()
  {
    openni::OpenNI::getVersion();
    openni::OpenNI::initialize();
    device.open( openni::ANY_DEVICE );

    /* Requires VGA depth & color input, with both depth-color registration and time sync enabled.  See included sample for setting this up. */ 

    /* Create color stream */
    colorStream.create( device, openni::SENSOR_COLOR );
    initStream( colorStream );
    colorStream.start();

    /* Create depth stream */
    depthStream.create( device, openni::SENSOR_DEPTH );
    initStream( depthStream );
    depthStream.start();

    /* Synchronism Depth & Color */
    device.setDepthColorSyncEnabled( true );

    /* Fit the Depth to Color */
    device.setImageRegistrationMode( openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR );

    /* Put together stream */
    streams.push_back( &colorStream );
    streams.push_back( &depthStream );
  }

  /* Initialize the OpenNI stream */
  void initStream( openni::VideoStream& stream )
  {
    openni::VideoMode videoMode = stream.getVideoMode();
    videoMode.setFps( 30 );
    videoMode.setResolution( 640, 480 );
    stream.setVideoMode( videoMode );
  }


  /* Main loop */
  void run()
  {

    /* Automatic */
    int automatic = 0;
    int changeMode = 0;
    int Stop_flag = 1;
    int Odometry_flag = 0;
    int flag_start = 0;
    int flag_goal = 0;

    /* Init depthValue*/
    struct depthIndex depthValue;

    /* Init arduino */
    int forward 	= 1;
    int stop 		= 2;
    int forwardRight 	= 3;
    int forwardLeft 	= 4;
    int back 		= 5;
    int left 		= 6;
    int right 		= 7;
    int backLeft 	= 8;
    int backRight 	= 9;
    int driveCount 	= 0;
    //int back = 1;
    int fd;
    int len; // データ数（バイト）
    unsigned char buffer[BUFF_SIZE], in_data[BUFF_SIZE];

    printf("start serial port read example..\n");

    // デバイスファイル（シリアルポート）オープン
    //fd = open(DEV_NAME, O_RDWR); // この書き方だとopenできない
    fd = open(DEV_NAME, O_RDWR | O_NONBLOCK );
    if(fd<0) { // デバイスオープンに失敗
      printf("ERROR on device open\n");
      exit(1);
    }

    printf("init serial port\n");
    serial_init(fd); // シリアルポートの初期化

    printf("start main loop...\n");
    //j=0;

     /* Save a movie */
    double fps = 7;
    Size size = Size(640, 480);
    const int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
    string filename_colorImage = "color.avi";
    //string filename_depthImage = "depth.avi";
    VideoWriter colorWriter(filename_colorImage, fourcc, fps, size);
    //VideoWriter depthWriter(filename_depthImage, fourcc, fps, size);


    /* Init visual odometry */
    Point2d kinect_point, target_point;
    kinect_point.x = 0; kinect_point.y = 0;
    target_point.x = 0; target_point.y = 0;
    //double sita;
    //bool running = true;
    vector<Mat> Rts;
    MyTickMeter gtm;
    //int count = 0;
    //vector<double> x, y, X, Y;
    //plt::xlim(-15.0, 15.0);
    //plt::ylim(-15.0, 15.0);

    float fx = 525.0f, // default
          fy = 525.0f,
          cx = 319.5f,
          cy = 239.5f;

    setCameraMatrixFreiburg1(fx, fy, cx, cy);

    Mat cameraMatrix = Mat::eye(3,3,CV_32FC1);
    {
        cameraMatrix.at<float>(0,0) = fx;
        cameraMatrix.at<float>(1,1) = fy;
        cameraMatrix.at<float>(0,2) = cx;
        cameraMatrix.at<float>(1,2) = cy;
    }

    Ptr<OdometryFrame> frame_prev = Ptr<OdometryFrame>(new OdometryFrame()),
                       frame_curr = Ptr<OdometryFrame>(new OdometryFrame());
    Ptr<Odometry> odometry = Odometry::create("RgbdOdometry");
    if(odometry.empty())
    {
        cout << "Can not create Odometry algorithm. Check the passed odometry name." << endl;
    }
    odometry->setCameraMatrix(cameraMatrix);

    //viz::Viz3d myWindow("Point Cloud");
    cv::Affine3d cam_pose;


    /* Init visual odometry */
    Point2d kinect_point1, target_point1;
    kinect_point1.x = 0; kinect_point1.y = 0;
    target_point1.x = 0; target_point1.y = 0;
    //double sita;
    //bool running = true;
    vector<Mat> Rts1;
    MyTickMeter gtm1;
    //int count = 0;
    //vector<double> x, y, X, Y;
    //plt::xlim(-15.0, 15.0);
    //plt::ylim(-15.0, 15.0);

    float fx1 = 525.0f, // default
          fy1 = 525.0f,
          cx1 = 319.5f,
          cy1 = 239.5f;

    setCameraMatrixFreiburg1(fx1, fy1, cx1, cy1);

    Mat cameraMatrix1 = Mat::eye(3,3,CV_32FC1);
    {
        cameraMatrix1.at<float>(0,0) = fx1;
        cameraMatrix1.at<float>(1,1) = fy1;
        cameraMatrix1.at<float>(0,2) = cx1;
        cameraMatrix1.at<float>(1,2) = cy1;
    }

    Ptr<OdometryFrame> frame_prev1 = Ptr<OdometryFrame>(new OdometryFrame()),
                       frame_curr1 = Ptr<OdometryFrame>(new OdometryFrame());
    Ptr<Odometry> odometry1 = Odometry::create("RgbdOdometry");
    if(odometry1.empty())
    {
        cout << "Can not create Odometry algorithm. Check the passed odometry name." << endl;
    }
    odometry1->setCameraMatrix(cameraMatrix1);

    //viz::Viz3d myWindow("Point Cloud");
    cv::Affine3d cam_pose1;

    /* Output txetfile */
    fstream outputGazefile;
    outputGazefile.open("result_Gaze.txt", ios::out | ios::out);
    fstream outputOdomfile;
    outputGazefile.open("result_Odometry.txt", ios::out | ios::out);
    int file_count = 0;

    short targetGazeDepth;
    float xx = 0;
    float yy = 0;
    float zz = 0;
    int iJpeg = 0;
    char capturename[20];

    openni::VideoMode videoMode = depthStream.getVideoMode();
    int gazeCount = 0;
    int currentGazePoint = 0;
    int targetGazePoint = 0;
    int beforeGazePoint = 0;
    int targetGazeX = 0;
    int targetGazeY = 0;

    //int tracking_init = 0;

    /* Tracking */
    Ptr<Tracker> trackerKCF;
    Rect2d KCFroi;
    KCFroi.x = 0;	KCFroi.y = 0;
    //Scalar colorkcf = Scalar(200, 0, 0); // Color initialize

    /* UDP transfer Image */
    string servAddress = "192.168.1.1"; // First arg: server address
    unsigned short servPort = Socket::resolveService("12345", "udp");
    UDPSocket udp_sock;
    int jpegqual =  ENCODE_QUALITY; // Compression Parameter
    vector < uchar > encoded;

    /* TCP reciever gaze_x */
    int GazeX;
    struct sockaddr_in server_gazeX;
    int sock_gazeX;
    int buf_gazeX[1];
    int n_gazeX;
    int max_gazeX = 640;
    int min_gazeX = 0;

    /* TCP reciever gaze_y */
    int GazeY;
    struct sockaddr_in server_gazeY;
    int sock_gazeY;
    int buf_gazeY[1];
    int n_gazeY;
    int max_gazeY = 480;
    int min_gazeY = 0;
    
    /* For Image processing */
    Mat frame(320, 240, CV_8UC3);
    Mat send(320, 240, CV_8UC3);
    Mat depthImage(640, 480, CV_16UC1);
    Mat depthImage1(640, 480, CV_16UC1);
    Mat colorImage(640, 480, CV_8UC3);
    Mat colorImage1(640, 480, CV_8UC3);

    // clock_t last_cycle = clock();

    while ( 1 ) {

      //GazeX = 170;
      //GazeY = 379;


      try {
      	//Recieve gaze_x data
      	sock_gazeX = socket(AF_INET, SOCK_STREAM, 0);
      	server_gazeX.sin_family = AF_INET;
      	server_gazeX.sin_port = htons(22333);
      	server_gazeX.sin_addr.s_addr = inet_addr("192.168.1.1");

      	connect(sock_gazeX, (struct sockaddr *)&server_gazeX, sizeof(server_gazeX));

      	memset(buf_gazeX, 0, sizeof(buf_gazeX));
      	n_gazeX = read(sock_gazeX, buf_gazeX, sizeof(buf_gazeX));

      	if (n_gazeX < 1) {
      	   perror("read");
      	   // break;
      	}
      	close(sock_gazeX);

      	GazeX = *buf_gazeX;
        GazeX = GazeX / 2;
      	if (GazeX < 0) {
      	   GazeX = min_gazeX;
      	}
	else if (GazeX >= 640) {
       	  GazeX = max_gazeX;
      	}

      	//Recieve gaze_y data
      	sock_gazeY = socket(AF_INET, SOCK_STREAM, 0);
      	server_gazeY.sin_family = AF_INET;
      	server_gazeY.sin_port = htons(33222);
      	server_gazeY.sin_addr.s_addr = inet_addr("192.168.1.1");

      	connect(sock_gazeY, (struct sockaddr *)&server_gazeY, sizeof(server_gazeY));

      	memset(buf_gazeY, 0, sizeof(buf_gazeY));
      	n_gazeY = read(sock_gazeY, buf_gazeY, sizeof(buf_gazeY));

      	if (n_gazeY < 1) {
      	   perror("read");
      	   // break;
      	}
      	close(sock_gazeY);

      	GazeY = *buf_gazeY;
        GazeY = GazeY / 1.5;
      	if (GazeY < 0) {
       	  GazeY = min_gazeY;
      	}
	else if (GazeY >= 480) {
       	  GazeY = max_gazeY;
      	}
      }
      catch (SocketException & e) {
    	 cerr << e.what() << endl;
         GazeX = min_gazeX;
         GazeY = min_gazeY;
    	 // exit(1);
      }


      int changedIndex;
      openni::OpenNI::waitForAnyStream( &streams[0], streams.size(), &changedIndex );

      //Get the frame of depth & color
      openni::VideoFrameRef depthFrame;
      depthStream.readFrame( &depthFrame );
      openni::VideoFrameRef colorFrame;
      colorStream.readFrame( &colorFrame );  

      depthImage  = showDepthStream( depthFrame );
      depthImage1 = showDepthStream( depthFrame );
      colorImage  = showColorStream( colorFrame );
      colorImage1 = showColorStream( colorFrame );

      CV_Assert(!colorImage.empty());
      CV_Assert(!depthImage.empty());
      CV_Assert(depthImage.type() == CV_16UC1);


      //openni::VideoMode videoMode = depthStream.getVideoMode();
      unsigned short* depth;
      depth = (unsigned short*)depthFrame.getData();

      beforeGazePoint = currentGazePoint;
      currentGazePoint = GazeY * videoMode.getResolutionX() + GazeX;

      // targetGazePoint = (KCFroi.y - GAZEY_BIAS) * videoMode.getResolutionX() + (KCFroi.x - GAZEX_BIAS);
      //targetGazePoint = targetGazeY * videoMode.getResolutionX() + targetGazeX;

      //currentGazeDepth = depth[currentGazePoint];
      //targetGazeDepth = depth[targetGazePoint];

      //openni::CoordinateConverter::convertDepthToWorld(depthStream, targetGazeX, targetGazeY, targetGazeDepth, &xx, &yy, &zz);
      target_point.x = (xx * 0.001) - (kinect_point.x * 10);	target_point.y = (zz * 0.001) - (kinect_point.y * 10);
      
      //x.push_back(kinect_point.x * 10);
      //y.push_back(kinect_point.y * 10);
      //x.push_back(0);
      //y.push_back(0);
      //X.push_back(xx * 0.001);
      //Y.push_back(zz * 0.001);
      //X.push_back(target_point.x);
      //Y.push_back(target_point.y);
      //x[0] = point.x;
      //y[0] = point.y; 
      //plt::plot(x, y, "xr");
      //plt::plot(X, Y, ".-b");
      //plt::pause(0.001);

      //stringstream ss;
      //ss << " targetGazeDepth : " << currentGazeDepth << " CurrentPoint : ( " << currentGazeDepth << ", " << GazeY << " ) ";
      //ss << "TargetDepth : " << targetGazeDepth << " TargetPoint : ( " << xx << ", " << yy << ", " << zz << " ) " << " CurrentDepth : " << currentGazeDepth << " CurrentPoint : ( " << GazeX << ", " << GazeY << " ) ";
      //putText( colorImage, ss.str(), Point( 0, 50 ), FONT_HERSHEY_SIMPLEX, 1.0, Scalar( 255 ) );


#if BILATERAL_FILTER
        MyTickMeter tm_bilateral_filter1;
#endif
        {
            //scale depth
            Mat depth_flt1;
            depthImage1.convertTo(depth_flt1, CV_32FC1, 1.f/8500.f);
#if !BILATERAL_FILTER
            depth_flt1.setTo(std::numeric_limits<float>::quiet_NaN(), depthImage1 == 0);
            depthImage1 = depth_flt1;
#else
            tm_bilateral_filter1.start();
            depthImage1 = Mat(depth_flt1.size(), CV_32FC1, Scalar(0));
            const double depth_sigma1 = 0.03;
            const double space_sigma1 = 4.5;  // in pixels
            Mat invalidDepthMask1 = depth_flt1 == 0.f;
            depth_flt1.setTo(-5*depth_sigma1, invalidDepthMask1);
            bilateralFilter(depth_flt1, depth, -1, depth_sigma1, space_sigma1);
            depthImage1.setTo(std::numeric_limits<float>::quiet_NaN(), invalidDepthMask1);
            tm_bilateral_filter1.stop();
            cout << "Time filter " << tm_bilateral_filter1.getTimeSec() << endl;
#endif
            //timestamps.push_back( timestap );
        }

        {
            Mat gray1;
            cvtColor(colorImage1, gray1, COLOR_BGR2GRAY);
            frame_curr1->image = gray1;
            frame_curr1->depth = depthImage1;
             
            Mat Rt1;
            if(!Rts1.empty())
            {
              MyTickMeter tm1;
              tm1.start();
              gtm1.start();
              bool res1 = odometry1->compute(frame_curr1, frame_prev1, Rt1);
              gtm1.stop();
              tm1.stop();
              //count++;
              //cout << "Time " << tm.getTimeSec() << endl;
#if BILATERAL_FILTER
              cout << "Time ratio " << tm_bilateral_filter1.getTimeSec() / tm1.getTimeSec() << endl;
#endif
              if(!res1)
                Rt1 = Mat::eye(4,4,CV_64FC1);
            }

            if( Rts1.empty() )
              Rts1.push_back(Mat::eye(4,4,CV_64FC1));
            else
            {
	      Mat& prevRt1 = *Rts1.rbegin();
              //cout << "Rt " << Rt << endl;
              Rts1[0] = ( prevRt1 * Rt1 );
            }

	    //10フレームごと変換して表示
	    /*Mat rot = Rts[count](Rect(0, 0, 3, 3)).t();
            Mat tvec = Rts[count](Rect(3, 0, 1, 3)).t();
	    if (count % 10 == 0){
		int downSamplingNum = 4; //e.g. 4
		Mat image2(image.rows / downSamplingNum, image.cols / downSamplingNum, CV_8UC3);
		resize(image, image2, image2.size(), 0, 0, INTER_LINEAR);
		Mat pCloud(image.rows / downSamplingNum, image.cols / downSamplingNum, CV_64FC3);

		for (int y = 0; y < 480; y += downSamplingNum){

			for (int x = 0; x < 640; x += downSamplingNum){
				if (depth.at<float>(y, x) < 8.0 && depth.at<float>(y, x) > 0.4){
					//RGB-D Dataset
					Mat pmat(1, 3, CV_64F);
					pmat.at<double>(0, 2) = (double)depth.at<float>(y, x);
					pmat.at<double>(0, 0) = (x - cx) * pmat.at<double>(0, 2) / fx;
					pmat.at<double>(0, 1) = (y - cy) * pmat.at<double>(0, 2) / fy;
					pmat = (pmat)*rot + tvec;
					Point3d p(pmat);
					pCloud.at<Point3d>(y / downSamplingNum, x / downSamplingNum) = p;
					pmat.release();
				}
				else{
					//RGB-D Dataset
					pCloud.at<Vec3d>(y / downSamplingNum, x / downSamplingNum) = Vec3d(0.f, 0.f, 0.f);
				}
			}
		}

		viz::WCloud wcloud(pCloud, image2);
		string myWCloudName = "CLOUD" + NumToString(count);
		myWindow.showWidget(myWCloudName, wcloud, cloud_pose_global);

		pCloud.release();
		image2.release();
	    }

	    cam_pose = cv::Affine3d(rot.t(), tvec);
	    viz::WCameraPosition cpw(0.1); // Coordinate axes
   	    viz::WCameraPosition cpw_frustum(cv::Matx33d(cameraMatrix), 0.1, viz::Color::white()); // Camera frustum
	    string widgetPoseName = "CPW" + NumToString(count);
	    string widgetFrustumName = "CPW_FRUSTUM" + NumToString(count);
	    myWindow.showWidget(widgetPoseName, cpw, cam_pose);
	    //myWindow.showWidget(widgetFrustumName, cpw_frustum, cam_pose);
	    myWindow.spinOnce(1, true);

	    rot.release();
	    tvec.release(); */

            if(!frame_prev1.empty())
              frame_prev1->release();
            std::swap(frame_prev1, frame_curr1);
        }
        kinect_point1 = writeResults1(Rts1);


      /* Draw the circle */
      circle(colorImage, Point(GazeX, GazeY), 5, Scalar(10, 10, 50), 3, 4);

      if (flag_start == 0) {

	stringstream ss;

	line(colorImage, Point(150, 0), Point(150, 60), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(490, 0), Point(490, 60), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(0, 60), Point(640, 60), Scalar(0, 0, 200), 2, 3);

	if ((GazeX >= 150 && GazeX <= 490) && (GazeY >= 0 && GazeY < 60)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          ss << " Stop ";
          len = write(fd, &stop, sizeof(stop));
            if(len==0) {
              continue;
            }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  flag_start = 1;
          gazeCount = 0;
	}

	else if ((GazeX >= 0 && GazeX < 150) && (GazeY >= 0 && GazeY < 60)) {
	  gazeCount++;
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          ss << " Stop & Automatic ";
	  len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }

	  if (gazeCount == 50) {
	    automatic = 1;
	    flag_start = 1;
	    gazeCount = 0;
	  }
	}

	else if ((GazeX > 490 && GazeX <= 640) && (GazeY >= 0 && GazeY < 60)) {
	  gazeCount++;
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          ss << " Stop & Automatic ";
	  len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }

	  if (gazeCount == 50) {
	    automatic = 1;
	    flag_start = 1;
	    gazeCount = 0;
	  }
	}
        putText( colorImage, ss.str(), Point( 0, 50 ), FONT_HERSHEY_SIMPLEX, 1.0, Scalar( 255 ) );
      }


      if (automatic == 0 && flag_start == 1) {

        stringstream ss;

	line(colorImage, Point(150, 0), Point(150, 480), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(490, 0), Point(490, 480), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(0, 60), Point(640, 60), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(0, 200), Point(150, 200), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(490, 200), Point(640, 200), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(0, 340), Point(640, 340), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(0, 480), Point(640, 480), Scalar(0, 0, 200), 2, 3);
	//rectangle(colorImage, Point(220, 140), Point(420, 340), Scalar(0, 0, 200), 2, 3);
	//rectangle(colorImage, Point(235, 180), Point(405, 300), Scalar(0, 0, 200), 2, 3);

	if ((GazeX >= 150 && GazeX <= 490) && (GazeY >= 60 && GazeY <= 340)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 250), 3, 4);
          ss << " Forward ";
	  len = write(fd, &forward, sizeof(forward));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX >= 150 && GazeX <= 490) && (GazeY > 340 && GazeY <= 480)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(100, 100, 100), 3, 4);
          ss << " Back ";
	  len = write(fd, &back, sizeof(back));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX >= 0 && GazeX < 150) && (GazeY >= 200 && GazeY < 340)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(200, 50, 50), 3, 4);
          ss << " Left ";
	  len = write(fd, &left, sizeof(left));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX > 490 && GazeX <= 640) && (GazeY >= 200 && GazeY < 340)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(50, 200, 50), 3, 4);
          ss << " Right ";
	  len = write(fd, &right, sizeof(right));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX >= 0 && GazeX < 150) && (GazeY >= 60 && GazeY < 200)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(250, 0, 0), 3, 4);
          ss << " ForwardLeft ";
	  len = write(fd, &forwardLeft, sizeof(forwardLeft));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX > 490 && GazeX <= 640) && (GazeY >= 60 && GazeY < 200)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 250, 0), 3, 4);
          ss << " ForwardRight ";
	  len = write(fd, &forwardRight, sizeof(forwardRight));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX >= 0 && GazeX < 150) && (GazeY >= 340 && GazeY <= 480)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(50, 200, 50), 3, 4);
          ss << " BackLeft ";
	  len = write(fd, &backLeft, sizeof(backLeft));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX > 490 && GazeX <= 640) && (GazeY >= 340 && GazeY <= 480)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(50, 50, 200), 3, 4);
          ss << " BackRight ";
	  len = write(fd, &backRight, sizeof(backRight));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	  gazeCount = 0;
	}

	else if ((GazeX >= 150 && GazeX <= 490) && (GazeY >= 0 && GazeY < 60)) {
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          ss << " Stop ";
          len = write(fd, &stop, sizeof(stop));
            if(len==0) {
              continue;
            }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
          gazeCount = 0;
	}

	else if ((GazeX >= 0 && GazeX < 150) && (GazeY >= 0 && GazeY < 60)) {
	  gazeCount++;
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          ss << " Stop & Automatic ";
	  len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }

	  if (gazeCount == 50) {
	    automatic = 1;
	    gazeCount = 0;
	  }
	}

	else if ((GazeX > 490 && GazeX <= 640) && (GazeY >= 0 && GazeY < 60)) {
	  gazeCount++;
	  //circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          ss << " Stop & Automatic ";
	  len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }

	  if (gazeCount == 50) {
	    automatic = 1;
	    gazeCount = 0;
	  }
	}
        putText( colorImage, ss.str(), Point( 0, 50 ), FONT_HERSHEY_SIMPLEX, 1.0, Scalar( 255 ) );
      }


      if (automatic == 1 && flag_start == 1) {

	//circle(colorImage, Point(targetGazeX, targetGazeY), 30, Scalar(0, 200, 0), 3, 4);
	/*if (depth[targetGazePoint] <= 700 && depth[targetGazePoint] > 0) {
          flag_goal = 0;
	  gazeCount = 0;
	  //tracking_init = 0;
	}*/

        stringstream ss;

	line(colorImage, Point(150, 0), Point(150, 60), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(490, 0), Point(490, 60), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(0, 60), Point(150, 60), Scalar(0, 0, 200), 2, 3);
	line(colorImage, Point(490, 60), Point(640, 60), Scalar(0, 0, 200), 2, 3);

	if ((GazeX >= 0 && GazeX <= 150) && (GazeY >= 0 && GazeY <= 60)) {
	  gazeCount++;
	  Stop_flag = 1;
          ss << " Stop & Mode change ";
	  len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }

	  if (gazeCount == 20) {
            flag_start = 0;
	    automatic = 0;
	    gazeCount = 0;
	    flag_goal = 0;
            Odometry_flag = 0;
            //target_point.x = 0;
	    //target_point.y = 0;
	  }
	}

	else if ((GazeX >= 490 && GazeX <= 640) && (GazeY >= 0 && GazeY <= 60)) {
	  gazeCount++;
	  Stop_flag = 1;
	  //flag_goal= 0;
          ss << " Stop & Mode change ";
	  len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }

	  if (gazeCount == 20) {
            flag_start = 0;
	    automatic = 0;
	    gazeCount = 0;
	    flag_goal = 0;
            Odometry_flag = 0;
            //target_point.x = 0;
	    //target_point.y = 0;
	  }
	}

        if ( !((GazeX >= 0 && GazeX <= 150) && (GazeY >= 0 && GazeY <= 60)) &&
	       !((GazeX >= 490 && GazeX <= 640) && (GazeY >= 0 && GazeY <= 60)) &&
	         (currentGazePoint == beforeGazePoint && gazeCount < 20) && flag_goal == 0) {

          gazeCount++;
	  if (gazeCount == 20) {
	    flag_goal = 1;
	    targetGazeX = GazeX;
	    targetGazeY = GazeY;

            targetGazePoint = targetGazeY * videoMode.getResolutionX() + targetGazeX;
            targetGazeDepth = depth[targetGazePoint];
	    if (targetGazeDepth == 0) {
	      targetGazeDepth = 10000;
	    }
            openni::CoordinateConverter::convertDepthToWorld(depthStream, targetGazeX, targetGazeY, targetGazeDepth, &xx, &yy, &zz);
	  }
        }

        else if ( !((GazeX >= 0 && GazeX <= 150) && (GazeY >= 0 && GazeY <= 60)) &&
	          !((GazeX >= 490 && GazeX <= 640) && (GazeY >= 0 && GazeY <= 60)) &&
	           (currentGazePoint != beforeGazePoint)) {

	  gazeCount = 0;
	  Stop_flag = 0;
          ss << " Automatic Mode ";
          putText( colorImage, ss.str(), Point( 0, 50 ), FONT_HERSHEY_SIMPLEX, 1.0, Scalar( 255 ) );
        }

	if (flag_goal == 1 && (target_point.x < 0.7 && target_point.x > -0.7) &&
	     (target_point.y < 0.7 && target_point.y > -0.7) && (target_point.y != 0 && target_point.y != 0)) {

          flag_start = 0;
	  flag_goal = 0;
          gazeCount = 0;
	  automatic = 0;
          Odometry_flag = 0;

          len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
	}

      }

      /* Tracker */
      /*if (flag_goal == 1) {
	if (tracking_init == 0) {
	  //Tracking init
	  trackerKCF = Tracker::create("KCF");
	  KCFroi.x = targetGazeX + GAZEX_BIAS;	KCFroi.y = targetGazeY + GAZEY_BIAS;
	  KCFroi.width = 100;	KCFroi.height = 100;
	  trackerKCF->init(colorImage, KCFroi);
	  tracking_init++;
        }
	trackerKCF->update(colorImage, KCFroi);
	rectangle(colorImage, KCFroi, colorkcf, 1, 1);
	//putText(colorImage, "- KCF", Point(5, 20), FONT_HERSHEY_SIMPLEX, .5, colorkcf, 1, CV_AA);
      }*/


      /* if (flag_goal == 1) {
        outputfile << file_count << " " << targetGazeDepth << endl;
	file_count++;
      } */


      if (flag_goal == 1 && automatic == 1 && Stop_flag == 0) {

	if (Odometry_flag == 0) {
          /* Init visual odometry */
          Point2d kinect_point, target_point;
          kinect_point.x = 0; kinect_point.y = 0;
          target_point.x = 0; target_point.y = 0;
          double sita;
          bool running = true;
          vector<Mat> Rts;
          MyTickMeter gtm;
          int count = 0;
          vector<double> x, y, X, Y;
          //plt::xlim(-15.0, 15.0);
          //plt::ylim(-15.0, 15.0);

          float fx = 525.0f, // default
                fy = 525.0f,
                cx = 319.5f,
                cy = 239.5f;

          setCameraMatrixFreiburg1(fx, fy, cx, cy);

          Mat cameraMatrix = Mat::eye(3,3,CV_32FC1);
          {
              cameraMatrix.at<float>(0,0) = fx;
              cameraMatrix.at<float>(1,1) = fy;
              cameraMatrix.at<float>(0,2) = cx;
              cameraMatrix.at<float>(1,2) = cy;
          }

          Ptr<OdometryFrame> frame_prev = Ptr<OdometryFrame>(new OdometryFrame()),
                             frame_curr = Ptr<OdometryFrame>(new OdometryFrame());
          Ptr<Odometry> odometry = Odometry::create("RgbdOdometry");
          if(odometry.empty())
          {
              cout << "Can not create Odometry algorithm. Check the passed odometry name." << endl;
          }
          odometry->setCameraMatrix(cameraMatrix);

          viz::Viz3d myWindow("Point Cloud");
          cv::Affine3d cam_pose;
	  Odometry_flag = 1;
	}

#if BILATERAL_FILTER
        MyTickMeter tm_bilateral_filter;
#endif
        {
            // scale depth
            Mat depth_flt;
            depthImage.convertTo(depth_flt, CV_32FC1, 1.f/8500.f);
#if !BILATERAL_FILTER
            depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depthImage == 0);
            depthImage = depth_flt;
#else
            tm_bilateral_filter.start();
            depthImage = Mat(depth_flt.size(), CV_32FC1, Scalar(0));
            const double depth_sigma = 0.03;
            const double space_sigma = 4.5;  // in pixels
            Mat invalidDepthMask = depth_flt == 0.f;
            depth_flt.setTo(-5*depth_sigma, invalidDepthMask);
            bilateralFilter(depth_flt, depth, -1, depth_sigma, space_sigma);
            depthImage.setTo(std::numeric_limits<float>::quiet_NaN(), invalidDepthMask);
            tm_bilateral_filter.stop();
            cout << "Time filter " << tm_bilateral_filter.getTimeSec() << endl;
#endif
            //timestamps.push_back( timestap );
        }

        {
            Mat gray;
            cvtColor(colorImage, gray, COLOR_BGR2GRAY);
            frame_curr->image = gray;
            frame_curr->depth = depthImage;
             
            Mat Rt;
            if(!Rts.empty())
            {
              MyTickMeter tm;
              tm.start();
              gtm.start();
              bool res = odometry->compute(frame_curr, frame_prev, Rt);
              gtm.stop();
              tm.stop();
              //count++;
              //cout << "Time " << tm.getTimeSec() << endl;
#if BILATERAL_FILTER
              cout << "Time ratio " << tm_bilateral_filter.getTimeSec() / tm.getTimeSec() << endl;
#endif
              if(!res)
                Rt = Mat::eye(4,4,CV_64FC1);
            }

            if( Rts.empty() )
              Rts.push_back(Mat::eye(4,4,CV_64FC1));
            else
            {
	      Mat& prevRt = *Rts.rbegin();
              //cout << "Rt " << Rt << endl;
              Rts[0] = ( prevRt * Rt );
            }

	    //10フレームごと変換して表示
	    /*Mat rot = Rts[count](Rect(0, 0, 3, 3)).t();
            Mat tvec = Rts[count](Rect(3, 0, 1, 3)).t();
	    if (count % 10 == 0){
		int downSamplingNum = 4; //e.g. 4
		Mat image2(image.rows / downSamplingNum, image.cols / downSamplingNum, CV_8UC3);
		resize(image, image2, image2.size(), 0, 0, INTER_LINEAR);
		Mat pCloud(image.rows / downSamplingNum, image.cols / downSamplingNum, CV_64FC3);

		for (int y = 0; y < 480; y += downSamplingNum){
			for (int x = 0; x < 640; x += downSamplingNum){
				if (depth.at<float>(y, x) < 8.0 && depth.at<float>(y, x) > 0.4){
					//RGB-D Dataset
					Mat pmat(1, 3, CV_64F);
					pmat.at<double>(0, 2) = (double)depth.at<float>(y, x);
					pmat.at<double>(0, 0) = (x - cx) * pmat.at<double>(0, 2) / fx;
					pmat.at<double>(0, 1) = (y - cy) * pmat.at<double>(0, 2) / fy;
					pmat = (pmat)*rot + tvec;
					Point3d p(pmat);
					pCloud.at<Point3d>(y / downSamplingNum, x / downSamplingNum) = p;
					pmat.release();
				}
				else{
					//RGB-D Dataset
					pCloud.at<Vec3d>(y / downSamplingNum, x / downSamplingNum) = Vec3d(0.f, 0.f, 0.f);
				}
			}
		}

		viz::WCloud wcloud(pCloud, image2);
		string myWCloudName = "CLOUD" + NumToString(count);
		myWindow.showWidget(myWCloudName, wcloud, cloud_pose_global);

		pCloud.release();
		image2.release();
	    }

	    cam_pose = cv::Affine3d(rot.t(), tvec);
	    viz::WCameraPosition cpw(0.1); // Coordinate axes
   	    viz::WCameraPosition cpw_frustum(cv::Matx33d(cameraMatrix), 0.1, viz::Color::white()); // Camera frustum
	    string widgetPoseName = "CPW" + NumToString(count);
	    string widgetFrustumName = "CPW_FRUSTUM" + NumToString(count);
	    myWindow.showWidget(widgetPoseName, cpw, cam_pose);
	    //myWindow.showWidget(widgetFrustumName, cpw_frustum, cam_pose);
	    myWindow.spinOnce(1, true);

	    rot.release();
	    tvec.release(); */

            if(!frame_prev.empty())
              frame_prev->release();
            std::swap(frame_prev, frame_curr);
        }
        kinect_point = writeResults(Rts);
        //}
        //sita = getAngleofVec(kinect_point, target_point);


        //Driving
        //if (flag_goal == 1 && automatic == 1) {
        depthValue = showDistance(/*depthImage,*/ depthFrame );


	//forward
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x >= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+1560)) {

	  if (target_point.x > 0 && target_point.x > 0.7 && target_point.y < 0.7) {
            circle(colorImage, Point(320, 240), 20, Scalar(0, 250, 0), 3, 4);
            len = write(fd, &forwardRight, sizeof(forwardRight));
	    //forwardFlag = false;
            if(len==0) {
              continue;
            }
            if(!len) { // I/Oエラー
              //printf("%s: ERROR\n", argv[0]);
              perror("");
              exit(2);
            }
	  }

	  if (target_point.x > -0.7 && target_point.y > 0 && target_point.y > 0.7) {
            circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 250), 3, 4);
	    len = write(fd, &forward, sizeof(forward));
            if(len==0) {
              continue;
            }
            if(!len) { // I/Oエラー
              //printf("%s: ERROR\n", argv[0]);
              perror("");
              exit(2);
            }
	  }

	  if (target_point.x < 0 && target_point.x < -0.7 && target_point.y > -0.7) {
            circle(colorImage, Point(320, 240), 20, Scalar(250, 0, 0), 3, 4);
            len = write(fd, &forwardLeft, sizeof(forwardLeft));
	    //forwardFlag = false;
            if(len==0) {
              continue;
            } 
            if(!len) { // I/Oエラー
              //printf("%s: ERROR\n", argv[0]);
              perror("");
              exit(2);
            }
	  }

	  if (target_point.x < 0.7 && target_point.y < 0 && target_point.y < -0.7) {
            circle(colorImage, Point(320, 240), 20, Scalar(0, 250, 0), 3, 4);
            len = write(fd, &forwardRight, sizeof(forwardRight));
	    //forwardFlag = false;
            if(len==0) {
              continue;
            }
            if(!len) { // I/Oエラー
              //printf("%s: ERROR\n", argv[0]);
              perror("");
              exit(2);
            }
	  }

          /*circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 250), 3, 4);
	  len = write(fd, &forward, sizeof(forward));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }*/
        }

	//Forward
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x >= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(0, 250, 0), 3, 4);
          len = write(fd, &forward, sizeof(forward));
	  //forwardFlag = false;
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
        }

	//Right
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x >= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(0, 250, 0), 3, 4);
          len = write(fd, &forwardRight, sizeof(forwardRight));
	  //forwardFlag = false;
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
        }

	//Right
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x <= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(0, 250, 0), 3, 4);
          len = write(fd, &forwardRight, sizeof(forwardRight));
	  //forwardFlag = false;
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
        }

	//Right
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x <= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(0, 250, 0), 3, 4);
          len = write(fd, &forwardRight, sizeof(forwardRight));
	  //forwardFlag = false;
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
        }

	//Right
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x <= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          len = write(fd, &forwardRight, sizeof(forwardRight));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);

            perror("");
            exit(2);
          }
        }

	//Left
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x >= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(250, 0, 0), 3, 4);
          len = write(fd, &forwardLeft, sizeof(forwardLeft));
	  //forwardFlag = false;
          if(len==0) {
            continue;
          } 
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
        }

	//Left
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x <= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x > 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(250, 0, 0), 3, 4);
          len = write(fd, &forwardLeft, sizeof(forwardLeft));
	  //forwardFlag = false;
          if(len==0) {
            continue;
          } 
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);
            perror("");
            exit(2);
          }
        }

/*
	//stop
        if ((find_if(depthValue.pointIndex+661, depthValue.pointIndex+900, [](int x){return x >= 0 && x <= 700;}) != depthValue.pointIndex+900) && (find_if(depthValue.pointIndex+1, depthValue.pointIndex+660, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+660) && (find_if(depthValue.pointIndex+901, depthValue.pointIndex+1560, [](int x){return x > 0 && x <= 700;}) != depthValue.pointIndex+1560)) {

          circle(colorImage, Point(320, 240), 20, Scalar(0, 0, 0), 3, 4);
          len = write(fd, &stop, sizeof(stop));
          if(len==0) {
            continue;
          }
          if(!len) { // I/Oエラー
            //printf("%s: ERROR\n", argv[0]);

            perror("");
            exit(2);
          }
        }
*/
      }


      //Send to server
      resize(colorImage, frame, Size(320, 240));
      if(frame.size().width==0)	continue; // simple integrity check; skip erroneous data...
      resize(frame, send, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_LINEAR);
      vector < int > compression_params;
      compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
      compression_params.push_back(jpegqual);

      imencode(".jpg", send, encoded, compression_params);
      int total_pack = 1 + (encoded.size() - 1) / PACK_SIZE;

      int ibuf[1];
      ibuf[0] = total_pack;
      udp_sock.sendTo(ibuf, sizeof(int), servAddress, servPort);

      for (int i = 0; i < total_pack; i++) {
	udp_sock.sendTo( & encoded[i * PACK_SIZE], PACK_SIZE, servAddress, servPort);
      }
      waitKey(FRAME_INTERVAL);


      /* Capture depth image */
      /*if (iJpeg < 100000) {
	sprintf(capturename, "I1_%06d.png", iJpeg);
        imwrite(capturename, colorImage);
	iJpeg++;
      }*/


      /* Show the Image and save the video */
      imshow("Depth", depthImage);
      imshow("Color", colorImage);
      colorWriter << colorImage;
      //depthWriter << depthImage;

      int key = waitKey( 10 );
      if ( key == 'q' ) {
        len = write(fd, &stop, sizeof(stop));
        if(len==0) {
          continue;
        }
        if(!len) { // I/Oエラー
          //printf("%s: ERROR\n", argv[0]);
          perror("");
          exit(2);
        }
        break;
      }

      /* For debug */
      driveCount += 1;
      //cout << " depthValue[" << 0 << "] " << depthValue.pointIndex[0] << endl;
      //cout << " depthValue[" << 1 << "] " << depthValue.pointIndex[1] << endl;
      //cout << " depthValue[" << 300 << "] " << depthValue.pointIndex[300] << endl;
      //cout << " depthValue[" << 720 << "] " << depthValue.pointIndex[720] << endl;
      //cout << " depthValue[" << 1140 << "] " << depthValue.pointIndex[1140] << endl;
      //cout << " driveCount " << driveCount << endl;
      //cout << " x :" << GazeX << " y :" << GazeY << endl;
      cout << " gazeCount " << gazeCount << endl;
      cout << " odometry.x = " << kinect_point.x * 10 << " odometry.y = " << kinect_point.y * 10 << endl;
      cout << " targetPoint.x = " << xx * 0.001 << " targetPoint.y = " << zz * 0.001 << endl;
      cout << " Disstance.x = " << target_point.x << " Disstance.y = " << target_point.y << endl;
      cout << " myPosition.x = " << kinect_point1.x << " myPosition.y = " << kinect_point1.y << endl;
      //cout << " sita " << sita << endl;
      //cout << "x.size = " << x.size() << "y.size = " << y.size() << endl;

      /* clock_t next_cycle = clock();
      double duration = (next_cycle - last_cycle) / (double) CLOCKS_PER_SEC;
      cout << "\teffective FPS:" << (1 / duration) << " \tkbps:" << (PACK_SIZE * total_pack / duration / 1024 * 8) << endl;
      cout << next_cycle - last_cycle;
      last_cycle = next_cycle; */

    }
  }


/*void showCenterDistance( Mat& depthImage, const openni::VideoFrameRef& depthFrame)
  {
    openni::VideoMode videoMode = depthStream.getVideoMode();

    int centerX = videoMode.getResolutionX() / 2;
    int centerY = videoMode.getResolutionY() / 2;
    int centerIndex = (centerY * videoMode.getResolutionX()) + centerX;

    unsigned short* depth = (unsigned short*)depthFrame.getData();

    stringstream ss;
    ss << "Center Point :" << depth[centerIndex];
    putText( depthImage, ss.str(), Point( 0, 50 ),
               FONT_HERSHEY_SIMPLEX, 1.0, Scalar( 255 ) );
 }*/

  /* Get the length of vector */
  double getVeclength(Point2d v) {
    return pow((v.x * v.x) + (v.y * v.y), 0.5);
  }

  double getVecdotproduct(Point2d vl, Point2d vr) {
    return ((vl.x * vr.x) + (vl.y * vr.y));
  }

  double getAngleofVec(Point2d kinect, Point2d target) {

    double length_kinect = getVeclength(kinect);
    double length_target = getVeclength(target);

    double cos_sita = getVecdotproduct(kinect, target) / (length_kinect * length_target);

    double sita = acos(cos_sita);

    sita = sita * 180.0 / 3.14159;

    return sita;
  }


  /* Convert depthstream to a form that can be displayed */
  Mat showDepthStream( const openni::VideoFrameRef& depthFrame )
  {
    /* Imaging a distance data(16bit) */
    Mat depthImage = Mat( depthFrame.getHeight(),
                                 depthFrame.getWidth(),
                                 CV_16UC1, (unsigned short*)depthFrame.getData() );
    
    /* Convert 0-10000mm to 0-255(8bit) */
    //depthImage.convertTo( depthImage, CV_8UC1, 255.0 / 10000 );
    //depthImage.convertTo( depthImage, CV_16UC1, 1.f / 8500.f );

    /* Display distance of the center point */
    //showCenterDistance( depthImage, depthFrame );

    return depthImage;
  }

  struct depthIndex showDistance(/*Mat& depthImage,*/ const openni::VideoFrameRef& depthFrame)
  {
    openni::VideoMode videoMode = depthStream.getVideoMode();
    struct depthIndex Index;
    unsigned short* depth = (unsigned short*)depthFrame.getData();
    int depthCenterX = 270;
    int depthLeftX   = 40;
    int depthRightX  = 360;
    int depthCenterY = 160;
    int depthLRY = 160;

    for (int i = 1; i <= 660; i++) {

      if (!(i >= 240)) {
        Index.pointIndex[i+660] = depth[(depthCenterY * videoMode.getResolutionX()) + depthCenterX];	//Forward
      }
      Index.pointIndex[i] = depth[(depthLRY * videoMode.getResolutionX()) + depthLeftX];		//Left
      Index.pointIndex[i+900] = depth[(depthLRY * videoMode.getResolutionX()) + depthRightX];		//Right

      //cout << " depthValue[" << i << "] " << Index.pointIndex[i] << endl;
      //circle(depthImage, Point(depthCenterX, depthCenterY), 3, Scalar(0, 0, 200), 3, 4);
      //circle(depthImage, Point(depthLeftX, depthLRY), 3, Scalar(0, 0, 200), 3, 4);
      //circle(depthImage, Point(depthRightX, depthLRY), 3, Scalar(0, 0, 200), 3, 4);

      depthLeftX  += 10;
      depthRightX += 10;
      if (!(i >= 240)) {
	depthCenterX  += 10;
      }

      if (((i % 8) == 0) && !(i >= 240)) {
        depthCenterY += 10;
        depthCenterX = 270;
      }

      if ((i % 22) == 0) {
        depthLRY += 10;
        depthLeftX   = 40;
	depthRightX  = 360;
      }
    }
    return Index;
  }

  /* Convert colorstream to a form that can be displayed */
  Mat showColorStream( const openni::VideoFrameRef& colorFrame )
  {
    /* Convert to form that is OpenCV */
    Mat colorImage = Mat( colorFrame.getHeight(), colorFrame.getWidth(), CV_8UC3, (unsigned char*)colorFrame.getData() );
    
    /* Convert BGR to RGB */
    cvtColor( colorImage, colorImage, CV_RGB2BGR );
    return colorImage;
  }


  void serial_init(int fd) {
    struct termios tio;
    memset(&tio, 0, sizeof(tio));
    tio.c_cflag = CS8 | CLOCAL | CREAD;
    tio.c_cc[VTIME] = 100;
    // ボーレートの設定
    cfsetispeed(&tio, BAUD_RATE);
    cfsetospeed(&tio, BAUD_RATE);
    // デバイスに設定を行う
    tcsetattr(fd, TCSANOW, &tio);
  }


private:

  openni::Device device;

  openni::VideoStream colorStream;
  openni::VideoStream depthStream;
  vector<openni::VideoStream*> streams;

};

int main(void)
{
  try {
    GrabDetectorSample app;
    app.initialize();
    app.run();
  }
  catch ( exception& ) {
    std::cout << openni::OpenNI::getExtendedError() << std::endl;
  }
}

