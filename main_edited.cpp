#include <iostream>
#include <list>
#include <stdexcept>
#include <OpenNI.h>
#include <opencv.hpp>

using namespace std;
using namespace cv;

class GrabDetectorSample
{
public:

  //initialize
  void initialize()
  {
    initOpenNI();
  }

  // OpenNI's initialize
  void initOpenNI()
  {
    openni::OpenNI::getVersion();

    openni::OpenNI::initialize();

    device.open( openni::ANY_DEVICE );

    // Requires VGA depth & color input, with both depth-color registration and time sync 
    // enabled.  See included sample for setting this up. 

    // Create color stream
    colorStream.create( device, openni::SENSOR_COLOR );
    initStream( colorStream );
    colorStream.start();

    // Create depth stream
    depthStream.create( device, openni::SENSOR_DEPTH );
    initStream( depthStream );
    depthStream.start();

    // Synchronism Depth & Color
    device.setDepthColorSyncEnabled( true );

    // Fit the Depth to Color
    device.setImageRegistrationMode( openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR );

    // Put together stream
    streams.push_back( &colorStream );
    streams.push_back( &depthStream );
  }

  // Initialize the OpenNI stream
  void initStream( openni::VideoStream& stream )
  {
    openni::VideoMode videoMode = stream.getVideoMode();
    videoMode.setFps( 30 );
    videoMode.setResolution( 640, 480 );
    stream.setVideoMode( videoMode );
  }


  // Main loop
  void run()
  {

    bool running = true;
    Mat depthImage(480, 640, CV_16UC1);
    Mat colorImage(480, 640, CV_8UC3);
    Mat showdepthImage(240, 320, CV_16UC1);
    Mat showcolorImage(240, 320, CV_8UC3);

    while ( running ) {
      int changedIndex;
      openni::OpenNI::waitForAnyStream( &streams[0], streams.size(), &changedIndex );

      // Get the frame of depth & color
      openni::VideoFrameRef depthFrame;
      depthStream.readFrame( &depthFrame );
      openni::VideoFrameRef colorFrame;
      colorStream.readFrame( &colorFrame );      

      depthImage = showDepthStream( depthFrame );
      colorImage = showColorStream( colorFrame );

      resize(colorImage, showcolorImage, Size(240, 320));
      resize(depthImage, showdepthImage, Size(240, 320));

      // imshow( "Depth", depthImage );
      // imshow( "Coror", colorImage );

      // imwrite("img.jpg", colorImage);

      imshow("Depth", showdepthImage);
      imshow("Color", showcolorImage);

      int key = waitKey( 10 );
      if ( key == 'q' ) {
        break;
      }
    }
  }


  void showCenterDistance( Mat& depthImage, const openni::VideoFrameRef& depthFrame)
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
 }


  // Convert depthstream to a form that can be displayed
  Mat showDepthStream( const openni::VideoFrameRef& depthFrame )
  {
    // Imaging a distance data(16bit)
    Mat depthImage = Mat( depthFrame.getHeight(),
                                 depthFrame.getWidth(),
                                 CV_16UC1, (unsigned short*)depthFrame.getData() );
    
    // Convert 0-10000mm to 0-255(8bit)
    depthImage.convertTo( depthImage, CV_8U, 255.0 / 10000 );
    
    // Display distance of the center point
    showCenterDistance( depthImage, depthFrame );

    return depthImage;
  }


  // Convert colorstream to a form that can be displayed
  Mat showColorStream( const openni::VideoFrameRef& colorFrame )
  {
    // Convert to form that is OpenCV
    Mat colorImage = Mat( colorFrame.getHeight(), colorFrame.getWidth(), CV_8UC3, (unsigned char*)colorFrame.getData() );
    
    // Convert BGR to RGB
    cvtColor( colorImage, colorImage, CV_RGB2BGR );
    return colorImage;
  }

private:

  openni::Device device;

  openni::VideoStream colorStream;
  openni::VideoStream depthStream;
  vector<openni::VideoStream*> streams;

};

int main(int argc, char * argv[])
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
