#pragma once

/*

BASED off of the original ofxKinect example !
github.com/ofTheo/ofxKinect

I just added ofxUI and made some stuff a little easier to digest

*/

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxKinect.h"
#include "ofxSimpleMask.h"
#include "ofxUI.h"
#include "ofxCv.h"
using namespace ofxCv;
using namespace cv;
#include "Clone.h"
#include "ofxFaceTrackerThreaded.h"
#include "ofxFaceTracker.h"

// uncomment this to read from two kinects simultaneously
//#define USE_TWO_KINECTS

class ofApp : public ofBaseApp {
public:

	void setup();
	void update();
	void draw();
	void exit();

	void keyPressed(int key);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void windowResized(int w, int h);

	void setup_ofxUI();
	void analyzeTwitter();
	void drawStrings(int num);
	void drawShaderFragment();
	void drawShaderFragment2();
	void drawPointCloud();
	void pointCloud();
	void findFace();
	void loadFace(string face);
	void dragEvent(ofDragInfo dragInfo);
	void makeJudge();
	void drawOutput();

	int resizedWidth = 2800;
	int resizedHeight = 1920;

	ofxKinect kinect;
	ofEasyCam easyCam;

#ifdef USE_TWO_KINECTS
	ofxKinect kinect2;
#endif

	ofxCvColorImage colorImg;
	ofxCvGrayscaleImage grayImage; // grayscale depth image
	ofxCvGrayscaleImage grayThreshNear; // the near thresholded image
	ofxCvGrayscaleImage grayThreshFar; // the far thresholded image

	ofxCvContourFinder contourFinder;
	ofxCvHaarFinder haarFinder;

	bool bThreshWithOpenCV;
	bool bDrawPointCloud;

	int nearThreshold;
	int farThreshold;
	bool bKinectOpen;

	int angle;
	ofVec3f normalizedRange;

	//added for ofxUI
	ofxUICanvas *gui;
	float guiWidth;
	void guiEvent(ofxUIEventArgs &e);

	float minBlobSize, maxBlobSize;
	float pointCloudMinZ, pointCloudMaxZ;

	ofShader shader;
	ofShader shader2;
	ofxSimpleMask mask;

	ofFbo shaderFbo;
	ofFbo shaderFbo2;
	ofFbo maskFbo;
	ofFbo trailsFbo;
	float trailsAlpha;
	//Analyze Twitter
	ofFile mytextFile;
	string content;
	string username;
	string subString;
	string upperString;
	string reportContent;
	ofTrueTypeFont font;
	ofTrueTypeFont subFont;
	ofRectangle maxCur;
	ofImage usernameFrame;

	int upbeat;
	int worried;
	int angry;
	int depressed;
	int pluggedIn;
	int personable;
	int arrogant;
	int spacy;
	int analytic;
	int sensory;
	int inTheMoment;
	string s1, s2, s3;
	int emotionNum[12];
	string emotionName[12];
	float femalIndex, maleIndex;

	int lineHeight = 30;
	string typedLine;
	float           nextLetterTime;
	int             lineCount;
	int             letterCount;
	vector <string> seussLines;
	int yPos = 150;
	ofTrueTypeFont contentFont;

	int maxDis = 200;
	int posX[3];
	int posY[3];

	int stage = -1;
	int numPart;
	int lastTime;

	ofxFaceTrackerThreaded camTracker;
	ofVideoGrabber cam;

	ofxFaceTracker srcTracker;
	ofImage src;
	vector<ofVec2f> srcPoints;

	bool cloneReady;
	Clone clone;
	ofFbo srcFboFace, maskFboFace;

	ofDirectory faces;
	int currentFace;

	ofImage halo;
	ofImage halo2;
};
