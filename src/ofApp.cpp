#include "ofApp.h"
ofImage colorImage;
using namespace ofxCv;
using namespace cv;

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetLogLevel(OF_LOG_VERBOSE);
	//ofToggleFullscreen();
	cout << ofGetWindowPositionX() << endl;
	cout << ofGetWindowPositionY() << endl;

	// enable depth->video image calibration
	kinect.setRegistration(true);
	
	kinect.init();
	kinect.open();

	cam.setDeviceID(0);
	cam.initGrabber(1920, 1080);
	cam.setUseTexture(true);

	colorImg.allocate(kinect.width, kinect.height);
	grayImage.allocate(kinect.width, kinect.height);
	grayThreshNear.allocate(kinect.width, kinect.height);
	grayThreshFar.allocate(kinect.width, kinect.height);
	haarFinder.setup("haarcascade_frontalface_default.xml");

	ofSetFrameRate(30);

	bKinectOpen = true;
	bDrawPointCloud = true;

	setup_ofxUI();

	//FragmentShader
	mask.setup("shader/composite_rgb", ofRectangle(0, 0, ofGetWidth(), ofGetHeight()));
	//shader.load("shader/neuralSpace.vert", "shader/neuralSpace.frag");
	shader.load("shader/neonLightTunnel.vert", "shader/SpikeField.frag");
	shader2.load("shader/neuralSpace.vert", "shader/neuralSpace.frag");
	shaderFbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA);
	shaderFbo2.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA);
	shaderFbo.begin();
	ofClear(1, 1, 1, 0);
	shaderFbo.end();
	shaderFbo2.begin();
	ofClear(1, 1, 1, 0);
	shaderFbo2.end();

	maskFbo.allocate(ofGetWidth(), ofGetHeight());
	maskFbo.begin();
	ofClear(1, 1, 1, 0);
	maskFbo.end();

	//Alpha Fbo
	trailsAlpha = 48.0f;
	trailsFbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA32F_ARB);
	trailsFbo.begin();
	ofClear(0, 0, 0, 0);
	trailsFbo.end();

	//AnalyzeTwitter
	usernameFrame.loadImage("usernameFrame.png");
	font.load("20th Centenary Faux.ttf", 48);
	subFont.load("814yzx.ttf", 24);
	username = "";
	subString = "Please enter your twitter account here.";
	femalIndex = 0;
	maleIndex = 0;
	contentFont.load("20th Centenary Faux.ttf", 24);

	//TrackFace
	maxCur.width = -1;
	stage = -1;
	numPart = 0;

	//FaceSubstitute
	ofSetVerticalSync(true);
	cloneReady = false;
	clone.setup(resizedWidth,resizedHeight);
	//clone.setup(cam.getWidth(), cam.getHeight());
	ofFbo::Settings settings;
	settings.width = resizedWidth;
	settings.height = resizedHeight;
	maskFboFace.allocate(settings);
	srcFboFace.allocate(settings);
	camTracker.setup();
	srcTracker.setup();
	srcTracker.setIterations(25);
	srcTracker.setAttempts(4);

	faces.allowExt("jpg");
	faces.allowExt("png");
	faces.listDir("faces");
	currentFace = 0;
	if (faces.size() != 0) {
		loadFace(faces.getPath(currentFace));
	}

	halo.loadImage("halo.png");
	halo2.loadImage("halo2.png");
}

void ofApp::setup_ofxUI()
{
	//Setup ofxUI
	float dim = 24;
	float xInit = OFX_UI_GLOBAL_WIDGET_SPACING;
	guiWidth = 210 - xInit;

	gui = new ofxUICanvas(0, 0, guiWidth + xInit, ofGetHeight());
	gui->addWidgetDown(new ofxUILabel("KINECT PARAMS", OFX_UI_FONT_MEDIUM));
	gui->addWidgetDown(new ofxUIRangeSlider(guiWidth, dim, 0.0, 255.0, farThreshold, nearThreshold, "DEPTH RANGE"));
	gui->addWidgetDown(new ofxUIRangeSlider("BLOB SIZE", guiWidth, dim, 0.0, ((kinect.width * kinect.height) / 2), minBlobSize, maxBlobSize));
		//gui->addWidgetDown(new ofxUISlider("MOTOR ANGLE", guiWidth, dim, -30.0f, 30.0f, angle));
	gui->addWidgetDown(new ofxUIToggle("OPEN KINECT", bKinectOpen, dim, dim));
	if (stage == 1 || stage == 2) {
		gui->addWidgetDown(new ofxUIToggle("THRESHOLD OPENCV", bThreshWithOpenCV, dim, dim));
	}
	if (stage == 4) {
		gui->addWidgetDown(new ofxUIRangeSlider("Z RANGE", guiWidth, dim, 0, 2000, pointCloudMinZ, pointCloudMaxZ));
		gui->addWidgetDown(new ofxUIToggle("DRAW POINT CLOUD", bDrawPointCloud, dim, dim));
	}
	ofAddListener(gui->newGUIEvent, this, &ofApp::guiEvent);
	gui->loadSettings("GUI/kinectSettings.xml");
}

//--------------------------------------------------------------
void ofApp::update() {
	ofSetWindowTitle("MergedFunction - FPS:" + ofToString(ofGetElapsedTimef()));

	switch (stage)
	{
		//Find a new user
	case -1:
		cam.update();
		/*
		if (cam.isFrameNew()) {
			colorImage.setFromPixels(cam.getPixels());
			colorImage.resize(640, 360);
			haarFinder.findHaarObjects(colorImage);
		}*/
		break;
		//Enter username
	case 0:
		cam.update();
		break;
	case 1:
		kinect.update();
		if (kinect.isFrameNew()) {
			// load grayscale depth image from the kinect source
			grayImage.setFromPixels(kinect.getDepthPixels(), kinect.width, kinect.height);
			grayImage.mirror(false, true);
				// or we do it ourselves - show people how they can work with the pixels
				unsigned char * pix = grayImage.getPixels();
				int numPixels = grayImage.getWidth() * grayImage.getHeight();
				for (int i = 0; i < numPixels; i++) {
					if (pix[i] < nearThreshold && pix[i] > farThreshold) {
						pix[i] = 255;
					}
					else {
						pix[i] = 0;
					}
				}
			grayImage.flagImageChanged();
			contourFinder.findContours(grayImage, minBlobSize, maxBlobSize, 20, false);
		}
		break;
	case 2:
		kinect.update();
		cam.update();
		if (kinect.isFrameNew()) {
			// load grayscale depth image from the kinect source
			grayImage.setFromPixels(kinect.getDepthPixels(), kinect.width, kinect.height);
			grayImage.mirror(false, true);
			if (bThreshWithOpenCV) {
				grayThreshNear = grayImage;
				grayThreshFar = grayImage;
				grayThreshNear.threshold(nearThreshold, true);
				grayThreshFar.threshold(farThreshold);
				cvAnd(grayThreshNear.getCvImage(), grayThreshFar.getCvImage(), grayImage.getCvImage(), NULL);
			}
			else {
				// or we do it ourselves - show people how they can work with the pixels
				unsigned char * pix = grayImage.getPixels();
				int numPixels = grayImage.getWidth() * grayImage.getHeight();
				for (int i = 0; i < numPixels; i++) {
					if (pix[i] < nearThreshold && pix[i] > farThreshold) {
						pix[i] = 255;
					}
					else {
						pix[i] = 0;
					}
				}
			}
			grayImage.flagImageChanged();
			contourFinder.findContours(grayImage, minBlobSize, maxBlobSize, 20, false);
		}
		break;
		//ChangeFace
	case 3:
		kinect.update();
		cam.update();
		break;
	case 4:
		cam.update();
		if (cam.isFrameNew()) {
			camTracker.update(toCv(cam));
			cloneReady = camTracker.getFound();
			if (cloneReady) {
				ofMesh camMesh = camTracker.getImageMesh();
				camMesh.clearTexCoords();
				camMesh.addTexCoords(srcPoints);

				maskFboFace.begin();
				ofClear(0, 255);
				camMesh.draw();
				maskFboFace.end();

				srcFboFace.begin();
				ofClear(0, 255);
				src.bind();
				camMesh.draw();
				src.unbind();
				srcFboFace.end();

				clone.setStrength(16);
				clone.update(srcFboFace.getTextureReference(), cam.getTextureReference(), maskFboFace.getTextureReference());
			}
		}
	
		break;
	default:
		break;
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofBackground(0, 0, 0);
	ofSetColor(255, 255, 255);
	kinect.setCameraTiltAngle(-30.0f);
	//cout << stage << endl;
	switch (stage)
	{
		case -1:
			findFace();
			colorImage.draw(0, 0);
			cout << maxCur.width << endl;
			if (maxCur.width > 90) {
				stage++;
				nextLetterTime = ofGetElapsedTimeMillis();
				lineCount = 0;
				letterCount = 0;
				ofBuffer buffer = ofBufferFromFile("test.txt");
				if (buffer.size()) {
					for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
						string line = *it;
						if (line.empty() == false) {
							seussLines.push_back(line);
						}
						cout << line << endl;
					}
				}
			}
			break;
		case 0:
			glPushMatrix();
			glTranslatef(resizedWidth, 0, 0);
			glScalef(-1, 1, 1);
			glTranslatef(-65, 0, 0);
			cam.draw(0, 0, resizedWidth, resizedHeight);
			glPopMatrix();
			//cam.draw(0, 0);
			drawOutput();
			break;
		case 1:
			drawShaderFragment();		
		//	grayImage.draw(0, 0);
			halo.draw(875, 250);
			usernameFrame.draw(ofGetWidth() / 2 - 240, ofGetHeight() * 2 / 3 +100, 480, 150);
			font.drawString(username, ofGetWidth() / 2 - font.stringWidth(username) / 2, ofGetHeight() * 2 / 3+182);
			subFont.drawString(subString, ofGetWidth() / 2 - subFont.stringWidth(subString) / 2, 100 + ofGetHeight() * 2 / 3 +250);
			break;
		case 2:
			ofPushMatrix();
			ofTranslate(100, 0);
			drawShaderFragment2();
			//findFace();
			halo2.draw(875, 250);
			if (int(ofGetElapsedTimef())-lastTime>0 && int(ofGetElapsedTimef()) % 3 == 0) {
				posX[0] = ofRandom(1200 - maxDis, 1200 - 50); posY[0] = ofRandom(500 - maxDis, 500 - 50);
				posX[1] = ofRandom(1600 + 50, 1600 + maxDis); posY[1] = ofRandom(1200 + 50, 1200 + maxDis);
				posX[2] = ofRandom(1200 - maxDis, 1200 - 50); posY[2] = ofRandom(1200 + 50, 1200 + maxDis);
				numPart++;
				if (numPart > 5) stage++;
				lastTime = int(ofGetElapsedTimef());
			}
			//colorImage.draw(0, 0, resizedWidth, resizedHeight);			
		    drawStrings(numPart);
			ofPopMatrix();
			upperString = "Analyzing the structure...";
			subFont.drawString(upperString, ofGetWidth() / 2 - subFont.stringWidth(upperString) / 2, 150);
			break;
		case 3:
			pointCloudMinZ = 0.0f;
			pointCloudMaxZ = 1600.0f;
			ofPushMatrix();
			drawPointCloud();
			halo2.draw(875, 250);
			//findFace();
			if (int(ofGetElapsedTimef()) - lastTime>0 && int(ofGetElapsedTimef()) % 3 == 0) {
				posX[0] = ofRandom(1200 - maxDis, 1200 - 50); posY[0] = ofRandom(500 - maxDis, 500 - 50);
				posX[1] = ofRandom(1600 + 50, 1600 + maxDis); posY[1] = ofRandom(1200 + 50, 1200 + maxDis);
				posX[2] = ofRandom(1200 - maxDis, 1200 - 50); posY[2] = ofRandom(1200 + 50, 1200 + maxDis);
				numPart++;
				if (numPart > 10) {
					
					stage++;
					makeJudge();
				}
				lastTime = int(ofGetElapsedTimef());
			}
			drawStrings(numPart);
			upperString = "Rebuilding your appearance...";
			subFont.drawString(upperString, ofGetWidth() / 2 - subFont.stringWidth(upperString) / 2, 150);
			break;
		case 4:
			ofSetColor(255);
			if (src.getWidth() > 0 && cloneReady) {
				ofPushMatrix();
				ofTranslate(2800, 0);
				ofScale(-1.45, 1.68);
				clone.draw(0, 0);
				ofPopMatrix();
			}
			else {
			//	ofScale(1, 1);
				glPushMatrix();
				glTranslatef(resizedWidth, 0, 0);
				glScalef(-1, 1, 1);
				cam.draw(0, 0, resizedWidth, resizedHeight);
				glPopMatrix();
			}

			if (!camTracker.getFound()) {
				drawHighlightString("camera face not found", 10, 10);
			}
			if (src.getWidth() == 0) {
				drawHighlightString("drag an image here", 10, 30);
			}
			else if (!srcTracker.getFound()) {
				drawHighlightString("image face not found", 10, 30);
			}

			drawOutput();
			break;
	default:
		break;
	}
	
	//to debug to the screen use
	//ofDrawBitmapString( "STRING" , x , y ) ;

	cam.draw(0, 0, 320, 180);
	kinect.draw(0, 180);
}

void ofApp::loadFace(string face) {
	src.loadImage(face);
	if (src.getWidth() > 0) {
		srcTracker.update(toCv(src));
		srcPoints = srcTracker.getImagePoints();
	}
}

void ofApp::drawPointCloud() {
	ofBackground(0);
	ofSetColor(255);
	ofPushMatrix();
	//ofTranslate(guiWidth + 10, 0);
	if (bDrawPointCloud) {

			easyCam.begin();
			  ofPushMatrix();
			  ofTranslate(2800, 0);
				ofScale(-3.2, 3.2);
				ofTranslate(-35, 0);
				pointCloud();
			  ofPopMatrix();
			easyCam.end();

	}
	else {
		// draw from the live kinect
		kinect.drawDepth(10, 10, 400, 300);
		kinect.draw(420, 10, 400, 300);
	}

	ofPopMatrix();
}

void ofApp::pointCloud() {
	int w = 640;
	int h = 480;
	ofMesh mesh;
	mesh.setMode(OF_PRIMITIVE_POINTS);

	float normalizedZ = ofNoise(ofGetElapsedTimef()) * 0.35 + 0.5;
	float currentZ = ofLerp(pointCloudMinZ, pointCloudMaxZ, normalizedZ);
	float minZ = ofLerp(pointCloudMinZ, pointCloudMaxZ, normalizedZ - normalizedRange.z);
	float maxZ = ofLerp(pointCloudMinZ, pointCloudMaxZ, normalizedZ + normalizedRange.z);

	float normalizedY = ofNoise(ofGetElapsedTimef() * 0.25) * 0.5 + 0.5;
	float currentY = ofLerp(0, kinect.getHeight(), normalizedY);
	float minY = ofLerp(0, kinect.getHeight(), normalizedY - normalizedRange.y);
	float maxY = ofLerp(0, kinect.getHeight(), normalizedY + normalizedRange.y);

	float normalizedX = ofNoise(ofGetElapsedTimef() * 0.125) * 0.5 + 0.5;
	float currentX = ofLerp(0, kinect.getHeight(), normalizedX);
	float minX = ofLerp(0, kinect.getWidth(), normalizedX - normalizedRange.x);
	float maxX = ofLerp(0, kinect.getWidth(), normalizedX + normalizedRange.x);

	int step = 2;
	ofColor offset = ofColor::fromHsb(sin(ofGetElapsedTimef())* 128.0f + 128.0f, 255, 255);
	for (int y = 0; y < h; y += step) {
		for (int x = 0; x < w; x += step) {
			if (kinect.getDistanceAt(x, y) > 0) {

				ofVec3f vertex = kinect.getWorldCoordinateAt(x, y);

				if (vertex.z > pointCloudMinZ && vertex.z < pointCloudMaxZ)
				{
					mesh.addVertex(vertex);
					ofColor col = kinect.getColorAt(x, y);
					int hue = col.getHue();
					hue += offset.getHue();
					if (hue > 255)
						hue = 255 - hue;
					ofColor newColor = ofColor::fromHsb(hue, 255, 255);
					mesh.addColor(newColor);
				}
			}
		}
	}

	ofEnableBlendMode(OF_BLENDMODE_ADD);
	glPointSize(3);
	ofPushMatrix();
	ofTranslate(900, 0);
	ofScale(0.8, -0.8, -1);
	ofTranslate(0, 0, -1000); // center the points a bit
	glEnable(GL_DEPTH_TEST);
	mesh.drawVertices();
	glDisable(GL_DEPTH_TEST);
	ofPopMatrix();
	ofEnableBlendMode(OF_BLENDMODE_ALPHA);
}

void ofApp::drawShaderFragment() {
	ofSetColor(255);
	ofPushMatrix(); //
	shaderFbo.begin();
	shader.begin();
	shader.setUniform1f("time", ofGetElapsedTimef());
	shader.setUniform2f("resolution", ofGetWidth(), ofGetHeight());
	shader.setUniform1f("spaceMovement", 25.0f);
	shader.setUniform1f("flowAmount", 0.0005);
	ofSetColor(255);
	ofRect(0, 0, ofGetWidth(), ofGetHeight());
	shader.end();
	ofSetColor(255);
	shaderFbo.end();


	maskFbo.begin();
	ofClear(1, 1, 1, 0);
	grayImage.draw(0, 0, ofGetWidth(), ofGetHeight());
	maskFbo.end();

	ofPopMatrix();

	ofSetColor(255);
	mask.drawMask(shaderFbo.getTextureReference(), maskFbo.getTextureReference(), 0, 0, 1.0f);

}

void ofApp::drawShaderFragment2() {
	ofSetColor(255);
	ofPushMatrix(); //
		shaderFbo2.begin();
			shader2.begin();
				shader2.setUniform1f("time", ofGetElapsedTimef());
				shader2.setUniform2f("resolution", ofGetWidth(), ofGetHeight());
				shader2.setUniform1f("spaceMovement", 25.0f);
				shader2.setUniform1f("flowAmount", 0.0005);
				ofSetColor(255);
				ofRect(0, 0, ofGetWidth(), ofGetHeight());
			shader2.end();
			ofSetColor(255);
		shaderFbo2.end();


		maskFbo.begin();
			ofClear(1, 1, 1, 0);
			grayImage.draw(0, 0, ofGetWidth(), ofGetHeight());
		maskFbo.end();

	ofPopMatrix();

	ofSetColor(255);
	mask.drawMask(shaderFbo2.getTextureReference(), maskFbo.getTextureReference(), 0, 0, 1.0f);

}
//--------------------------------------------------------------
void ofApp::guiEvent(ofxUIEventArgs &e)
{
	string name = e.widget->getName();
	int kind = e.widget->getKind();

	if (name == "DEPTH RANGE")
	{
		ofxUIRangeSlider *slider = (ofxUIRangeSlider *)e.widget;
		farThreshold = slider->getScaledValueLow();
		nearThreshold = slider->getScaledValueHigh();
	}

	if (name == "BLOB SIZE")
	{
		ofxUIRangeSlider *slider = (ofxUIRangeSlider *)e.widget;
		minBlobSize = slider->getScaledValueLow();
		maxBlobSize = slider->getScaledValueHigh();
	}

	if (name == "THRESHOLD OPENCV")
	{
		ofxUIToggle *toggle = (ofxUIToggle *)e.widget;
		bThreshWithOpenCV = toggle->getValue();
	}


	if (name == "OPEN KINECT")
	{
		ofxUIToggle *toggle = (ofxUIToggle *)e.widget;
		bKinectOpen = toggle->getValue();
		if (bKinectOpen == true)
			kinect.open();
		else
			kinect.close();
	}
	/*
	if (name == "Z RANGE")
	{
		ofxUIRangeSlider *slider = (ofxUIRangeSlider *)e.widget;
		pointCloudMinZ = slider->getScaledValueLow();
		pointCloudMaxZ = slider->getScaledValueHigh();
	}
	*/
	if (name == "DRAW POINT CLOUD")
	{
		ofxUIToggle *toggle = (ofxUIToggle *)e.widget;
		bDrawPointCloud = toggle->getValue();
	}

	gui->saveSettings("GUI/kinectSettings.xml");
}

void ofApp::drawStrings(int num) {
	
	ofPushMatrix();
	trailsFbo.begin();
		ofEnableAlphaBlending();
		ofFill();
		ofSetColor(0, 0, 0, trailsAlpha);
		ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
		ofSetColor(255);
		if (num > 0) {
			string ss = emotionName[num] + ": " + ofToString(emotionNum[num]);
			subFont.drawString(ss, posX[0] - subFont.stringWidth(ss) / 2, posY[0]);
			//cout << posX[0] << endl;
			ofDrawLine(posX[0], posY[0], ofRandom(1200, 1400), ofRandom(600, 800));
			subFont.drawString(ss, posX[1] - subFont.stringWidth(ss) / 2, posY[1]);
			ofDrawLine(posX[1], posY[1], ofRandom(1200, 1400), ofRandom(800, 1000));
			subFont.drawString(ss, posX[2] - subFont.stringWidth(ss) / 2, posY[2]);
			ofDrawLine(posX[2], posY[2], ofRandom(1400, 1600), ofRandom(800, 1000));
		}
		ofDisableAlphaBlending();
	trailsFbo.end();
	trailsFbo.draw(0, 0);

	subFont.drawString(subString, ofGetWidth() / 2 - subFont.stringWidth(subString) / 2, 100 + ofGetHeight() * 4 / 5);

}

void ofApp::findFace() {
	if (int(ofGetElapsedTimef()) % 3 == 0) {
		maxCur.width = -1;
		if (cam.isFrameNew()) {
			//Find a Face			
			colorImage.setFromPixels(cam.getPixels());
			colorImage.resize(640, 360);
			//colorImage.resize(ofGetWidth(), ofGetHeight());
			haarFinder.findHaarObjects(colorImage);
			if (haarFinder.blobs.size() > 0) {
				for (unsigned int i = 0; i < haarFinder.blobs.size(); i++) {
					ofRectangle cur = haarFinder.blobs[i].boundingRect;
					if (cur.width > maxCur.width && cur.width > 50) {
						maxCur = cur;
					}
				}
			}
			ofNoFill();
			//	ofDrawRectangle(maxCur.x, maxCur.y, maxCur.width, maxCur.height);
		}

		if (stage == 2 || stage == 3) {
			if (kinect.isFrameNew()) {
				//Find a Face			
				colorImage.setFromPixels(kinect.getPixels());
				colorImage.resize(640, 360);
				//colorImage.resize(ofGetWidth(), ofGetHeight());
				haarFinder.findHaarObjects(colorImage);
				if (haarFinder.blobs.size() > 0) {
					for (unsigned int i = 0; i < haarFinder.blobs.size(); i++) {
						ofRectangle cur = haarFinder.blobs[i].boundingRect;
						if (cur.width > maxCur.width && cur.width > 50) {
							maxCur = cur;
						}
					}
				}
				ofNoFill();
				//	ofDrawRectangle(maxCur.x, maxCur.y, maxCur.width, maxCur.height);
			}
		}
		//colorImage.draw(0, 0, resizedWidth, resizedHeight);
	}
}

void ofApp::drawOutput() {
	usernameFrame.draw(ofGetWidth() / 2 - 400, 100, 800, seussLines.size()*lineHeight + 50);
	if (lineCount < seussLines.size()) {
		typedLine = seussLines[lineCount].substr(0, letterCount);

		ofSetColor(255);

		for (int i = 0; i < lineCount; i++) {
			contentFont.drawString(seussLines[i], (ofGetWidth() - contentFont.stringWidth(seussLines[i])) / 2, 150 + (lineHeight * i));
		}
		contentFont.drawString(typedLine, (ofGetWidth() - contentFont.stringWidth(typedLine)) / 2, yPos);

		float time = ofGetElapsedTimeMillis() - nextLetterTime;

		if (time > 9) {
			if (letterCount < (int)seussLines[lineCount].size()) {
				letterCount++;
				nextLetterTime = ofGetElapsedTimeMillis();
			}
			else {
				if (time > 300) {
					nextLetterTime = ofGetElapsedTimeMillis();
					letterCount = 0;
					lineCount++;
					yPos += lineHeight;
					// lineCount %= seussLines.size();
				}
			}
		}
	}
	else {
		for (int i = 0; i < seussLines.size(); i++) {
			contentFont.drawString(seussLines[i], (ofGetWidth() - contentFont.stringWidth(seussLines[i])) / 2, 150 + (lineHeight * i));
		}
		if (stage == 0) {
			findFace();
			cout << maxCur.width << endl;
			if (maxCur.width > 105) {
				stage++;
				seussLines.clear();
				yPos = 150;
			}
		}
	}
}

void ofApp::makeJudge() {
	if (pluggedIn > 50) femalIndex += 0.5;
	if (personable > 50) femalIndex += 0.5;
	if (arrogant > 50) maleIndex += 1;
	if (spacy > 50) femalIndex += 0.5;
	if (analytic > 50) maleIndex += 1;
	if (sensory > 50) femalIndex += 0.5;

	if (femalIndex > 1 || (femalIndex < 1 && maleIndex < 1)) {
		if (upbeat < 50 && worried < 50 && angry < 50 && depressed < 50) {
			//female normal
			currentFace = 2;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("f_normal.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (upbeat > 50 && upbeat > worried&&upbeat > angry&&upbeat > depressed) {
			//female smile
			currentFace = 3;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("f_upbeat.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (worried > 50 && worried > upbeat&&worried > angry&&worried > depressed) {
			//femail worried
			currentFace = 4;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("f_worried.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (angry > 50 && angry > upbeat&&angry > worried&&angry > depressed) {
			//female angry
			currentFace = 0;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("f_angry.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (depressed > 50 && depressed > upbeat&&depressed > worried&&depressed > angry) {
			//female depressed
			currentFace = 1;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("f_depressed.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
	}
	if (maleIndex > 1) {
		seussLines.clear();
		yPos = 150;
		if (upbeat < 50 && worried < 50 && angry < 50 && depressed < 50) {
			//male normal
			currentFace = 10;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("m_normal.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (upbeat > 50 && upbeat > worried&&upbeat > angry&&upbeat > depressed) {
			//male smile
			currentFace = 12;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("m_upbeat.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (worried > 50 && worried > upbeat&&worried > angry&&worried > depressed) {
			//male worried
			currentFace = 14;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("m_worried.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (angry > 50 && angry > upbeat&&angry > worried&&angry > depressed) {
			//male angry
			currentFace = 6;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("m_angry.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
		else if (depressed > 50 && depressed > upbeat&&depressed > worried&&depressed > angry) {
			//male depressed
			currentFace = 8;
			nextLetterTime = ofGetElapsedTimeMillis();
			lineCount = 0;
			letterCount = 0;
			ofBuffer buffer = ofBufferFromFile("m_depressed.txt");
			if (buffer.size()) {
				for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
					string line = *it;
					if (line.empty() == false) {
						seussLines.push_back(line);
					}
					cout << line << endl;
				}
			}
		}
	}

	loadFace(faces.getPath(currentFace));
}

void ofApp::analyzeTwitter() {
	ofHttpResponse resp = ofLoadURL("http://analyzewords.com/?handle=" + username);
	content = resp.data;

	ofStringReplace(content, "\n", "");
	ofStringReplace(content, "\r", "");
	ofStringReplace(content, "\"", "");
	ofStringReplace(content, " ", "");
	ofStringReplace(content, username, "");
	ofStringReplace(content, "http://analyzewords.com/?handle=", "");
	ofStringReplace(content, "http://analyzewords.com/index.php?handle=", "");
	ofStringReplace(content, "	", "");
	ofStringReplace(content, "\\", "");
	ofStringReplace(content, ".", "");
	ofStringReplace(content, "&", "");
	ofStringReplace(content, "http://", "");
	ofStringReplace(content, "'", "");
	ofStringReplace(content, "#", "");
	ofStringReplace(content, ":", "");
	string delimiter1 = "Analysisoftweetsfrom";
	string delimiter2 = "Tweetsanalyzedfrom";
	content = content.substr(content.find(delimiter1), content.find(delimiter2));
	content = content.substr(0, content.find(delimiter2));

	string del;
	string num;

	//upbeat
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	upbeat = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//worried
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	worried = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//angry
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	angry = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//depressed
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	depressed = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//pluggedIn
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	pluggedIn = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//personable
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	personable = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//arrogant
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	arrogant = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//spacy
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	spacy = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//analytic
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	analytic = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//sensory
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	sensory = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());
	//inTheMoment
	del = content.substr(0, content.find("align=rightnowrap>nbsp;"));
	ofStringReplace(content, del, "");
	content = content.substr(23, content.length());
	num = content.substr(0, content.find_first_of("nbsp"));
	inTheMoment = ofToInt(num);
	cout << num << endl;
	content = content.substr(num.length(), content.length() - num.length());

	emotionNum[1] = upbeat;		emotionName[1] = "upbeat";
	emotionNum[2] = worried;	emotionName[2] = "Worried";
	emotionNum[3] = angry;		emotionName[3] = "Angry";
	emotionNum[4] = depressed;  emotionName[4] = "Depressed";
	emotionNum[5] = pluggedIn;  emotionName[5] = "Plugged In";
	emotionNum[6] = personable; emotionName[6] = "Personable";
	emotionNum[7] = arrogant;   emotionName[7] = "Arrogant";
	emotionNum[8] = spacy;		emotionName[8] = "Spacy";
	emotionNum[9] = analytic;   emotionName[9] = "Analytic";
	emotionNum[10] = sensory;	emotionName[10] = "Sensory";
	emotionNum[11] = inTheMoment; emotionName[11] = "In the Moment";

	mytextFile.open("analyzewords.txt", ofFile::WriteOnly);
	mytextFile << content;
	mytextFile.close();
}

//--------------------------------------------------------------
void ofApp::exit() {
	kinect.close();

#ifdef USE_TWO_KINECTS
	kinect2.close();
#endif
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

		// example of how to create a keyboard event
		if (stage == 1) {
			if (key != OF_KEY_RETURN) {
				// some trickery: ignore the backspace key
				if (key != OF_KEY_BACKSPACE && key != OF_KEY_SHIFT) {
					username += key;
				}
				else {
					if (username.size() > 0) {
						username.erase(username.end() - 1);
					}
				}
			}
			else if (key == OF_KEY_RETURN) {
				analyzeTwitter();
				stage++;
				subString = "Analyzing your personality...";
			}
		}

		if (stage == 4) {
			/*
			switch (key) {
			case OF_KEY_UP:
				currentFace++;
				break;
			case OF_KEY_DOWN:
				currentFace--;
				break;
			}
			currentFace = ofClamp(currentFace, 0, faces.size());
			if (faces.size() != 0) {
				loadFace(faces.getPath(currentFace));
			}
			*/
			if (key == OF_KEY_RETURN) {
				bKinectOpen = true;
				bDrawPointCloud = true;
				//AnalyzeTwitter
				usernameFrame.loadImage("usernameFrame.png");
				font.load("20th Centenary Faux.ttf", 48);
				subFont.load("814yzx.ttf", 24);
				username = "";
				subString = "Please enter your twitter account here.";
				femalIndex = 0;
				maleIndex = 0;
				contentFont.load("20th Centenary Faux.ttf", 24);

				//TrackFace
				maxCur.width = -1;
				stage = -1;
				numPart = 0;

				yPos = 150;
				seussLines.clear();
				
				femalIndex = 0;
				maleIndex = 0;
				 upbeat = 0;
				 worried = 0;
				 angry=0;
				 depressed=0;
				 pluggedIn=0;
				 personable=0;
				 arrogant=0;
				 spacy=0;
				 analytic=0;
				 sensory=0;
				 inTheMoment=0;
			}
		}
	
	cout << username << endl;
	ofSaveScreen(ofToDataPath(ofToString(ofGetUnixTime()) + ".jpg"));
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button)
{}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button)
{
	cout << mouseY << endl;
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button)
{
	//cout << ofGetWindowPositionX() << endl;
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h)
{}


void ofApp::dragEvent(ofDragInfo dragInfo) {
	loadFace(dragInfo.files[0]);
}
