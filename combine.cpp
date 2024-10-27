#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include<vector>
#include<raylib.h>

using namespace std;
using namespace cv;

const int screenWidth = 1200, screenHeight = 900;
const int squareWeight = (int)(screenWidth / 13.2), squareHeight = (int)(screenHeight / 9.9);
Color color[6][9] = { 0 };

struct ColorRange {      Scalar lower, upper, color;    };

vector<ColorRange> colorRanges = {
    { {0, 182, 121, 255}, {216, 151, 60, 255}, {216, 151, 60, 255} },  // Yellow
    { {153, 0, 0, 255}, {217, 4, 41, 255}, {217, 4, 41, 255} },        // Red
    { {0, 0, 82, 255}, {242, 76, 0, 255}, {242, 76, 0, 255} },         // Orange
    { {0, 0, 163, 255}, {5, 130, 202, 255}, {5, 130, 202, 255} },      // Blue
    { {106, 0, 0, 255}, {63, 125, 32, 255}, {63, 125, 32, 255} },      // Green
    { {0, 88, 0, 255}, {235, 242, 250, 255}, {235, 242, 250, 255} }    // White
};

void setColorsRange(vector<Scalar>& scalars, const vector<ColorRange>& colorRanges) {
    for (auto& s : scalars) {
        for (const auto& range : colorRanges) {
            if (s[0] >= range.lower[0] && s[0] <= range.upper[0] &&
                s[1] >= range.lower[1] && s[1] <= range.upper[1] &&
                s[2] >= range.lower[2] && s[2] <= range.upper[2]) {
                s = range.color;
                break;
            }
        }
    }
}

int index = 0;

string s[6] = { "Show Front Face","Show Left Face With Respect to Front Face","Show Back Face With Respect to Front Face","Show Right Face With Respect to Front Face","Show Top Face With Respect to Front Face","Show Buttom Face With Respect to Front Face" };

Mat img, temp;
Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));

class cube {
    Color color[6][9];
    vector<Scalar> colors;
public:
    cube(Color* c)
    {
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 9; j++)
                color[i][j] = c[i];
        }
    }
    void changeColor(vector<Scalar>colors, int k)
    {
        setColorsRange(colors,colorRanges);
        for (int i = 0; i < 9; i++)
        {
            color[k][i].r = static_cast<int>(colors[i][2]);
            color[k][i].g = static_cast<int>(colors[i][1]);
            color[k][i].b = static_cast<int>(colors[i][0]);
            color[k][i].a = 255;
        }
    }
    void drawCubeFace(double x, double y, Color* c)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                DrawRectangle(x + j * squareWeight, y + i * squareHeight, squareWeight, squareHeight, c[3 * i + j]);
                DrawRectangleLines(x + j * squareWeight, y + i * squareHeight, squareWeight, squareHeight, Color{ 1, 8, 20, 255 });

            }
        }
    }
    Color* getColor(int k)
    {
        return color[k];
    }
};

void doIt(cube& cube, vector<Scalar>colors)
{
    cube.changeColor(colors, index);
    index++;
}

Mat imgProcess(const Mat& img)
{
    Mat gray, lab, claheImg, blueChannel;

    // Split the image into its Blue, Green, Red channels
    vector<Mat> bgr_planes;
    split(img, bgr_planes);
    blueChannel = bgr_planes[0]; // Isolate the Blue channel

    // Apply CLAHE to the Blue channel
    Ptr<CLAHE> clahe = createCLAHE(2.5, Size(8, 8));
    clahe->apply(blueChannel, blueChannel);

    // Merge the enhanced Blue channel back with the others
    bgr_planes[0] = blueChannel;
    merge(bgr_planes, gray);

    // Convert to LAB color space and enhance contrast further
    cvtColor(gray, lab, COLOR_BGR2Lab);
    vector<Mat> lab_planes(3);
    split(lab, lab_planes);

    // Apply CLAHE on the 'L' and 'b' channels to enhance brightness and blue-yellow differences
    clahe->apply(lab_planes[0], lab_planes[0]); // L channel (Lightness)
    clahe->apply(lab_planes[2], lab_planes[2]); // b channel (Blue-Yellow)
    merge(lab_planes, lab);

    cvtColor(lab, gray, COLOR_Lab2BGR);
    cvtColor(gray, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, gray, Size(3, 3), 3, 3);
    // Apply Median Blur
    medianBlur(gray, gray, 5);

    // Adjust Canny thresholds
    Canny(gray, gray, 5, 150);  // Increased thresholds for better edge detection

    // Morphological operations to strengthen edges
    dilate(gray, gray, kernel);
    erode(gray, gray, kernel);

    return gray;
}

void contoursDetect(Mat& originalImg, const Mat& editImg, cube& cube)
{
    int count = 0,num;
    vector<vector<Point>> contours;
    vector<vector<Point>> contours1;
    vector<Vec4i> hierarchy;
    int centerX, centerY;
    Mat uneditimage = originalImg.clone();
    vector<Scalar> colors = { 0 };
    findContours(editImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours)
    {
        double area = contourArea(contour);
        if (area <= 1000 || area >= 10000) continue;  // Skip contours not in desired area range

        float perimeter = arcLength(contour, true);
        double circularity1 = perimeter * perimeter / 16.0 - area;
        double circularity2 = perimeter * perimeter / 12.566 - area;

        if (circularity1 >= 100 && circularity2 >= 100) continue;  // Skip if circularity conditions are not met

        vector<Point> conPoly;
        approxPolyDP(contour, conPoly, 0.02 * perimeter, true);

        // If the polygon does not have 4, 6, or 8 vertices, skip it
        if (conPoly.size() != 4 && conPoly.size() != 6 && conPoly.size() != 8) continue;

        Rect boundingBox = boundingRect(conPoly);
        double aspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;

        // Skip if the aspect ratio is not within the desired range
        if (aspectRatio < 0.8 || aspectRatio > 1.2) continue;
        count++;
        // If all conditions are met, add to contours1
        contours1.push_back(contour);
    }
    drawContours(originalImg, contours1, -1, Scalar(50, 50, 250), 4);
    putText(originalImg, "Colours : " + to_string(contours1.size()), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(222, 222, 22), 4);
    if (count == 9)     {
        
            putText(img, "Press Enter if all colors are selected", Point(0, 60), FONT_HERSHEY_COMPLEX, 0.9, Scalar(195, 48, 90), 2);
            if (waitKey(0) == 13) {
                for (int i = 0; i < count; i++) {
                    // Create a mask where the current contour is white and everything else is black
                    Mat mask = Mat::zeros(uneditimage.size(), CV_8UC1);
                    drawContours(mask, contours1, i, Scalar(255), FILLED);

                    // Calculate the mean color inside the contour using the mask
                    Scalar meanColor = mean(uneditimage, mask);
                    cout << endl << meanColor[0] << "\t" << meanColor[1] << "\t" << meanColor[2] << endl;
                    waitKey(1);
                    colors.push_back(meanColor);
                    circle(img,Point(20, 40 * i), 25, meanColor,FILLED);
                }
                doIt(cube, colors);
            }
    }
}

int main()
{
    vector<Scalar>colors;
    Color c[6] = {
            {217, 4, 41, 255},    // Dark red
            { 235, 242, 250, 255 },  // white
            { 216, 151, 60, 255 },   // yellow
            {242, 76, 0, 255},      // orange
            {5, 130, 202, 255},    // Blue
            {63, 125, 32, 255}    // Green
    };
    cube cube(c);
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Error opening Camera" << endl;
        return -1;
    }
    InitWindow(screenWidth, screenHeight, "Cube");
    SetTargetFPS(60);
    Color backgroundColor = { 131, 144, 220,255 };
    while (!WindowShouldClose())
    {
        // Read from the camera
        cap.read(img);
        if (img.empty())
        {
            cerr << "Error capturing frame" << endl;
            break;
        }

        // Process the image and display it
        temp = imgProcess(img);
        contoursDetect(img, temp, cube);
        putText(img, s[index], Point(5, 90), FONT_HERSHEY_DUPLEX, 0.8, Scalar(195, 148, 190), 2);
        imshow("Hi", img);
        // Begin Raylib drawing
        BeginDrawing();
        ClearBackground(backgroundColor);

        for (int i = 0; i < 4; i++)
        {
            cube.drawCubeFace(25 * (i + 1) + i * screenWidth / 4.5, screenHeight / 2.9, cube.getColor(i));
        }
        cube.drawCubeFace(screenWidth / 4.5 + 50, screenHeight / 3 - screenHeight / 3.3 - 10, cube.getColor(4));
        cube.drawCubeFace(screenWidth / 4.5 + 50, screenHeight / 3 + screenHeight / 3.3 + 30, cube.getColor(5));

        EndDrawing();
        if (index >= 6)    break;
    }
    while (1)
    {

    }
    CloseWindow();
    
    return 0;
}
