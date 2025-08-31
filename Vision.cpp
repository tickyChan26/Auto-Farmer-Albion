// файл Vision.cpp (классы реализованы в h)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)


#include <windows.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>

#include "Vision.h"


#include <opencv2/opencv.hpp>

#include <baseapi.h>
#include <leptonica/allheaders.h>    

extern string Path_directory; //Глобальный путь в директорию

using namespace cv;
using namespace std;


// Конструктор
Vision::Vision() 
{
    namedWindow("Bot Vision", WINDOW_NORMAL); // Создаём окно для отладки
}
// Деструктор
Vision::~Vision() {
    destroyWindow("Bot Vision"); // Закрываем окно
}

Mat Vision::CaptureScreen(int resol)
{
    // Находим окно игры
    HWND hWnd = FindWindow(NULL, L"Albion Online Client");
    if (!hWnd)
    {
        cerr << "Окно 'Albion Online Client' не найдено. Захватываем весь экран.\n";
        hWnd = GetDesktopWindow();
    }

    // Получаем размеры   области окна
    int screenWidth, screenHeight;
    if(resol == 1)
    {
        screenWidth = 1920;
        screenHeight = 1080;
    }
    else if(resol == 2)
    {
    
        screenWidth = 1366;
        screenHeight = 768;
    }

    HDC hScreenDC = GetDC(hWnd);
    HDC hMemoryDC = CreateCompatibleDC(hScreenDC);
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, screenWidth, screenHeight);
    HGDIOBJ hOldBitmap = SelectObject(hMemoryDC, hBitmap);

    // Используем PrintWindow вместо BitBlt
    if (!PrintWindow(hWnd, hMemoryDC, PW_RENDERFULLCONTENT))
    {
        cerr << "Ошибка в PrintWindow: " << GetLastError() << "\n";
        SelectObject(hMemoryDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(hWnd, hScreenDC);
        return Mat();
    }

    // Настраиваем BITMAPINFOHEADER
    BITMAPINFOHEADER bi = { 0 };
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = screenWidth;
    bi.biHeight = -screenHeight;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;

    Mat screen(screenHeight, screenWidth, CV_8UC4);
    if (!GetDIBits(hMemoryDC, hBitmap, 0, screenHeight, screen.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS))
    {
        cerr << "Ошибка в GetDIBits: " << GetLastError() << "\n";
        SelectObject(hMemoryDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(hWnd, hScreenDC);
        return Mat();
    }

    // Конвертируем из BGRA в BGR
    Mat bgrScreen;
    cvtColor(screen, bgrScreen, COLOR_BGRA2BGR);

    // Очистка
    SelectObject(hMemoryDC, hOldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hMemoryDC);
    ReleaseDC(hWnd, hScreenDC);

   

    return bgrScreen;
}


string Vision::RecognizeText(const Mat& image)
{
    tesseract::TessBaseAPI tess;
    if (tess.Init(NULL, "rus")) {
        cerr << "Не удалось инициализировать Tesseract\n";
        return "";
    }

    Mat gray, thresh;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);

    tess.SetImage(thresh.data, thresh.cols, thresh.rows, 1, thresh.step);
    string text = tess.GetUTF8Text();
    tess.End();

    return text;
}


void Vision::Show_Window(const Mat& screen)
{
    if (screen.empty()) {
        cerr << "Ошибка: пустое изображение для отображения\n";
        return;
    }

    const int displayWidth = 1200;
    const int displayHeight = 800;

    double aspectRatio = static_cast<double>(screen.cols) / screen.rows;
    int newWidth = displayWidth;
    int newHeight = static_cast<int>(newWidth / aspectRatio);

    if (newHeight > displayHeight) {
        newHeight = displayHeight;
        newWidth = static_cast<int>(newHeight * aspectRatio);
    }

    Mat display;
    resize(screen, display, Size(newWidth, newHeight), 0, 0, INTER_AREA);

    resizeWindow("Bot Vision", newWidth, newHeight);
    imshow("Bot Vision", display);
    waitKey(5);
}
