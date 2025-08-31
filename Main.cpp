// файл Main
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

#include <opencv2/opencv.hpp>

#include "Vision.h"

using namespace std;
string Path_directory; //Глобальный путь в директорию
int DEBUG = 0; int savescreen = 0; int resol = 0;

//Вспомогательные функции
int RandomInt(int min, int max)
{
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);
    return dis(gen);
}
std::wstring utf8_to_wstring(const std::string& str)
{
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(size_needed - 1, 0); // -1, чтобы не включать null-terminator
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size_needed);
    return wstr;
}
std::string wstring_to_cp1251(const std::wstring& wstr) {
    int size_needed = WideCharToMultiByte(1251, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string result(size_needed, 0);
    WideCharToMultiByte(1251, 0, wstr.c_str(), -1, &result[0], size_needed, nullptr, nullptr);
    return result;
}
std::wstring to_lower_wstring(const std::wstring& input) 
{
    std::wstring result = input;
    std::transform(result.begin(), result.end(), result.begin(), towlower); // работает с русскими
    return result;
}
void remove_garbage_chars(std::wstring& str) 
{
    // Функция удаления нежелательных символов из std::wstring

    const std::wstring garbage =
        L"\\'\"@`““«»®™‘‚*&^$#-–—“”°*=.:’_:_+|{}[]<>%ф0123456789\n\r\t"
        L",!?;()"
        L"©™§¶¢‰∞†‡"
        L"‚„‹›”‘’′″〝〞"
        L"═║╚╝╔╗╩╦╠╣╬"
        L"♪♫€£¥±÷×"
        L"…‒—–−￼�"
        L"√∫∂∑∏&"
        L"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        L"abcdefghijklmnopqrstuvwxyz";

    str.erase(std::remove_if(str.begin(), str.end(),
        [&garbage](wchar_t c) 
        {
            return garbage.find(c) != std::wstring::npos;
        }),
        str.end());
}

using Clock = std::chrono::steady_clock;
Clock::time_point Cl_Resou = Clock::now();
Clock::time_point last_move_time = Clock::now();
Clock::time_point wait_bar = Clock::now();
Clock::time_point wait_targ = Clock::now();
Clock::time_point clock_fish = Clock::now(); 
Clock::time_point clock_bars = Clock::now();
Clock::time_point Clock_check = Clock::now();
Clock::time_point clock_trhow = Clock::now();
Clock::time_point clock_wati = Clock::now();
Clock::time_point clock_weatbefore = Clock::now();
Clock::time_point clock_minute = Clock::now();



cv::Mat visual;
void SaveDebugScreenshot(const cv::Mat& bgrScreen, string name)
{
    if (savescreen == 1)
    {
        if (bgrScreen.empty()) return;

        // Создаём папку, если её нет
        string folderPath = Path_directory + "D_screenshot\\";
        // Формируем имя файла
        string savePath = folderPath + name + ".png";

        // Пытаемся сохранить
        if (cv::imwrite(savePath, bgrScreen)) {}
        else {}


    }
}

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////   Продвинутый бот  /////////////////////////////////////
// Структура для узла A* алгоритма
struct PathNode 
{
    int x, y;                    // Координаты на мини-карте
    double g, h, f;              // Стоимости для A*
    PathNode* parent;            // Родительский узел

    PathNode(int x = 0, int y = 0) : x(x), y(y), g(0), h(0), f(0), parent(nullptr) {}
};

// Глобальные переменные 
cv::Mat minimap_mask_global;     // Маска проходимости мини-карты
double resolution_X; double resolution_Y; //разрешение экрана

cv::Rect minimap_roi(1560, 740, 310, 240); //вырезка миникарты на всём экране
double squaretomap_center_x; double squaretomap_center_y;   // координаты центра мини карты (где находится игрок)

double player_center_x; double player_center_y; // координаты игрока в центре реального экрана
double scale_worldtomap = 15.0; // 1 пиксель на миникарте = Х пикселей в игровом мире (работает как множитель)

// Коэффициент масштабирования Windows
float scaleFactor; //пример: 125% масштабирование виндовс
//(запись) как правильно расчитывать центр целей: SmoothMove(player_center_x / scaleFactor, player_center_y / scaleFactor); // делим на скейл фактор  если скейл 100 то надо *



bool is_moving = false;               // Персонаж движется
std::vector<cv::Point> current_path;  // Текущий путь
size_t current_path_index = 1;        // Индекс следующей точки

cv::Point currentTarget = { -1, -1 };
bool hasTarget;

// функции для управления
void SmoothMove(int targetX, int targetY)
{
    int steps = RandomInt(17, 25); 
    int maxDelayMs = RandomInt(9, 11); 

    POINT start;
    if (!GetCursorPos(&start)) {
        cerr << "Ошибка: Не удалось получить текущую позицию мышки\n";
        return;
    }

    // Проверяем валидность целевых координат
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    if (targetX <= 0 || targetY <= 0 || targetX >= screenWidth || targetY >= screenHeight) {
        cerr << "Ошибка: Некорректные целевые координаты (" << targetX << ", " << targetY << ")\n";
        return;
    }

    double dx = targetX - start.x;
    double dy = targetY - start.y;
    double distance = sqrt(dx * dx + dy * dy);

    if (distance < 4)
    {
        cerr << "мышка уже на месте\n";
        return; // Если уже на месте, ничего не делаем
    }

    double dirX = dx / distance;
    double dirY = dy / distance;
    double normalX = -dirY;
    double normalY = dirX;

    double arcHeight = distance * 0.02; // Уменьшенная высота дуги
    double arcDirection = 1.0;

    for (int i = 1; i <= steps; ++i) {
        double t = i / static_cast<double>(steps);
        double smoothT = t * t * (3 - 2 * t);

        double x = start.x + dx * smoothT;
        double y = start.y + dy * smoothT;

        double arcOffset = sin(t * 3.14159265) * arcHeight * arcDirection;
        x += normalX * arcOffset;
        y += normalY * arcOffset;

        // Проверка на валидность промежуточных координат
        if (x <= 0 || y <= 0 || x >= screenWidth || y >= screenHeight) {
            cerr << "Ошибка: Промежуточные координаты вне экрана (" << x << ", " << y << ")\n";
            continue;
        }
        if (!SetCursorPos(static_cast<int>(x), static_cast<int>(y))) {
            cerr << "Ошибка: SetCursorPos не удалось установить координаты (" << x << ", " << y << ")\n";
            continue;
        }


        // Прерывистая задержка: в начале/конце чуть дольше
        double edgeBias = 1.0 - abs(0.5 - t) * 2; // 0 в центре, 1 на краях
        int delay = static_cast<int>(maxDelayMs * edgeBias + RandomInt(0, 3));
        this_thread::sleep_for(chrono::milliseconds(max(1, delay)));
        // Имитация мышечного замедления — редкие микро-паузы
        if (RandomInt(0, 100) < 2 && i > steps / 2)
        {
            this_thread::sleep_for(chrono::milliseconds(RandomInt(20, 50)));
        }
    }
}
void LeftClick()
{
    // для добычи
    if (DEBUG == 1) { cout << ">>>> LeftClick\n"; }
    this_thread::sleep_for(chrono::milliseconds(100));
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);

    this_thread::sleep_for(chrono::milliseconds(50));
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
}
void RightClick()
{
    // для движения
    if (DEBUG == 1) { cout << ">>>> RightClick\n"; }
    this_thread::sleep_for(chrono::milliseconds(100));
    mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
    this_thread::sleep_for(chrono::milliseconds(50));
    mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);
}
void HoldMouseLeft(int duration_ms)
{
    // Нажать ЛКМ
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    cout << "Держим мышку  "<< duration_ms <<"  миллисек\n";
    // Подержать
    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

    // Отпустить ЛКМ
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
}
void PressKey(WORD keyCode, int delayMs)
{
    INPUT input[2] = {};

    // Нажатие
    input[0].type = INPUT_KEYBOARD;
    input[0].ki.wVk = keyCode; // виртуальный код клавиши

    // Отжатие
    input[1].type = INPUT_KEYBOARD;
    input[1].ki.wVk = keyCode;
    input[1].ki.dwFlags = KEYEVENTF_KEYUP;

    // Отправляем
    SendInput(2, input, sizeof(INPUT));

    Sleep(delayMs); // задержка
}



// Умное перемещение мыши на точку с миникарты преобразованную в глобальную игру
void Smart_SmoothMove(const cv::Point2d& minimap_point, int windowX, int windowY)
{
    // Максимальное смещение в игровых координатах (пикселях экрана от центра)
    const double max_offset_game = 200.0; // 200 пикселей от центра — безопасно для 1920x1080

    // Смещение от центра миникарты
    double dx_map = minimap_point.x - squaretomap_center_x;
    double dy_map = minimap_point.y - squaretomap_center_y;

    // Перевод в игровые координаты
    double dx_game = dx_map * scale_worldtomap;
    double dy_game = dy_map * scale_worldtomap;

    // Ограничение по длине вектора смещения в игровых координатах
    double dist_game = std::sqrt(dx_game * dx_game + dy_game * dy_game);
    if (dist_game > max_offset_game) {
        double scale = max_offset_game / dist_game;
        dx_game *= scale;
        dy_game *= scale;
    }

    int target_x = static_cast<int>(player_center_x + dx_game);
    int target_y = static_cast<int>(player_center_y + dy_game);
    int screen_target_x, screen_target_y;

    // УЧЁТ МАСШТАБИРОВАНИЯ
    if (scaleFactor == 1.00f) // умножаем потому что scaleFactor == 1
    {
        screen_target_x = static_cast<int>(target_x * scaleFactor);
        screen_target_y = static_cast<int>(target_y * scaleFactor);
    }
    else
    {
        screen_target_x = static_cast<int>(target_x / scaleFactor);
        screen_target_y = static_cast<int>(target_y / scaleFactor);
    }


   

    // Перемещаем курсор
    SmoothMove(screen_target_x, screen_target_y);

}


//////////// просчёты движения ////////////
// Функция получения эвристического расстояния (Manhattan distance)
double GetHeuristic(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}
// Проверка, проходима ли точка на мини-карте
bool IsPassable(int x, int y, const cv::Mat& mask) {
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) return false;

    // Белый цвет (255) = проходимо, черный (0) = непроходимо
    return mask.at<uchar>(y, x) > 200;
}
// Получение соседних узлов (8-направленное движение)
std::vector<cv::Point> GetNeighbors(int x, int y) {
    std::vector<cv::Point> neighbors;

    // 8 направлений движения
    int dx[] = { -1, -1, -1,  0,  0,  1,  1,  1 };
    int dy[] = { -1,  0,  1, -1,  1, -1,  0,  1 };

    for (int i = 0; i < 8; i++) {
        neighbors.push_back(cv::Point(x + dx[i], y + dy[i]));
    }

    return neighbors;
}
// A* алгоритм поиска пути на мини-карте
std::vector<cv::Point> FindPath(cv::Point start, cv::Point goal, const cv::Mat& passability_mask) {
    
    std::vector<cv::Point> path;
    if (!IsPassable(start.x, start.y, passability_mask) ||
        !IsPassable(goal.x, goal.y, passability_mask)) {
        std::cout << "Стартовая или целевая точка непроходима!\n";
        return path;
    }

    // Списки открытых и закрытых узлов
    std::vector<PathNode*> openList;
    std::vector<PathNode*> closedList;
    std::vector<std::vector<PathNode*>> allNodes(passability_mask.cols,
        std::vector<PathNode*>(passability_mask.rows, nullptr));

    // Создаем стартовый узел
    PathNode* startNode = new PathNode(start.x, start.y);
    startNode->g = 0;
    startNode->h = GetHeuristic(start.x, start.y, goal.x, goal.y);
    startNode->f = startNode->g + startNode->h;

    openList.push_back(startNode);
    allNodes[start.x][start.y] = startNode;

    while (!openList.empty()) {
        // Находим узел с наименьшей стоимостью f
        auto current = std::min_element(openList.begin(), openList.end(),
            [](PathNode* a, PathNode* b) { return a->f < b->f; });

        PathNode* currentNode = *current;
        openList.erase(current);
        closedList.push_back(currentNode);

        // Достигли цели
        if (currentNode->x == goal.x && currentNode->y == goal.y) {
            // Восстанавливаем путь
            PathNode* node = currentNode;
            while (node != nullptr) {
                path.insert(path.begin(), cv::Point(node->x, node->y));
                node = node->parent;
            }
            break;
        }

        // Проверяем всех соседей
        auto neighbors = GetNeighbors(currentNode->x, currentNode->y);
        for (const auto& neighbor : neighbors) {
            // Проверяем проходимость
            if (!IsPassable(neighbor.x, neighbor.y, passability_mask)) continue;

            // Проверяем, не в закрытом ли списке
            bool inClosedList = false;
            for (const auto& closed : closedList) {
                if (closed->x == neighbor.x && closed->y == neighbor.y) {
                    inClosedList = true;
                    break;
                }
            }
            if (inClosedList) continue;

            // Стоимость движения (диагональное движение дороже)
            double moveCost = (abs(neighbor.x - currentNode->x) + abs(neighbor.y - currentNode->y)) == 2 ? 1.414 : 1.0;
            double tentativeG = currentNode->g + moveCost;

            PathNode* neighborNode = allNodes[neighbor.x][neighbor.y];

            if (neighborNode == nullptr) {
                // Создаем новый узел
                neighborNode = new PathNode(neighbor.x, neighbor.y);
                neighborNode->g = tentativeG;
                neighborNode->h = GetHeuristic(neighbor.x, neighbor.y, goal.x, goal.y);
                neighborNode->f = neighborNode->g + neighborNode->h;
                neighborNode->parent = currentNode;

                allNodes[neighbor.x][neighbor.y] = neighborNode;
                openList.push_back(neighborNode);
            }
            else if (tentativeG < neighborNode->g) {
                // Нашли лучший путь к этому узлу
                neighborNode->g = tentativeG;
                neighborNode->f = neighborNode->g + neighborNode->h;
                neighborNode->parent = currentNode;

                // Добавляем в открытый список, если его там нет
                bool inOpenList = std::find(openList.begin(), openList.end(), neighborNode) != openList.end();
                if (!inOpenList) {
                    openList.push_back(neighborNode);
                }
            }
        }
    }

    // Очистка памяти
    for (int x = 0; x < passability_mask.cols; x++) {
        for (int y = 0; y < passability_mask.rows; y++) {
            delete allNodes[x][y];
        }
    }

    if (!path.empty()) 
    {
        if (DEBUG == 1) { std::cout << "Найден путь из " << path.size() << " точек\n"; }
    }

    return path;
}
// Упрощение пути (убираем лишние точки)
void SimplifyPath(std::vector<cv::Point>& path) {
    if (path.size() <= 2) return;

    std::vector<cv::Point> simplified;
    simplified.push_back(path[0]);

    for (size_t i = 1; i < path.size() - 1; i++) {
        cv::Point prev = path[i - 1];
        cv::Point curr = path[i];
        cv::Point next = path[i + 1];

        // Проверяем, лежат ли три точки на одной прямой
        int cross = (curr.x - prev.x) * (next.y - prev.y) - (curr.y - prev.y) * (next.x - prev.x);

        if (abs(cross) > 0) { // Не на одной прямой
            simplified.push_back(curr);
        }
    }

    simplified.push_back(path.back());
    path = simplified;


    if (DEBUG == 1) { std::cout << "Путь упрощен до " << path.size() << " точек\n"; }
}
// добавляем промежуточные точки для того что бы мышка при движении не выходила за экран
std::vector<cv::Point> EnforceMaxDistance(const std::vector<cv::Point>& path, double maxDistance)
{
    if (path.empty()) return {};

    std::vector<cv::Point> result;
    result.push_back(path[0]);

    for (size_t i = 1; i < path.size(); ++i)
    {
        cv::Point prev = result.back();
        cv::Point next = path[i];
        double dist = cv::norm(next - prev);

        if (dist <= maxDistance)
        {
            result.push_back(next);
        }
        else
        {
            // Вставляем промежуточные точки
            int steps = static_cast<int>(std::ceil(dist / maxDistance));
            for (int s = 1; s <= steps; ++s)
            {
                double alpha = static_cast<double>(s) / steps;
                cv::Point interpolated = prev + (next - prev) * alpha;
                result.push_back(interpolated);
            }
        }
    }

    if (DEBUG == 1) { std::cout << "Путь оптимизирован до " << result.size() << " точек\n"; }

    return result;
}

//////////////////////////////////////////

// функция посика блуждания по пути в обход непроходимой месности
cv::Point goal; int attempts = 0; 
void Random_wandering_in_search(cv::Mat& screen, cv::Mat& visual, int windowX, int windowY)
{
    // Центр мини-карты (игрок)
    cv::Point start(squaretomap_center_x, squaretomap_center_y);


    // Поиск случайной проходимой цели
    do
    {
        int radius = 40 + rand() % 40; // Радиус от 40 до 80
        double angle = (rand() % 360) * CV_PI / 180.0;
        int dx = static_cast<int>(radius * cos(angle));
        int dy = static_cast<int>(radius * sin(angle));

        goal = start + cv::Point(dx, dy);
        attempts++;
        if (attempts > 100)
        {
            std::cout << "Не удалось найти проходимую точку!\n";
            return;
        }
    } while (!IsPassable(goal.x, goal.y, minimap_mask_global));


    // Поиск пути
    auto path = FindPath(start, goal, minimap_mask_global);


    if (!path.empty())
    {

        SimplifyPath(path); // Упрощаем путь 
        path = EnforceMaxDistance(path, 15.0); // Шаги не более Х пикселей

        // === Выполняем плавное перемещение к следующей точке ===
        current_path = path;
        current_path_index = 1;
        is_moving = true;
    }

}


/// /// /// нахождение прямоугольника с текстом на ресурсе
cv::Mat Color_Mask_Text(const cv::Mat& image)
{
    // image входит чисто в ч\б
    

    // alpha — контраст (1.0 — без изменений), beta — яркость (0 — без изменений)
    double alpha = 1.5;  // >1 усиливает контраст   
    int beta = -25;      // <0 затемняет
    image.convertTo(image, -1, alpha, beta);


    // Если значение пикселя > X, он станет B (255 белым)
    // Если значение пикселя ≤ X, он станет 0 (чёрным)
    int X = 230,   B = 255;

    cv::threshold(image, image, X, B, cv::THRESH_BINARY);
    SaveDebugScreenshot(image, "equalize____threshold_");

    return image;
}

/// /// /// прогрес бар сбора.  // обработка добыается ли лесурс.
vector<cv::Point> Progress_zone;
cv::Mat progress_bar_mask(const cv::Mat& BGR_image)
{
    cv::Mat mask_g;

    if (BGR_image.channels() != 3 || BGR_image.type() != CV_8UC3) 
    {
        std::cerr << "Ошибка: Изображение должно быть 3-канальным BGR (CV_8UC3)\n";
        return cv::Mat();
    }

    // Основной диапазон зелёной полосы B-G-R
    cv::inRange(BGR_image, cv::Scalar(0, 75, 0), cv::Scalar(12, 255, 20), mask_g); // тёмная зелень

    return mask_g;
   
}
bool Detect_progress_bar_collection(const cv::Mat& screen, cv::Mat& visual)
{
    if (screen.empty())
    {
        std::cerr << "Пустое изображение для анализа синего цвета\n";
    }

    Progress_zone.clear(); // Очищаем старые центры
    cv::Mat cut_scr = screen.clone();

    // предварительно обрезаем зону где будет находится прогресс сбора ресурса.
    cv::Rect roi(853, 674, 247, 16);
    cut_scr = cut_scr(roi);

    cv::Mat bar_masked = progress_bar_mask(cut_scr);


    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bar_masked, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    bool found = false; 
    for (const auto& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);

        if (rect.area() > 10)
        {
            found = true;
            // Переводим координаты из cut_scr в глобальные
            cv::Point global_center = rect.tl() + roi.tl() + cv::Point(rect.width / 2, rect.height / 2);
            Progress_zone.push_back(global_center);

            // Визуализация
            cv::rectangle(visual, rect + roi.tl(), cv::Scalar(200, 0, 25), 2);

        }
    }


    return found;
}

/// /// /// нахождение Т4 ресурса пенька и обвод в рамки
vector<cv::Point> detectedCenters; 
double IsNight(const cv::Mat& image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return cv::mean(gray)[0];
}
cv::Mat Color_Mask_Penka(const cv::Mat& BGR_image)
{
    // BGR_image - изображения с игры (предварительно закрыт UI)
    // преобразование  BGR_image в BGR_inverted  (лучше видно нужный ресурс) (инверсия цвета)
    cv::Mat BGR_inverted; cv::Mat mask, mask_white, mask_combined;  int avgBrightness;
    cv::cvtColor(BGR_image, BGR_inverted, cv::COLOR_BGR2RGB);
    avgBrightness = IsNight(BGR_inverted);

    // Поиск бутонов
    cv::inRange(BGR_inverted, cv::Scalar(220, 210, 220), cv::Scalar(255, 255, 255), mask_white);
    cv::morphologyEx(mask_white, mask_white, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);

    // Поиск листьев
    if (avgBrightness >= 90)
    {
        cv::inRange(BGR_inverted, cv::Scalar(16, 50, 65), cv::Scalar(75, 115, 172), mask);

        // Объединение масок
        cv::bitwise_or(mask, mask_white, mask_combined);

        // [опционально] очистка шума
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
        SaveDebugScreenshot(mask, "result\\mask_Bright-" + to_string(avgBrightness) + "--");

    }
    else if (avgBrightness >= 70 && avgBrightness < 90)
    {
       
        cv::inRange(BGR_inverted, cv::Scalar(18, 40, 70), cv::Scalar(69, 110, 175), mask);

        // Объединение масок
        cv::bitwise_or(mask, mask_white, mask_combined);

        // [опционально] очистка шума
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
        SaveDebugScreenshot(mask, "result\\mask_Bright-" + to_string(avgBrightness) + "--");

    }
    else if (avgBrightness >= 60 && avgBrightness < 70)
    {
        cv::inRange(BGR_inverted, cv::Scalar(10, 42, 105), cv::Scalar(45, 76, 150), mask);

        // Объединение масок
        cv::bitwise_or(mask, mask_white, mask_combined);

        // [опционально] очистка шума
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
        SaveDebugScreenshot(mask, "result\\mask_Bright-" + to_string(avgBrightness) + "--");
    }
    else if (avgBrightness >= 50 && avgBrightness < 60)
    {
        cv::inRange(BGR_inverted, cv::Scalar(10, 42, 90), cv::Scalar(45, 76, 150), mask);

        // Объединение масок
        cv::bitwise_or(mask, mask_white, mask_combined);

        // [опционально] очистка шума
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
        SaveDebugScreenshot(mask, "result\\mask_Bright-" + to_string(avgBrightness) + "--");
    }
    else if (avgBrightness >= 0 && avgBrightness < 50)
    {
        cv::inRange(BGR_inverted, cv::Scalar(10, 42, 60), cv::Scalar(50, 80, 145), mask);

        // Объединение масок
        cv::bitwise_or(mask, mask_white, mask_combined);

        // [опционально] очистка шума
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
        SaveDebugScreenshot(mask, "result\\mask_Bright-" + to_string(avgBrightness) + "--");

    }

    cv::GaussianBlur(mask_combined, mask_combined, cv::Size(7, 7), 3);     // размываем
    cv::threshold(mask_combined, mask_combined, 50, 255, cv::THRESH_BINARY); // закрашиваем размытое

    SaveDebugScreenshot(mask_combined, "result\\FINAL_blur_threshold_");

    return mask_combined;
}
std::vector<cv::Point>DetectAndDrawBlueObjects(const cv::Mat& screen, cv::Mat& visual)
{
    if (screen.empty())
    {
        std::cerr << "Пустое изображение для анализа синего цвета\n";
        return screen;
    }

    detectedCenters.clear(); // Очищаем старые центры
    
    cv::Mat clear_UI = screen.clone(); // для создания маски
    // Закрашиваем ненужный UI, в чёрный.  на экране размером 1920 на 1080 стартовая точка слева вверху 0 на 0;
    {
        // Закрашиваем ненужный UI чёрным (BGR: 0, 0, 0)

        // 1. Левый верхний угол (Статус персонажа)
        cv::rectangle(clear_UI, cv::Rect(0, 0, 610, 185), cv::Scalar(0, 0, 0), cv::FILLED);

        // 2. Верхняя панель справа (настройки)
        cv::rectangle(clear_UI, cv::Rect(1300, 0, 620, 85), cv::Scalar(0, 0, 0), cv::FILLED);

        // 3. Узкая вертикальная панель справа (настройки)
        cv::rectangle(clear_UI, cv::Rect(1860, 0, 60, 260), cv::Scalar(0, 0, 0), cv::FILLED);

        // 4. Нижняя панель (способности)
        cv::rectangle(clear_UI, cv::Rect(490, 970, 1430, 110), cv::Scalar(0, 0, 0), cv::FILLED);

        // 5. Правая нижняя панель (Карта)
        cv::rectangle(clear_UI, cv::Rect(1510, 705, 410, 375), cv::Scalar(0, 0, 0), cv::FILLED);
    }

    // Маска цветов для Т4 ресурса (Пенька)
    cv::Mat mask = Color_Mask_Penka(clear_UI);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int count = 0; 
    for (const auto& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() > 50 && rect.width >= 55 && rect.height >= 55 && rect.width <= 125 && rect.height <= 115)
        {
            // Отсекаем слишком вытянутые (например 90x6, 6x90 и т.п.)
            double aspectRatio = static_cast<double>(rect.width) / rect.height;
            if (aspectRatio < 0.5 || aspectRatio > 2.0) continue;

            cv::rectangle(visual, rect, cv::Scalar(255, 0, 0), 2);

            cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
            detectedCenters.push_back(center);

        }
    }

    return detectedCenters;
}


/// /// /// нахождение полос здоровья враждебных существ.
vector<cv::Point> det_enem;
cv::Mat enemies_Mask(const cv::Mat& BGR_image)
{
    cv::Mat BGR_inverted;   cv::Mat finall, blue1, blue2;
    cv::cvtColor(BGR_image, BGR_inverted, cv::COLOR_BGR2RGB);
    

    // полоска хп состоит из 4-рёх цветов
    cv::inRange(BGR_inverted, cv::Scalar(190, 40, 0), cv::Scalar(255, 66, 33), blue1);
    // полоска хп состоит из 4-рёх цветов
    cv::inRange(BGR_inverted, cv::Scalar(93, 19, 0), cv::Scalar(120, 33, 18), blue2);



    // Объединение масок
    cv::bitwise_or(blue1, blue2, finall);

    SaveDebugScreenshot(BGR_inverted, "map\\enemies_BGR_inverted");

    SaveDebugScreenshot(finall, "map\\enemies_finish");
    return finall;

}
bool Detect_enemies_Objects(const cv::Mat& screen, cv::Mat& visual)
{
    det_enem.clear(); // Очищаем старые центры

    cv::Mat clear_UI = screen.clone(); // для создания маски
    // Закрашиваем ненужный UI, в чёрный.  на экране размером 1920 на 1080 стартовая точка слева вверху 0 на 0;
    {
        cv::rectangle(clear_UI, cv::Rect(0, 0, 610, 185), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::rectangle(clear_UI, cv::Rect(1300, 0, 620, 85), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::rectangle(clear_UI, cv::Rect(1860, 0, 60, 260), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::rectangle(clear_UI, cv::Rect(490, 970, 1430, 110), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::rectangle(clear_UI, cv::Rect(1510, 705, 410, 375), cv::Scalar(0, 0, 0), cv::FILLED);
    }

    // Маска врагов
    cv::Mat mask = enemies_Mask(clear_UI);

    // Поиск контуров
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    bool found = false;
    for (const auto& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);

        // Фильтруем по высоте и ширине
        if (rect.height <= 4 && rect.height >= 1 && rect.width <= 115 && rect.width >= 110)
        {
            found = true;
 
            cv::rectangle(visual, rect, cv::Scalar(150, 65, 0), 1); // Красный прямоугольник
            cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
            det_enem.push_back(center);
        }
    }

    return found;
}
void Run_out_enemy()
{
    if (det_enem.empty())
    {
        std::cout << "Врагов не найдено для ухода\n";
        return;
    }

    // Центр игрока на миникарте
    cv::Point2d player_map_pos(squaretomap_center_x, squaretomap_center_y);

    // Суммарный вектор опасности
    cv::Point2d danger_vector(0, 0);

    for (const auto& enemy_screen : det_enem)
    {

        // УЧЁТ МАСШТАБИРОВАНИЯ
        double dx_game, dy_game;
        if (scaleFactor == 1.00f)   // делаю вариацию на будущее если понадобится
        {
            dx_game = (enemy_screen.x * scaleFactor) - player_center_x;
            dy_game = (enemy_screen.y * scaleFactor) - player_center_y;

        }
        else
        {
            dx_game = (enemy_screen.x * scaleFactor) - player_center_x;
            dy_game = (enemy_screen.y * scaleFactor) - player_center_y;
        }


        // Преобразуем экранные координаты врага в координаты миникарты
        double dx_map = dx_game / scale_worldtomap;
        double dy_map = dy_game / scale_worldtomap;

        cv::Point2d enemy_map_pos = player_map_pos + cv::Point2d(dx_map, dy_map);

        // Вектор от врага к игроку (обратный к вектору убегания)
        cv::Point2d v = player_map_pos - enemy_map_pos;

        // Добавляем в суммарный вектор
        danger_vector += v;
    }

    double length = sqrt(danger_vector.x * danger_vector.x + danger_vector.y * danger_vector.y);

    cv::Point2d run_target_map;

    if (length > 1e-6)
    {
        // Нормируем вектор и умножаем на желаемое расстояние ухода
        double run_distance = 30.0;
        run_target_map = player_map_pos + (danger_vector * (run_distance / length));
    }
    else
    {
        // Враги вокруг или слишком близко — пробуем найти любое проходимое направление

        const int attempts = 16; // число направлений, например 16 (каждые 22.5 градуса)
        const double run_distance = 25.0;

        bool found = false;
        for (int i = 0; i < attempts; i++)
        {
            double angle = (2 * CV_PI / attempts) * i;
            cv::Point2d direction(cos(angle), sin(angle));
            cv::Point2d candidate = player_map_pos + direction * run_distance;

            int cx = static_cast<int>(candidate.x);
            int cy = static_cast<int>(candidate.y);

            if (IsPassable(cx, cy, minimap_mask_global))
            {
                run_target_map = candidate;
                found = true;
                break;
            }
        }

        if (!found)
        {
            std::cout << "Не удалось найти проходимую точку ухода, остаюсь на месте\n";
            return;
        }
    }

    // Проверяем проходимость целевой точки (дополнительно)
    int tx = static_cast<int>(run_target_map.x);
    int ty = static_cast<int>(run_target_map.y);
    if (!IsPassable(tx, ty, minimap_mask_global))
    {
        std::cout << "Целевая точка непроходима, остаюсь на месте\n";
        return;
    }

    // Старт и цель для поиска пути
    cv::Point start(static_cast<int>(player_map_pos.x), static_cast<int>(player_map_pos.y));
    cv::Point goal(static_cast<int>(run_target_map.x), static_cast<int>(run_target_map.y));

    auto path = FindPath(start, goal, minimap_mask_global);
    SimplifyPath(path);
    auto final_path = EnforceMaxDistance(path, 15.0);

    if (final_path.empty())
    {
        std::cout << "Путь ухода не найден\n";
        return;
    }

    current_path = final_path;
    current_path_index = 1;
    is_moving = true;
    std::cout << "Начинаю уход от врага\n";
}

/// /// /// нахождение непроходимых обьектов. (камней, зданий) на минни карте
int location = 1; 
cv::Mat Mini_map_Mask(const cv::Mat& BGR_image)
{
    cv::Mat BGR_inverted;   cv::Mat finish; 
    cv::cvtColor(BGR_image, BGR_inverted, cv::COLOR_BGR2RGB);
    SaveDebugScreenshot(BGR_inverted, "map\\BGR_inverted");
   
    // пустыня
    if (location == 1) 
    {
        cv::Mat mask_road, mask_flag, mask_flag_two, water, water1, znak;
        //цвет непрозодимых препятсвий
        cv::inRange(BGR_inverted, cv::Scalar(130, 90, 65), cv::Scalar(160, 110, 85), mask_road);

        cv::GaussianBlur(mask_road, mask_road, cv::Size(3, 3), 1);     // размываем
        cv::threshold(mask_road, mask_road, 60, 255, cv::THRESH_BINARY); // закрашиваем размытое


        //маска флага который перекрывает непроходимую местность. (сделаем его тоже непроходимым)
        cv::inRange(BGR_inverted, cv::Scalar(65, 65, 20), cv::Scalar(95, 120, 40), mask_flag);
        cv::inRange(BGR_inverted, cv::Scalar(37, 45, 57), cv::Scalar(65, 65, 65), mask_flag_two);

        cv::bitwise_or(mask_flag, mask_flag_two, mask_flag);

        cv::GaussianBlur(mask_flag, mask_flag, cv::Size(3, 3), 1);     // размываем
        cv::threshold(mask_flag, mask_flag, 80, 255, cv::THRESH_BINARY); // закрашиваем размытое

        cv::bitwise_or(mask_flag, mask_road, finish);


        //маски для воды
        cv::inRange(BGR_inverted, cv::Scalar(125, 165, 175), cv::Scalar(190, 225, 235), water);
        cv::GaussianBlur(water, water, cv::Size(3, 3), 1);     // размываем
        cv::threshold(water, water, 100, 255, cv::THRESH_BINARY); // закрашиваем размытое


        cv::inRange(BGR_inverted, cv::Scalar(145, 150, 145), cv::Scalar(165, 175, 170), water1);
        cv::GaussianBlur(water1, water1, cv::Size(3, 3), 1);     // размываем
        cv::threshold(water1, water1, 100, 255, cv::THRESH_BINARY); // закрашиваем размытое


        cv::bitwise_or(water1, water, water);
        cv::bitwise_or(finish, water, finish);


        // пока сделаем приграды переходы между локациями, пока нет реализации глобального путишествия.
        cv::inRange(BGR_inverted, cv::Scalar(186,  121,  18), cv::Scalar(205,  133,  35), znak);
        for (int i = 0; i < 2; ++i) 
        {
            cv::GaussianBlur(znak, znak, cv::Size(7, 7), 3);     // размываем
            cv::threshold(znak, znak, 1, 255, cv::THRESH_BINARY); // закрашиваем размытое
        }
        cv::bitwise_or(finish, znak, finish);







        // Инвертируем: теперь дороги (проходимо) = белые, препятствия = чёрные
        cv::bitwise_not(finish, finish);
    }
    else if(location == 2)
    {
    
    
    
    }


    return finish;
}
cv::Mat Player_XY_tominimap(const cv::Mat& BGR_image)
{
    cv::Mat BGR_inverted; cv::Mat finish;
    cv::cvtColor(BGR_image, BGR_inverted, cv::COLOR_BGR2RGB);

    //цвет игрока
    cv::inRange(BGR_inverted, cv::Scalar(90, 168, 240), cv::Scalar(98, 179, 255), finish);

    SaveDebugScreenshot(finish, "map\\Player_minimap");

    return finish;
}
cv::Mat  Player_XY_image;
bool Impassable_objects_mini_map(const cv::Mat& screen, cv::Mat& visual)
{
    if (screen.empty())
    {
        std::cerr << "Пустое изображение для анализа синего цвета\n";
    }
    cv::Mat mini_map = screen.clone();
    mini_map = mini_map(minimap_roi);



    Player_XY_image = Player_XY_tominimap(mini_map); //смотрим не подошёл ли игрок к краю мира, из-за чего моет произойти так что игрок будет уже не в центре мини карты а скраю.(механика игры)
    std::vector<std::vector<cv::Point>> contours;     // Поиск игрока на мини-карте через контуры
    cv::findContours(Player_XY_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Point player_pos(-1, -1); // По умолчанию недействительная позиция
    for (const auto& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() > 2 && rect.width >= 3 && rect.height >= 3 && rect.width <= 15 && rect.height <= 15)
        {
            player_pos = cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
            break; // Нашли игрока, выходим из цикла
        }
    }

    // Обновляем глобальные координаты центра игрока
    squaretomap_center_x = player_pos.x;
    squaretomap_center_y = player_pos.y;
 



    cv::Mat minimap_mask = Mini_map_Mask(mini_map);
    minimap_mask_global = Mini_map_Mask(mini_map);




    // ===== Добавляем квадрат в игровой центр подобраный вручную  =====
    int square_size = 5; int half_size = square_size / 2;

    int square_x = squaretomap_center_x - half_size;
    int square_y = squaretomap_center_y - half_size;

    cv::cvtColor(minimap_mask, minimap_mask, cv::COLOR_GRAY2BGR);
    cv::rectangle(minimap_mask, cv::Rect(square_x, square_y, square_size, square_size), cv::Scalar(30, 30, 255), cv::FILLED);

     // Вставляем minimap_mask в визуал что бы видеть в реальном времени
    minimap_mask.copyTo(visual(minimap_roi));
    SaveDebugScreenshot(minimap_mask, "map\\final___");

    return true;
}


static std::vector<cv::Point> cachedTargets;
static size_t resource_index = 0;
//Функция для распознавания текста на русском языке
bool Text_detection_ru(const cv::Mat& image, int x, int y, int roiWidth, int roiHeight, const vector<wstring>& targetTexts, string& recognizedText, tesseract::TessBaseAPI& tesseract)
{
    if (image.empty()) {
        cerr << "Ошибка: Пустое изображение для OCR\n";
        recognizedText = "";
        return false;
    }

    // Определяем область ROI вокруг указанных координат
    int roiX = max(0, x - roiWidth / 2 );
    int roiY = max(0, y - roiHeight / 2 - 20);
    if (roiX + roiWidth > image.cols) roiWidth = image.cols - roiX;
    if (roiY + roiHeight > image.rows) roiHeight = image.rows - roiY;
    if (roiWidth <= 0 || roiHeight <= 0) {
        cerr << "Ошибка: Некорректные размеры ROI\n";
        recognizedText = "";
        return false;
    }

    cv::Rect roi(roiX, roiY, roiWidth, roiHeight);
    cv::Mat roiImage = image(roi).clone(); // Копируем ROI

    cv::cvtColor(roiImage, roiImage, cv::COLOR_BGR2GRAY);
    cv::Mat textMask = Color_Mask_Text(roiImage);
    

    // Увеличим изображение
    cv::resize(roiImage, roiImage, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
   

    // --- Tesseract OCR
   // Устанавливаем режим обработки: один текстовый блок
    tesseract.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    // Передаём изображение  напрямую
    tesseract.SetImage(roiImage.data, roiImage.cols, roiImage.rows, roiImage.channels(), roiImage.step);
    tesseract.SetSourceResolution(70);  // Подходит для скринов из игры

   
    // Распознаём текст
    char* text = tesseract.GetUTF8Text();
    recognizedText = text ? text : "";
    delete[] text;

    // Преобразуем UTF-8 строку из tesseract в wstring
    std::wstring wideText = utf8_to_wstring(recognizedText);
    std::wstring lowerWideText = to_lower_wstring(wideText);
    remove_garbage_chars(lowerWideText); 

    // Приводим текст к нижнему регистру для упрощения проверки
    transform(recognizedText.begin(), recognizedText.end(), recognizedText.begin(), ::tolower);

    // Сравниваем
    for (const auto& target : targetTexts)
    {
        if (lowerWideText.find(target) != std::wstring::npos)
        {
            std::wcout << L"\nНайдено слово: " << lowerWideText << L"\n";
            return true;
        }
    }

    return false;
}
//Метод нахождения ресурса и его добычи
bool FindAndHarvestResource( const std::vector<cv::Point>& targets, size_t& index, cv::Point& currentTarget, bool& hasTarget, int windowX, int windowY, tesseract::TessBaseAPI& tesseract, Vision& vision ) 
{
    if (targets.empty() || index >= targets.size()) {
        hasTarget = false;
        return false;
    }
    if (!hasTarget) 
    {
        // Выбираем ближайшую цель к центру экрана (персонажа)
        auto closest = std::min_element(targets.begin(), targets.end(), [&](const cv::Point& a, const cv::Point& b)
            {
                double distA = sqrt(pow(a.x - player_center_x, 2) + pow(a.y - player_center_y, 2));
                double distB = sqrt(pow(b.x - player_center_x, 2) + pow(b.y - player_center_y, 2));
                return distA < distB;
            });

        // Проверяем валидность ближайшей цели
        if (closest == targets.end() || closest->x <= 0 || closest->y <= 0)
        {
            hasTarget = false;
            currentTarget = cv::Point(-1, -1);
            cout << "Некорректная ближайшая цель, сброс\n";
            return false;
        }
        // Обновляем текущую цель
        currentTarget = targets[index];
        hasTarget = true;

    }

    // Отправляем мышку к цели и кликаем   // ждём выполнения добычы или распознавания ресурса перед следующим таким же действием.
    auto now = Clock::now();   auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - Cl_Resou).count();
    if (hasTarget && elapsed >= 4)
    {
        if (scaleFactor == 1.00f)   // умножаем
        {
            SmoothMove(windowX + static_cast<int>(currentTarget.x * scaleFactor), windowY + static_cast<int>(currentTarget.y * scaleFactor));

        }
        else
        {
            SmoothMove(windowX + static_cast<int>(currentTarget.x / scaleFactor), windowY + static_cast<int>(currentTarget.y / scaleFactor));
        }
       

        // проверяем точно ли это ресурс и добываем     // повторяем проверку текста  X раз
        for (int repeat = 0; repeat < 7; ++repeat)
        {
            string recognizedText;
            Mat screen = vision.CaptureScreen(resol);
            std::vector<std::wstring> targetTexts =
            {
                // Варианты для L"пенька"
                L"пенька",
                L"енька",
                L"пеньк",
                // Варианты для L"необычная пенька"
                L"необычная пенька",
                L"необычная пенка",    // замена "ь" на ""
                L"необычная пеньк",    // укороченный вариант
                L"необычна пенька",    // пропуск "я"                                  
                L"необычн пенька",     // пропуск "ая"
                L"неабычная пенька",   // замена "о" на "а"
                L"необычнaя пенка",    // замена "ь" и пропуск "я"

                // Варианты для L"Редкая пенька"
                L"Редкая пенька",
                L"Редкая пенка",       // замена "ь" на ""
                L"Редкая пеньк",       // укороченный вариант
                L"Редка пенька",       // пропуск "я"
                L"Pедкая пенька",      // замена "Р" на "P" (OCR-ошибка)
                L"Редкaя пенка",       // замена "ь" и пропуск "я"
                L"Реткая пенька",      // замена "д" на "т"

                // Варианты для L"Уникальная пенька"
                L"Уникальная пенька",
                L"Уникальная пенка",   // замена "ь" на ""
                L"Уникальная пеньк",   // укороченный вариант
                L"Уникальна пенька",   // пропуск "я"
                L"Уникaльная пенька",  // пропуск "а"
                L"Уникальная пенка",   // замена "ь" на ""

                // Варианты для L"Первозданная пенька"
                L"Первозданная пенька",
                L"Первозданная пенка", // замена "ь" на ""
                L"Первозданная пеньк", // укороченный вариант
                L"Первоздана пенька",  // пропуск "я"
                L"Первoзданная пенька",// замена "о" на "o"
                L"Первозданнaя пенка", // замена "ь" и пропуск "я"
                L"Перво3данная пенька" // замена "з" на "3" (OCR-ошибка)

            };

            if (Text_detection_ru(screen, currentTarget.x, currentTarget.y, 400, 80, targetTexts, recognizedText, tesseract))
            {
                if (DEBUG == 1) { cout << "\n-----Добываю ресурс----\n"; }
                repeat = 7;
                LeftClick();

                Cl_Resou = now;
                return true; // Ресурс найден и добывается

            }
        }

        //НЕ ресурс — переходим к следующей цели
        index++;  // следующая цель
        Cl_Resou = now; // сбрасываем общее время цыкла каждый раз для того что бы персонаж успел подойди к ресурсу

    }
    return false;
}




/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////   РЫБАЛКА НА МЕСТЕ  ///////////////////////////////////

/////////// ищем поплавок в воде
vector<cv::Point> detected_poplavok;  int strengobjvision; cv::Mat BGR_p_inverted;
cv::Mat Color_Mask_poplavok(const cv::Mat& BGR_image)
{
    cv::cvtColor(BGR_image, BGR_p_inverted, cv::COLOR_BGR2RGB);
    cv::Mat finish, blue1, blue2;  


    cv::inRange(BGR_p_inverted, cv::Scalar(145, 30, 35), cv::Scalar(255, 105, 123), blue1);  // светлый
    cv::inRange(BGR_p_inverted, cv::Scalar(85, 15, 15), cv::Scalar(230, 60, 115), blue2);  // тёмный
    

    cv::bitwise_or(blue1, blue2, finish);
    SaveDebugScreenshot(finish, "fish\\poplavok__mask");

    if (false) 
    {
        cv::GaussianBlur(finish, finish, cv::Size(3, 3), 1);     // размываем
        SaveDebugScreenshot(finish, "fish\\poplavok__razmiv");
        cv::threshold(finish, finish, 40, 255, cv::THRESH_BINARY); // закрашиваем размытое

        SaveDebugScreenshot(finish, "fish\\poplavok__finish_obrabotki");
    }

    return finish;
}
bool Detect_poplavok(const cv::Mat& screen, cv::Mat& visual, cv::Point& poplavok_center)
{
    if (screen.empty())
    {
        std::cerr << "Пустое изображение для анализа поплавка\n";
        return false;
    }
    detected_poplavok.clear();

    cv::Mat clearscrn = screen;

    // Закрашиваем ненужный UI, в чёрный.  на экране размером 1920 на 1080 стартовая точка слева вверху 0 на 0;
    if (resol == 1)
    {


        // Закрашиваем ненужный UI чёрным (BGR: 0, 0, 0)

        // 1. Левый верхний угол (Статус персонажа)
        cv::rectangle(clearscrn, cv::Rect(0, 0, 610, 185), cv::Scalar(0, 0, 0), cv::FILLED);

        // 2. Верхняя панель справа (настройки)
        cv::rectangle(clearscrn, cv::Rect(1300, 0, 620, 85), cv::Scalar(0, 0, 0), cv::FILLED);

        // 3. Узкая вертикальная панель справа (настройки)
        cv::rectangle(clearscrn, cv::Rect(1860, 0, 60, 260), cv::Scalar(0, 0, 0), cv::FILLED);

        // 4. Нижняя панель (способности)
        cv::rectangle(clearscrn, cv::Rect(490, 970, 1430, 110), cv::Scalar(0, 0, 0), cv::FILLED);



    }
    else if (resol == 2)
    {
        // Закрашиваем ненужный UI чёрным (BGR: 0, 0, 0)

            // 1. Левый верхний угол (Статус персонажа)
        cv::rectangle(clearscrn, cv::Rect(0, 0, 220, 150), cv::Scalar(0, 0, 0), cv::FILLED);

        // 2. Верхняя панель справа (настройки)
        cv::rectangle(clearscrn, cv::Rect(930, 0, 436, 60), cv::Scalar(0, 0, 0), cv::FILLED);

        // 3. Узкая вертикальная панель справа (настройки)
        cv::rectangle(clearscrn, cv::Rect(1320, 0, 46, 142), cv::Scalar(0, 0, 0), cv::FILLED);

        // 4. Нижняя панель (способности)
        cv::rectangle(clearscrn, cv::Rect(350, 692, resolution_X - 350, resolution_Y - 692), cv::Scalar(0, 0, 0), cv::FILLED);
    }

    POINT cursor;
    GetCursorPos(&cursor);

    // переводим координаты курсора в систему скриншота 
    int cursorX = static_cast<int>(cursor.x * scaleFactor);
    int cursorY = static_cast<int>(cursor.y * scaleFactor);

    // Размер области вокруг мыши 
    int roiW = 200;
    int roiH = 200;

    int roiX = std::max(0, cursorX - roiW / 2);
    int roiY = std::max(0, cursorY - roiH / 2);

    // Чтобы не вылезло за границы картинки
    if (roiX + roiW > screen.cols) roiX = screen.cols - roiW;
    if (roiY + roiH > screen.rows) roiY = screen.rows - roiH;

    cv::Rect roi(roiX, roiY, roiW, roiH);

    // На всякий случай проверим, чтобы ROI не вылез за границы
    roi &= cv::Rect(0, 0, screen.cols, screen.rows);
    
    cv::Mat cut_scr = clearscrn(roi);
    cv::Mat mask = Color_Mask_poplavok(cut_scr);



    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect bestRect; int bestArea = 0; bool found = false;

    for (const auto& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);

        // Условие нахождения для 1920 х 1080
        if (resol == 1 && rect.area() > 10 && rect.width >= 7 && rect.height >= 12 && rect.width <= 25 && rect.height <= 25)
        {
            if (rect.area() > bestArea)  // выбираем самый большой
            {
                bestArea = rect.area();
                bestRect = rect;
                found = true;
            }
        }
        // Условие нахождения для 1366 х 768
        else if (resol == 2 && rect.area() > 8 && rect.width >= 5 && rect.height >= 10 && rect.width <= 20 && rect.height <= 20)
        {
            if (rect.area() > bestArea)
            {
                bestArea = rect.area();
                bestRect = rect;
                found = true;
            }
        }
    }



    // после цикла проверяем, нашли ли лучший вариант
    if (found)
    {
        cv::Point global_center = bestRect.tl() + roi.tl() + cv::Point(bestRect.width / 2, bestRect.height / 2);
        detected_poplavok.push_back(global_center);

        // визуализация
        cv::rectangle(visual, bestRect + roi.tl(), cv::Scalar(2, 235, 2), 0.5);
        cv::Mat visualcut = visual(roi);
        SaveDebugScreenshot(visualcut, "fish\\poplavok__vision");

        poplavok_center = global_center;
        return true;
    }




    return false; // поплавка нет
}


/////////// прогресбар ловли рыбы
vector<cv::Point> Progress_f; 
cv::Mat progress_bar_minigame(const cv::Mat& BGR_image)
{
    cv::Mat finish; cv::Mat BGR_inverted; 
    cv::cvtColor(BGR_image, BGR_inverted, cv::COLOR_BGR2RGB);

    // Основной диапазон ловли рыбы
    cv::inRange(BGR_inverted, cv::Scalar(60, 115, 25), cv::Scalar(80, 136, 40), finish); //  зелень

    cv::GaussianBlur(finish, finish, cv::Size(7, 7), 3);     // размываем
    cv::threshold(finish, finish, 50, 255, cv::THRESH_BINARY); // закрашиваем размытое

   
    SaveDebugScreenshot(finish, "fish\\progress_bar_finish");
    return finish;

}
bool Detect_minigame(const cv::Mat& screen, cv::Mat& visual)
{

    Progress_f.clear(); // Очищаем старые центры
    cv::Mat cut_scr = screen.clone();

    // предварительно обрезаем зону 
    int x = 0, y = 0, w = 0, h = 0;
    if (resol == 1) { x = 837; y = 537, w = 1073 - x; h = 573 - y; }
    else if (resol == 2) { x = 601; y = 382, w = 764 - x; h = 407 - y; }

    cv::Rect roi = cv::Rect(x, y, w, h);

    


    cut_scr = cut_scr(roi);
    bool found = false;




    cv::Mat bar_masked = progress_bar_minigame(cut_scr);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bar_masked, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);

        if (rect.area() > 10 && rect.width >= 50 && rect.height >= 10)
        {
            found = true;
            // Переводим координаты из cut_scr в глобальные
            cv::Point global_center = rect.tl() + roi.tl() + cv::Point(rect.width / 2, rect.height / 2);
            Progress_f.push_back(global_center);

        }
    }


    return found;
}



///////////// нахождения обьекта мини игры
vector<cv::Point> object_f;
cv::Mat object_to_bar(const cv::Mat& BGR_image)
{
    cv::Mat finish; cv::Mat BGR_inverted; cv::Mat blue, white;
    cv::cvtColor(BGR_image, BGR_inverted, cv::COLOR_BGR2RGB);

    cv::inRange(BGR_inverted, cv::Scalar(150, 0, 0), cv::Scalar(250, 60, 20), blue);

    cv::inRange(BGR_inverted, cv::Scalar(190, 130, 110), cv::Scalar(250, 240, 220), white);
    cv::bitwise_or(blue, white, finish);

    cv::GaussianBlur(finish, finish, cv::Size(5, 5), 3);     // размываем
    cv::threshold(finish, finish, 50, 255, cv::THRESH_BINARY); // закрашиваем размытое

    return finish;

}
bool refresh_pos_object(const cv::Mat& screen, cv::Point& global_center, int& roi_center_x) 
{
    object_f.clear();

    // предварительно обрезаем зону 
    int x = 0, y = 0, w = 0, h = 0;
    if (resol == 1) { x = 837; y = 537, w = 1073 - x; h = 573 - y; }
    else if (resol == 2) { x = 601; y = 382, w = 764 - x; h = 407 - y;  }

     cv::Rect roi = cv::Rect(x, y, w, h);

    cv::Mat cut_scr = screen.clone();  cut_scr = cut_scr(roi);
    cv::Mat object_masked = object_to_bar(cut_scr);


    std::vector<std::vector<cv::Point>> conto;
    cv::findContours(object_masked, conto, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    roi_center_x = roi.x + (roi.width / 2 + 20); // Центр roi по x

    cv::Rect centerRect
    (
        roi_center_x - 3,   // смещение влево (чтобы ширина была 6px)
        roi.y,              // верх ROI
        6,                  // ширина
        roi.height          // высота (до низа ROI)
    );

    cv::rectangle(visual, centerRect, cv::Scalar(67, 2, 2), 1);
    SaveDebugScreenshot(visual, "fish\\visual_roi_center_x");

    for (const auto& contour : conto) {
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() > 10) 
        {
            global_center = rect.tl() + roi.tl() + cv::Point(rect.width / 2, rect.height / 2);
            object_f.push_back(global_center);

            return true; // Нашли объект
        }
    }
    return false; // Объект не найден


}


/////////// отслеживаем рывки поплавка
cv::Point last_poplavok = { -1, -1 }; 
bool Movement_poplavok(const cv::Point& current)
{
    static int prevY = -1;
    static auto prev_time = Clock::now();
    static std::deque<int> diffs;                  // история dy
    static std::deque<Clock::time_point> stamps;   // абсолютные таймштампы

    auto now = Clock::now();

    if (prevY == -1) {
        prevY = current.y;
        stamps.push_back(now);
        return false;
    }

    int dy = current.y - prevY;
    long long dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - prev_time).count();

    if (dt > 0) 
    {
        diffs.push_back(dy);
        stamps.push_back(now);

        // чистим всё, что старше 1500 мс
        auto now = Clock::now();  int checkk = std::chrono::duration_cast<std::chrono::milliseconds>(now - Clock_check).count();
        if(checkk > 2000)
        {
            diffs.clear();
            stamps.clear();
            Clock_check = now;
        }



        // считаем характеристики колебаний
        int direction_changes = 0;
        for (size_t i = 1; i < diffs.size(); i++) 
        {
            if ((diffs[i] > 0 && diffs[i - 1] < 0) || (diffs[i] < 0 && diffs[i - 1] > 0))  { direction_changes++; }
        }

        double sum_abs = 0;
        for (auto d : diffs) { sum_abs += std::abs(d); }

        if (DEBUG == 1 ) // && direction_changes != 0
        {
            std::cout << "Изменения=" << diffs.size()  << " Повторы=" << direction_changes  << " Сумм.ампл=" << sum_abs << "\n";
        }


        // Условие поклёвки для 1920 х 1080
        if (resol == 1 &&   (direction_changes >= 1 && sum_abs > 13 || direction_changes >= 2 && sum_abs > 9)   )
        {        
            return true;
        }
        // Условие поклёвки для 1366 х 768
        if (resol == 2 && (direction_changes >= 1 && sum_abs >= 10 || direction_changes >= 2 && sum_abs > 7)  )
        {         
            return true;
        }

    }

    prevY = current.y;
    return false;
}


/////////// Запись точек для рыбалки
struct FishPoint 
{
    int x, y;
};
std::vector<FishPoint> fishPoints;  // список точек для заброса
int currentPointIndex = 0;          // активная точка
int attemptCounter = 0;             // сколько раз уже пытались на этой точке
// запись точки при нажатии "+"
void RecordFishPoint() 
{
    
        cout << "\n=== Режим записи ловли===\n";
        cout << "-------------------------- СПРАВКА -----------------------\n";
        cout << "Отдалите камеру на максимум, включите RTS управление в настройках игры.\n";
        cout << "Забросьте удочку на максимум, наведите мышкой на поплавок и нажмите +\n";
        cout << "-  [ + ] чтобы записать шаг.\n";
        cout << "-  [ X ] Очистить все точки.\n";
        cout << "-  [ С ] Выход.\n";
        cout << "----------------------------------------------------------\n\n";

        while (true)
        {
            // ЛКМ
            if (GetAsyncKeyState(VK_ADD) & 0x8000)
            {
                POINT p;
                GetCursorPos(&p);
                fishPoints.push_back({ p.x, p.y });
                if (DEBUG) { std::cout << "Записана точка (" << p.x << ", " << p.y << ")\n"; }

                while (GetAsyncKeyState(VK_ADD) & 0x8000) Sleep(10);
            }

            // Очистить все точки (X)
            if (GetAsyncKeyState('X') & 0x8000)
            {
                fishPoints.clear();
                currentPointIndex = 0;
                attemptCounter = 0;
                cout << "Все точки очищены!\n";

                while (GetAsyncKeyState('X') & 0x8000) Sleep(10);
            }

            // Выйти
            if (GetAsyncKeyState('C') & 0x8000)
            {
                while (GetAsyncKeyState('C') & 0x8000)  Sleep(10);
                break;
            }
        }
   
}

////////  использование наживки 
struct BaitPoint
{
    int x, y;
};
std::vector<FishPoint> baitsPoints;  
int numberofbaits;          // кол-во наживки (сколько максимум раз можно использовать )
void Record_baits_Point()
{
    cout << "\n=== Режим записи использвования наживки===\n";
    cout << "-------------------------- СПРАВКА -----------------------\n";
    cout << "Наводьте на кнопки которые нужно нажимать по очерёдно, Инвентарь > Наживка > Использовать > Закрыть инвентарь\n\n";
    cout << "-  [ + ] чтобы записать точку где нужно нажать ЛКМ.\n";
    cout << "-  [ X ] Очистить все точки.\n";
    cout << "-  [ С ] Выход.\n";
    cout << "----------------------------------------------------------\n\n";

    while (true)
    {

        // для ЛКМ
        if (GetAsyncKeyState(VK_ADD) & 0x8000)
        {
            POINT p;
            GetCursorPos(&p);
            baitsPoints.push_back({ p.x, p.y });
            if (DEBUG) { std::cout << "Записана точка (" << p.x << ", " << p.y << ")\n"; }

            while (GetAsyncKeyState(VK_ADD) & 0x8000) Sleep(10);
        }

        // Очистить все точки (X)
        if (GetAsyncKeyState('X') & 0x8000)
        {
            baitsPoints.clear();
            cout << "Все точки очищены!\n";

            while (GetAsyncKeyState('X') & 0x8000) Sleep(10);
        }

        // Выйти
        if (GetAsyncKeyState('C') & 0x8000)
        {
            while (GetAsyncKeyState('C') & 0x8000)  Sleep(10);
            break;
        }
    }
}
void Use_bait()
{
    if (numberofbaits <= 0) { return; }


    if (baitsPoints.empty())
    {
        std::cout << "Нет записанных точек для наживки!\n";
        return;
    }

    if (DEBUG) { std::cout << "Использую наживку Осталось: " << numberofbaits - 1 << "\n"; }

    for (size_t i = 0; i < baitsPoints.size(); i++)
    {
        SmoothMove(baitsPoints[i].x, baitsPoints[i].y);
        LeftClick();
        std::this_thread::sleep_for(std::chrono::milliseconds(400));

    }

    numberofbaits--;

}

////////распознавания текста на русском языке
cv::Mat Color_Mask_Text_recon(const cv::Mat& image)
{
    // image входит чисто в ч\б
    SaveDebugScreenshot(image, "fish\\pre__text");

    // alpha — контраст (1.0 — без изменений), beta — яркость (0 — без изменений)
    double alpha = 1.5;  // >1 усиливает контраст   
    int beta = -25;      // <0 затемняет
    image.convertTo(image, -1, alpha, beta);
    // SaveDebugScreenshot(image, "fish\\orig_text");

    cv::threshold(image, image, 130, 255, cv::THRESH_BINARY);
    SaveDebugScreenshot(image, "fish\\text");

    return image;
}
bool Text_toreconect_detection_ru(const cv::Mat& image, int roiX, int roiY, int roiWidth, int roiHeight, const vector<wstring>& targetTexts, string& recognizedText, tesseract::TessBaseAPI& tesseract)
{
    if (image.empty()) {
        cerr << "Ошибка: Пустое изображение для OCR\n";
        recognizedText = "";
        return false;
    }

    // Определяем область ROI вокруг указанных координат
    cv::Rect roi(roiX, roiY, roiWidth, roiHeight);

    cv::Mat roiImage = image(roi).clone(); // Копируем ROI

    cv::cvtColor(roiImage, roiImage, cv::COLOR_BGR2GRAY);
    cv::Mat textMask = Color_Mask_Text_recon(roiImage);

    // Увеличим изображение
    cv::resize(roiImage, roiImage, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);


    // --- Tesseract OCR
   // Устанавливаем режим обработки: один текстовый блок
    tesseract.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    // Передаём изображение  напрямую
    tesseract.SetImage(roiImage.data, roiImage.cols, roiImage.rows, roiImage.channels(), roiImage.step);
    tesseract.SetSourceResolution(70);  // Подходит для скринов из игры


    // Распознаём текст
    char* text = tesseract.GetUTF8Text();
    recognizedText = text ? text : "";
    delete[] text;

    // Преобразуем UTF-8 строку из tesseract в wstring
    std::wstring wideText = utf8_to_wstring(recognizedText);
    std::wstring lowerWideText = to_lower_wstring(wideText);
    remove_garbage_chars(lowerWideText);

    // Приводим текст к нижнему регистру для упрощения проверки
    transform(recognizedText.begin(), recognizedText.end(), recognizedText.begin(), ::tolower);

    // Сравниваем
    for (const auto& target : targetTexts)
    {
        if (lowerWideText.find(target) != std::wstring::npos)
        {
            if (DEBUG) { std::wcout << L"\nНайдено слово: " << lowerWideText << L"\n"; }
            return true;
        }
    }

    return false;
}
void Reconecting(const cv::Mat& image, tesseract::TessBaseAPI& tesseract, int windowX, int windowY)
{
        string recognizedText;  int roiX = 0, roiY = 0, roiW = 0, roiH = 0; int centerX, centerY;

        std::vector<std::wstring> targetTexts = { L"ок", };
        if (resol == 1)  // 1920 х 1080
        {
            roiX = 910;  roiW = 1010 - roiX;
            roiY = 533;  roiH = 562 - roiY;

            if (scaleFactor == 1.00f) // умножаем потому что scaleFactor == 1
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) * scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) * scaleFactor);
            }
            else
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) / scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) / scaleFactor);
            }
        }
        else if (resol == 2) // 1366 х 768
        {
            roiX = 643;  roiW = 721 - roiX;
            roiY = 374;  roiH = 405 - roiY;
            if (scaleFactor == 1.00f) // умножаем потому что scaleFactor == 1
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) * scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) * scaleFactor);
            }
            else
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) / scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) / scaleFactor);
            }
        }
        bool text_OK = Text_toreconect_detection_ru(image, roiX, roiY, roiW, roiH, targetTexts, recognizedText, tesseract);  recognizedText.clear(); targetTexts.clear();
        if (text_OK) { SmoothMove(centerX, centerY); LeftClick(); std::this_thread::sleep_for(std::chrono::milliseconds(8000)); text_OK = false;  }

        // Рисуем его на visual
        cv::Rect roiok(roiX, roiY, roiW, roiH);
        cv::rectangle(visual, roiok, cv::Scalar(0, 123, 123), 1);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////



        targetTexts = { L"переподключение", L"переподкл",L"ключение", };
        if (resol == 1)  // 1920 х 1080
        {
            roiX = 862; roiW = 1058 - roiX;
            roiY = 751; roiH = 778 - roiY;
            if (scaleFactor == 1.00f) // умножаем потому что scaleFactor == 1
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) * scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) * scaleFactor);
            }
            else
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) / scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) / scaleFactor);
            }
        }
        else if (resol == 2) // 1366 х 768
        {
            roiX = 613; roiW = 753 - roiX;
            roiY = 534; roiH = 553 - roiY;
            if (scaleFactor == 1.00f) // умножаем потому что scaleFactor == 1
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) * scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) * scaleFactor);
            }
            else
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) / scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) / scaleFactor);
            }
        }
        bool text_reconect = Text_toreconect_detection_ru(image, roiX, roiY, roiW, roiH, targetTexts, recognizedText, tesseract);  recognizedText.clear(); targetTexts.clear();
        if (text_reconect)  {  SmoothMove(centerX, centerY); LeftClick(); std::this_thread::sleep_for(std::chrono::milliseconds(8000)); text_reconect = false;   }

        // Рисуем его на visual
        cv::Rect roiRec(roiX, roiY, roiW, roiH);
        cv::rectangle(visual, roiRec, cv::Scalar(123, 34, 123), 1);


        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        targetTexts = { L"войти в игру", L"войти", L"игру", };
        if (resol == 1)  // 1920 х 1080
        {
            roiX = 1135; roiW = 1286 - roiX;
            roiY = 872;  roiH = 905 - roiY;
            if (scaleFactor == 1.00f) // умножаем потому что scaleFactor == 1
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) * scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) * scaleFactor);
            }
            else
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) / scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) / scaleFactor);
            }
        }
        else if (resol == 2) // 1366 х 768
        {
            roiX = 808; roiW = 914 - roiX;
            roiY = 620; roiH = 644 - roiY;
            if (scaleFactor == 1.00f) // умножаем потому что scaleFactor == 1
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) * scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) * scaleFactor);
            }
            else
            {
                centerX = windowX + static_cast<int>((roiX + roiW / 2) / scaleFactor); centerY = windowY + static_cast<int>((roiY + roiH / 2) / scaleFactor);
            }
        }

        bool text_login = Text_toreconect_detection_ru(image, roiX, roiY, roiW, roiH, targetTexts, recognizedText, tesseract);  recognizedText.clear(); targetTexts.clear();
        if (text_login)  {  SmoothMove(centerX, centerY); LeftClick(); std::this_thread::sleep_for(std::chrono::milliseconds(8000)); text_login = false;  }


        // Рисуем его на visual
        cv::Rect roilogin(roiX, roiY, roiW, roiH);
        cv::rectangle(visual, roilogin, cv::Scalar(20, 123, 78), 1);
        SaveDebugScreenshot(visual, "fish\\visual_posle_texta");
    
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Простой скриптер. зпись-действие  /////////////////////////////////////

int interval_farmSteps, interval_way_to_city, interval_city, interval_way_out_city, custom_pause, count_farm;
string filename_txt; //название файла сохранения

struct ActionStep
{
    int x, y;
    string button; // "LMB", "RMB", или "WAIT"
};

void SaveClick(vector<ActionStep>& steps, const string& button)
{
    POINT p;
    GetCursorPos(&p);
    steps.push_back({ p.x, p.y, button });
    cout << "Сохранено: (" << p.x << ", " << p.y << ") - " << button << endl;
}
void ClearSteps(vector<ActionStep>& steps, const string& stageName)
{
    steps.clear();
    cout << "\n>>> Этап '" << stageName << "' был очищен. Запиши заново.\n";
}
int debug = 0;
void RecordSteps(vector<ActionStep>& steps, const string& stageName)
{
    cout << "\n=== Режим записи: " << stageName << " ===\n";
    if (debug == 0)
    {
        debug = 1;
        cout << "-------------------------- СПРАВКА -----------------------\n";
        cout << "-  ЛКМ или ПКМ чтобы записать шаг.\n";
        cout << "- 'D' чтобы перейти на следующий этап.\n";
        cout << "- 'C' чтобы очистить текущий этап и начать заново.\n";
        cout << "- 'Z' чтобы очистить текущий этап и начать заново.\n";
        cout << "----------------------------------------------------------\n\n";
    }

    while (true)
    {
        // ЛКМ
        if (GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        {
            SaveClick(steps, "LMB");
            while (GetAsyncKeyState(VK_LBUTTON) & 0x8000) Sleep(10);
        }

        // ПКМ
        if (GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        {
            SaveClick(steps, "RMB");
            while (GetAsyncKeyState(VK_RBUTTON) & 0x8000) Sleep(10);
        }

        // Клавиша — новый этап
        if (GetAsyncKeyState('D') & 0x8000)
        {
            cout << "Этап '" << stageName << "' завершён. Шагов: " << steps.size() << "\n";
            while (GetAsyncKeyState('D') & 0x8000) Sleep(10);
            break;
        }

        // Очистить этап
        if (GetAsyncKeyState('C') & 0x8000)
        {
            ClearSteps(steps, stageName);
            while (GetAsyncKeyState('C') & 0x8000)  Sleep(10);
        }

        // пауза
        if (GetAsyncKeyState(VK_SPACE) & 0x8000)
        {
            steps.push_back({ 0, 0, "WAIT" });
            cout << "Сохранено: Пауза\n";
            while (GetAsyncKeyState(VK_SPACE) & 0x8000)  Sleep(100);
        }

        if (GetAsyncKeyState('Z') & 0x8000) // 'Z' — откат
        {
            if (!steps.empty()) { steps.pop_back();  cout << "Удалён последний шаг\n"; }
            else { cout << "Список уже пуст\n"; }

            while (GetAsyncKeyState('Z') & 0x8000) Sleep(100);
        }

    }
}
/// функция для простого воспроизведения задач при первом варианте программы 
void work(const ActionStep& step, int custom_pause)
{
    if (step.button == "LMB")
    {
        SmoothMove(step.x, step.y); // плавное перемещение
        LeftClick();
    }
    else if (step.button == "RMB")
    {
        SmoothMove(step.x, step.y); // плавное перемещение
        RightClick();
    }
    else if (step.button == "WAIT")
    {
        cout << "[Пауза] Ждём " << custom_pause << " сек...\n";
        this_thread::sleep_for(chrono::seconds(custom_pause));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////


int load, load_ii, sca;  int time_second_wait = 0, miniGameCounter = 0, soupKeyVK = 0;   bool pecks = false, soup = false;    int ckl = 300;
int main(int argc, char** argv)
{
    //служебный блок
    {
        setlocale(LC_ALL, "RU");
        SetConsoleOutputCP(CP_UTF8);

        filesystem::path Path = filesystem::canonical(filesystem::path(argv[0])).remove_filename();
        Path_directory = Path.string(); // Записываем путь
        wcout << L"\n [ путь в директорию ] "; cout << Path_directory << "\n\n";
    }

    //блок преднастроек
    {

            cout << "=== Режим Программы ===\n";
            cout << "[ 1 ] - Простой скриптер (запись-дейсвтие)\n";
            cout << "[ 2 ] - ИИ Бот  ( Сбор Ресурсов )\n";
            cout << "[ 3 ] - ИИ Бот  ( Рыбалка )\n";
            cout << ">> "; cin >> load_ii;  
            if (load_ii == 3) 
            { 
                cout << "Укажите количество наживки от 0 до 999 в одном слоте: "; cin >> numberofbaits; cout << "\n";
                cout << "Есть бафы которые нужно использовать? [1] - да [0] - нет >> "; int answer; cin >> answer;
                if (answer == 1) { soup = true; }
                if (soup) 
                {
                    char keyChar;
                    cout << "Введи клавишу на которой у вас назначен баф (0-9, A-Z): ";
                    cin >> keyChar;

                    // цифры
                    if (keyChar >= '0' && keyChar <= '9') {
                        soupKeyVK = 0x30 + (keyChar - '0');  // VK_0 = 0x30
                    }
                    // буквы
                    else if (keyChar >= 'A' && keyChar <= 'Z') {
                        soupKeyVK = 0x41 + (keyChar - 'A');  // VK_A = 0x41
                    }
                    else if (keyChar >= 'a' && keyChar <= 'z') {
                        soupKeyVK = 0x41 + (keyChar - 'a');  // VK_A = 0x41 (регистр не важен)
                    }
                    else {
                        cout << "Некорректная клавиша! Использую по умолчанию '2'\n";
                        soupKeyVK = 0x32; // VK_2
                    }

                    cout << "Будет использоваться клавиша: " << keyChar  << " (VK=" << hex << soupKeyVK << ")\n";
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            system("cls");

            cout << "=== Включить отладку === (Не рекомендуется для обычной работы бота)\n";
            cout << "[ 1 ] - Да\n";
            cout << "[ 2 ] - Нет\n";
            cout << ">> "; cin >> DEBUG; 
          

            cout << "=== Включить сохранение скринов === (Не рекомендуется для обычной работы бота)\n";
            cout << "[ 1 ] - Да\n";
            cout << "[ 2 ] - Нет\n";
            cout << ">> "; cin >> savescreen; 
            

            cout << "=== Введите ваше разрешение экрана ===\n";
            cout << "[ 1 ] - 1920 х 1080\n";
            cout << "[ 2 ] - 1366 х 768\n";
            cout << ">> "; cin >> resol;
            if (resol == 1) { resolution_X = 1920; resolution_Y = 1080; }
            else if (resol == 2) { resolution_X = 1366; resolution_Y = 768; }  


            cout << "=== Какой у вас скейл фактор экрана ===\n";
            cout << "[ 1 ] - 100%\n";
            cout << "[ 2 ] - 125%\n";
            cout << "[ 2 ] - 150%\n";
            cout << ">> ";  cin >> sca;
            if (sca == 1) { scaleFactor = 1.00f; }
            else if (sca == 2) { scaleFactor = 1.25f; }
            else if (sca == 3) { scaleFactor = 1.50f; } 
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            system("cls");
    }


    if (load_ii == 1) 
    {

        vector<ActionStep> farmSteps;
        vector<ActionStep> way_to_city;
        vector<ActionStep> city;
        vector<ActionStep> way_out_city;

        cout << "=== Настройка скриптера - Интервал между нажатиями (в секундах)- ===\n";
        cout << "---------------------------\n";
        cout << "Фарм: ";            cin >> interval_farmSteps;
        cout << "Кол-во шагов фарма: \n";  cin >> count_farm;
        cout << "---------------------------\n";
        cout << "Путь в город: ";  cin >> interval_way_to_city;
        cout << "Путь на фарм как и в город. ";    interval_way_out_city = interval_way_to_city;
        cout << "---------------------------\n";
        cout << "Дела в городе: "; cin >> interval_city;
        cout << "---------------------------\n";
        cout << "Пауза на пробел: ";  cin >> custom_pause;
        cout << "---------------------------\n\n";
        


            cout << "\nНажимай ЛКМ или ПКМ чтобы записать точку.\n";
            cout << "Нажми 'S' для продолжения.\n";
            while (true)
            {
                // Клавиша S — завершить
                if (GetAsyncKeyState('S') & 0x8000)
                {
                    break;
                }
            }

            RecordSteps(farmSteps, "Фарм");
            RecordSteps(way_to_city, "Путь");
            RecordSteps(city, "Дела в городе");
            RecordSteps(way_out_city, "Путь из города");


        cout << "\nОжидание 10 секунд перед стартом цикла...\n";
        Sleep(10000);

        // === Основной цикл работы ===
        while (true)
        {

            int count;  count = 0; //кол-во повторов фарма
            while (count < count_farm)
            {
                if (farmSteps.empty())
                {
                    cout << "Нет шагов фарма — пропускаю.\n";
                    break;
                }
                cout << "\nВыполняю шаги фарма\n";
                for (const auto& step : farmSteps)
                {
                    work(step, custom_pause);
                    this_thread::sleep_for(chrono::seconds(interval_farmSteps)); // задержка между автокликами во время фарма
                }
                cout << "count = " << count << "\n";
                count++;
            }



            if (way_to_city.empty())
            {
                cout << "Шагов по пути в город нет. Пропускаю.\n";
            }
            else
            {
                cout << "\nВыполняю путь в город\n";
                for (const auto& step : way_to_city)
                {
                    work(step, custom_pause);
                    this_thread::sleep_for(chrono::seconds(interval_way_to_city)); // задержка между автокликами по пути
                }
            }


            if (city.empty())
            {
                cout << "Шагов в городе нет. Пропускаю.\n";
            }
            else
            {
                cout << "\nВыполняю дела в городе\n";
                for (const auto& step : city)
                {
                    work(step, custom_pause);
                    this_thread::sleep_for(chrono::seconds(interval_city)); // задержка между автокликами в городе
                }
            }


            if (way_out_city.empty())
            {
                cout << "Шагов из города нет. Пропускаю.\n";
            }
            else
            {
                cout << "\nУхожу из города\n";
                for (const auto& step : way_out_city)
                {
                    work(step, custom_pause);
                    this_thread::sleep_for(chrono::seconds(interval_way_out_city)); // задержка между автокликами по пути обратно
                }
            }
        }

    }
    else if (load_ii == 2)
    {
        ///// код для настройки окна бота /////
        Vision vision;
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        cout << "=== ИИ-режим: Поиск и добыча ресурсов ===\n";
        cout << "Нажмите 'Q' для выхода.\n";

        // Инициализация Tesseract с явным указанием пути к tessdata
        tesseract::TessBaseAPI tesseract;
        if (tesseract.Init("C:\\Libs\\tesseract\\tessdata", "rus")) 
        {
            cerr << "Ошибка: Не удалось инициализировать Tesseract\n";
            return 1;
        };

            // Проверяем наличие окна игры
        HWND hWnd = FindWindow(NULL, L"Albion Online Client");
        if (!hWnd) 
        {
            cerr << "Ошибка: Окно 'Albion Online Client' не найдено. Запустите игру.\n";
            return 1;
        }
        else{
            cerr << "Окно 'Albion Online Client' найдено!\n";
        }

        // Получаем позицию и размеры окна
        RECT windowRect;
        GetWindowRect(hWnd, &windowRect);
        int windowX = windowRect.left;
        int windowY = windowRect.top;
        int windowWidth = windowRect.right - windowRect.left;
        int windowHeight = windowRect.bottom - windowRect.top;

        

        ///////////////////////////////////////////////////////

       
        while (true)
        {
            if (GetAsyncKeyState('Q') & 0x8000) { break; }

            // Захват экрана
            Mat screen = vision.CaptureScreen(resol);
            if (screen.empty())
            {
                if (DEBUG == 1) { cerr << "Не удалось захватить экран, пропускаем кадр\n"; }
                this_thread::sleep_for(chrono::milliseconds(100));
                continue;
            }
            visual = screen.clone();

            ckl++;  // смотрим нужно ли реконектится условно 300 циклов.
            if (ckl >= 300)
            {
                ckl = 0;
                Reconecting(screen, tesseract, windowX, windowY);
            }


            //реальный центр на игроке
            player_center_x = resolution_X / 2 - 10; player_center_y = resolution_Y / 2 - 80;
            cv::rectangle(visual, cv::Rect(player_center_x, player_center_y, 15, 15), cv::Scalar(5, 5, 255), cv::FILLED);


            //смотрим минни карту просчитываем центр
            Impassable_objects_mini_map(screen, visual);
            // визуализация карты  
            if (!current_path.empty())
            {
                cv::Point start(squaretomap_center_x, squaretomap_center_y);
                cv::Rect roi(1560, 740, 310, 240); // x, y, width, height  // Позиция центра миникарты на экране
                // Масштаб координат из minimap в экран
                double size_Factor = static_cast<double>(roi.width) / minimap_mask_global.cols;

                //////  Отрисовка пути  //////
                for (size_t i = 1; i < current_path.size(); ++i)
                {
                    cv::Point p1 = current_path[i - 1] * size_Factor + cv::Point(roi.x, roi.y);
                    cv::Point p2 = current_path[i] * size_Factor + cv::Point(roi.x, roi.y);
                    cv::line(visual, p1, p2, cv::Scalar(0, 255, 0), 2);

                    // Рисуем точки
                    cv::circle(visual, p1, 2, cv::Scalar(100, 0, 255), -1);
                }
                // Начало и цель
                cv::circle(visual, start * size_Factor + cv::Point(roi.x, roi.y), 4, cv::Scalar(255, 0, 0), -1);
                cv::circle(visual, goal * size_Factor + cv::Point(roi.x, roi.y), 4, cv::Scalar(0, 0, 255), -1);

            }



            //смотрим есть ли вокруг враги
            bool enemies_found = Detect_enemies_Objects(screen, visual);
            //смотрим добывается ли сейчас ресурс
            bool progress_bar_found = Detect_progress_bar_collection(screen, visual);

           
           
            if (!enemies_found && !progress_bar_found)
            {

                // Поиск ресурса    // Только если нет текущих целей cachedTargets.empty()
                auto now = Clock::now();  auto notarg = std::chrono::duration_cast<std::chrono::seconds>(now - wait_targ).count();
                if (cachedTargets.empty() && notarg > time_second_wait)
                {
                    if (DEBUG == 1) { cout << "-Поиск ресурсов\n"; }
                    cachedTargets = DetectAndDrawBlueObjects(screen, visual);
                    resource_index = 0;

                    // включаем паузу поиска ресов в том случае если cachedTargets всё ещё пустой
                    if (cachedTargets.empty()) { time_second_wait = 5; }
                    else { time_second_wait = 0; }

                    wait_targ = now;
                }

                // Если цели ещё остались то продолжаем обработку
                if (!cachedTargets.empty())
                {
                    is_moving = false; current_path.clear();  current_path_index = 1;

                    // ждём пока прогресс бар возможно появится ещё раз
                    auto now = Clock::now();  auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - wait_bar).count();
                    if (sec > 1)
                    {
                        // Идём и обрабатываем ресурс (в будущем добавить улсловие: если здоровье больше 1000)
                        bool harvested = FindAndHarvestResource(cachedTargets, resource_index, currentTarget, hasTarget, windowX, windowY, tesseract, vision);

                        // Все цели просмотрены, сбрасываем
                        if (!harvested && resource_index >= cachedTargets.size())
                        {
                            if (DEBUG == 1) { cout << "Все цели просмотрены! \n"; }
                            cachedTargets.clear();  resource_index = 0;
                        }
                    }
                }
                else
                {
                    if (is_moving && current_path_index < current_path.size())
                    {
                        // Выполняем случайное блуждание  с ожиданием между кликами
                        auto now = Clock::now();   auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_move_time).count();
                        if (milliseconds > RandomInt(10, 70))
                        {

                            Smart_SmoothMove(current_path[current_path_index], windowX, windowY); 
                            RightClick();

                            current_path_index++;
                            last_move_time = now; // Сбрасываем таймер
                        }
                    }
                    else
                    {
                        // Просчитываем путь
                        is_moving = false; current_path.clear();  current_path_index = 1;
                        Random_wandering_in_search(screen, visual, windowX, windowY);
                    }
                }

            }
            else if(enemies_found)
            {
                // Враг есть — просчитываем путь от него (убегаем)
                Run_out_enemy();

                if (is_moving && current_path_index < current_path.size())
                {
                    auto now = Clock::now();   auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_move_time).count();
                    if (milliseconds > RandomInt(10, 70))
                    {
                        Smart_SmoothMove(current_path[current_path_index], windowX, windowY); 
                        RightClick();
                        current_path_index++;
                        last_move_time = now;
                    }
                }
                else
                {
                    is_moving = false;
                    current_path.clear();
                    current_path_index = 1;
                }
            }
            






            vision.Show_Window(visual);
            this_thread::sleep_for(chrono::milliseconds(RandomInt(2, 6))); // Задержка между циклами
        }

        tesseract.End(); // Очистка Tesseract
    }
    else if (load_ii == 3)
    {
        ///// код для настройки окна бота /////
        RecordFishPoint();             system("cls"); // записываем направления ловли
        if (numberofbaits >= 1) {  Record_baits_Point(); }             system("cls"); // записываем использование наживки


        Vision vision;
        cout << "=== ИИ-режим: Рыбалка ===\n";
        cout << "Нажмите 'Q' для выхода.\n";


        // Инициализация Tesseract с явным указанием пути к tessdata
        tesseract::TessBaseAPI tesseract;
        if (tesseract.Init("C:\\Libs\\tesseract\\tessdata", "rus"))
        {
            cerr << "Ошибка: Не удалось инициализировать Tesseract\n";
            return 1;
        };
        // проводим тесты на текст
        {
            string recognizedText;
            cv::Mat img = cv::imread(Path_directory + "D_screenshot\\test\\step_OK_.png");
            int roiX = 0, roiY = 0, roiW = 0, roiH = 0;

            roiX = 643;  roiW = 721 - roiX;
            roiY = 374;  roiH = 405 - roiY;

            std::vector<std::wstring> targetTexts = { L"ок", L"к", L"о", };
            if (!Text_toreconect_detection_ru(img, roiX, roiY, roiW, roiH, targetTexts, recognizedText, tesseract)) { if (DEBUG) { cout << "[ ок ] не найдено\n"; } }
            recognizedText.clear();
            targetTexts.clear();

        }



        // Проверяем наличие окна игры
        HWND hWnd = FindWindow(NULL, L"Albion Online Client");
        if (!hWnd)
        {
            cerr << "Ошибка: Окно 'Albion Online Client' не найдено. Запустите игру.\n";
            return 1;
        }
        else {
            cerr << "Окно 'Albion Online Client' найдено!\n";
        }

        // Получаем позицию и размеры окна
        RECT windowRect;
        GetWindowRect(hWnd, &windowRect);
        int windowX = windowRect.left;
        int windowY = windowRect.top;

        
        bool trhow = false; bool One_passability = false; bool poplavok_was_lost = false; 
        while (true)
        {
            auto now = Clock::now();
            if (GetAsyncKeyState('Q') & 0x8000) { break; }
            // Захват экрана
            Mat screen = vision.CaptureScreen(resol);
            if (screen.empty())
            {
                if (DEBUG == 1) { cerr << "Не удалось захватить экран, пропускаем кадр\n"; }
                continue;
            }
            visual = screen.clone();

            ckl++;  // смотрим нужно ли реконектится условно 300 циклов.
            if (ckl >= 300) 
            {
                ckl = 0;
                Reconecting(screen, tesseract, windowX, windowY);
            }

            //смотрим  есть ли мини игра
            bool progress_found = Detect_minigame(screen, visual);
            // Ловим переход false -> true (детекция фронта сигнала)
            if (progress_found && !One_passability)
            {
                miniGameCounter++;  One_passability = true;
                std::cout << "Мини-игра " << miniGameCounter << " началась!\n";
            }


            cv::Point poplavok_center;
            bool poplavok_found = Detect_poplavok(screen, visual, poplavok_center);  //смотрим закинулась ли удочка проверяя есть ли на экране попловок в воде 


            //progress_found = рыба попалась играем в мини игру - (балансируем игровой обьект в центре найденой области)
            int weatbefore = std::chrono::duration_cast<std::chrono::seconds>(now - clock_weatbefore).count();
            int elapd = std::chrono::duration_cast<std::chrono::seconds>(now - clock_bars).count();
            if (progress_found)
            {

                    // Нажать ЛКМ
                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);

                    // играем
                    bool left = false; int count = 0; 
                    while (true) 
                    {
                        cv::Mat screen = vision.CaptureScreen(resol);

                        cv::Point global_center;
                        int roi_center_x;
                        bool object_found = refresh_pos_object(screen, global_center, roi_center_x);
                        

                        if (object_found)
                        {
                            if (global_center.x < roi_center_x + 15 )
                            {  left = true;  }
                            else 
                            {  left = false; break;  }
                        }
                        else 
                        {
                            if(left == true && count < 7)
                            { count++;  }
                            else 
                            {  break;  }
                        }

                    }

                    // Отпустить ЛКМ
                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                    clock_bars = now;  //если не обнаружен прогрес бар то мини игра закончена    (не обновляем таймер)
                    trhow = true;  //так-же  разрешаем закидывать удочку снова 



            }
            else if (!progress_found && elapd > 3 && weatbefore > 4) //ждём время после миниигры // и достаточно прошло времени ли после закидывания
            {
                if (miniGameCounter >= 10 && numberofbaits > 0) {  miniGameCounter = 0;  Use_bait(); } //применяем приманку
                One_passability = false;  // обновляем состояние когда уже рыбачим гарантируя правильность уже оконечной игры состоянием рыбалки 

               
                int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - clock_fish).count();
                int sec = std::chrono::duration_cast<std::chrono::milliseconds>(now - clock_wati).count();
                // ждём успокоения поплавка после закидывания и только потом проверяем
                if (sec > 1500)
                {
                    // если пропал поплавок, ждём не появится ли он снова, если нет то разрешаем закидывать удочку
                    static auto last_seen_time = Clock::now();
                    if (poplavok_found)
                    {
                        trhow = false;

                        if (poplavok_was_lost)
                        {
                            auto now = Clock::now();
                            int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_seen_time).count();

                            if (elapsed <= 2) // успел вернуться до 2 сек
                            {
                                if (DEBUG == 1)
                                {
                                    cout << ">>  Возможная Поклёвка (пропал и вернулся)!" << endl;
                                }
                                HoldMouseLeft(RandomInt(87, 180));
                                clock_fish = now;
                                clock_bars = now;
                                clock_wati = now;
                            }
                        }

                        // только после проверки обновляем время!
                        last_seen_time = Clock::now();
                        poplavok_was_lost = false;
                    }
                    else
                    {
                        // считаем сколько прошло с последнего обнаружения
                        auto now = Clock::now();
                        int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_seen_time).count();

                        if (elapsed >= 2)  // прошло 2 секунды без поплавка
                        {
                            trhow = true;
                            if (DEBUG == 1)
                            {
                                cout << " прошло 2 секунды без поплавка\n" << endl;
                            }
                        }

                        // если за краткий срок поплавок пропал из виду то предпологаем что его сильно пошатнула рыба (вытаскиваем)
                        if (!poplavok_was_lost)
                        {
                            // фиксируем момент исчезновения
                            poplavok_was_lost = true;
                            last_seen_time = now;
                        }


                    }
                }


                if (poplavok_found)
                {
                    
                    bool pecks = Movement_poplavok(poplavok_center);
                    if (pecks)
                    {
                        if (DEBUG == 1) 
                        {
                            cout << ">>  Поклёвка!" << endl;
                        }
                        HoldMouseLeft(RandomInt(87, 180));
                        clock_fish = now;   // сбрасываем таймер
                        clock_bars = now;
                        clock_wati = now;
                        
                    }
                    else if (elapsed > 60)
                    {
                        if (DEBUG == 1) 
                        {
                            cout << "Слишком долго без поклёвки, перезакидываем" << endl;
                        }
                        HoldMouseLeft(RandomInt(200, 300)); // вынимаем
                        std::this_thread::sleep_for(std::chrono::milliseconds(400));
                        HoldMouseLeft(RandomInt(1902, 2140)); // закидываем
                        clock_weatbefore = now;
                        clock_fish = now;
                        attemptCounter = 2;
                    }


                }
                else if (!poplavok_found && trhow)
                {

                    if (DEBUG == 1) {
                        std::cout << "Поплавка нет, пробуем точку " << currentPointIndex << " (" << fishPoints[currentPointIndex].x << ", "  << fishPoints[currentPointIndex].y << ")\n";
                    }

                    int polch = std::chrono::duration_cast<std::chrono::minutes>(now - clock_minute).count();
                        //если прошло 30 минут кушаем суп
                        if(soup && polch >= 30)
                        {
                            if (DEBUG) { cout << "Нажимаю кнопку бафа - " << soupKeyVK << "\n"; }
                           
                            PressKey(soupKeyVK, 1500); 
                            clock_minute = now;
                        }

                    SmoothMove(fishPoints[currentPointIndex].x, fishPoints[currentPointIndex].y);


                    HoldMouseLeft(RandomInt(1902, 2140));
                    std::this_thread::sleep_for(std::chrono::milliseconds(400));
                    clock_fish = now;
                    clock_bars = now;
                    trhow = false;
                    clock_wati = now;
                    clock_weatbefore = now;

                    // каждых X попыток переходим на следующую точку
                    attemptCounter++;
                    if (attemptCounter >= 1) 
                    {
                        attemptCounter = 0;
                        currentPointIndex = (currentPointIndex + 1) % fishPoints.size(); // по кругу
                        if (DEBUG == 1) {
                            std::cout << "Переходим на точку " << currentPointIndex << std::endl;
                        }
                    }


                }
               




               
            }

            if (DEBUG == 1)  {  vision.Show_Window(visual);  }
        }
    }

    return 0;
}