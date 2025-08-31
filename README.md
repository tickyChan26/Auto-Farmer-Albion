# Albion Online Automation Bot

## **Overview**
This project is an automation bot designed for resource farming using **computer vision** and **optical character recognition (OCR)**.  
It supports three modes: a simple "record-play" script, an **AI-based resource gathering mode**, and an **AI-based fishing mode**.  

The bot uses **OpenCV** for image processing, **Tesseract OCR** for text recognition, and **Windows API** for simulating input.

---

## **Features**

### **Simple Scripter**
- Records and plays back mouse clicks and pauses for repetitive tasks.  
- Customizable intervals for farming, traveling around the city, and city activities.  

### **AI Resource Gathering (Work in Progress)**
- Detects and gathers resources using **color-based identification**.  
- Implements **A* pathfinding** for navigation on the in-game minimap.  
- Detects and avoids enemies by analyzing health bars.  
- Supports different screen resolutions (currently **1920x1080** and **1366x768**) and Windows scaling.  

### **AI Fishing Bot**
- Automates fishing by detecting bobbers, progress indicators, and mini-game elements.  
- Supports bait usage and timed buffs (e.g., soups).  
- Handles in-game reconnection requests.  

---

## **Debugging**
- **Debug mode** for detailed logging.  
- Optional saving of screenshots for visual debugging.  

---

## **Project Status**
The project is **functional** but in **active development**.  
Planned improvements include:
- Improved resource detection  
- Additional location support  
- Global pathfinding  
- Automatic selling  
- Consumable purchasing  

---

## **Technologies**
- **OpenCV**: Image processing  
- **Tesseract OCR**: Text recognition  
- **Windows API**: Mouse and keyboard input simulation  
