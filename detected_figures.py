import cv2
import numpy as np
from detect_shapes import *


def detect_shapes_and_colors(image_path: str, contours_list, shape_labels, color_hsv_ranges=None, output_path='output.jpg'):
    """
    Анализирует форму и цвет контуров на изображении.

    Args:
        image_path: Путь к входному файлу изображения.
        contours_list: Список контуров для анализа.
        shape_labels: Список меток форм для каждого контура.
        color_hsv_ranges: Словарь с диапазонами HSV для цветов.
        output_path: Путь для сохранения выходного изображения.

    Returns:
        List[tuple]: Список кортежей (контур, фигура, цвет) для каждого обнаруженного объекта.
    """

    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Ошибка в загрузке файла: {}".format(image_path))

    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color_hsv_ranges is None:
        color_hsv_ranges = {
            'red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([160, 50, 50]), np.array([180, 255, 255]))
            ],
            'green': (np.array([35, 50, 50]), np.array([85, 255, 255])),
            'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
            'yellow': (np.array([20, 50, 50]), np.array([35, 255, 255]))
        }

    # Функция для определения цвета по точке
    def get_color_at_point(hsv_img, x, y, color_ranges):
        
        if x < 0 or y < 0 or x >= hsv_img.shape[1] or y >= hsv_img.shape[0]:
            return 'unknown'
            
        hsv_pixel = hsv_img[y, x]
        
        for color_name, ranges in color_ranges.items():
            if isinstance(ranges, list):  
                for lower, upper in ranges:
                    # Создаем маску для одного пикселя
                    mask = cv2.inRange(np.array([[hsv_pixel]]), lower, upper)
                    if mask[0, 0] == 255:
                        return color_name
            else:  # Один диапазон
                lower, upper = ranges
                mask = cv2.inRange(np.array([[hsv_pixel]]), lower, upper)
                if mask[0, 0] == 255:
                    return color_name
        
        return 'unknown'

    # Анализ каждого контура
    results = []
    output_image = image.copy()
    
    for i, (contour, shape_label) in enumerate(zip(contours_list, shape_labels)):
        
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            
            if cx < 0 or cy < 0 or cx >= image.shape[1] or cy >= image.shape[0]:
                color = 'unknown'
            else:
                
                color = get_color_at_point(hsv, cx, cy, color_hsv_ranges)
            
           
            results.append((contour, shape_label, color))
            
            
            cv2.drawContours(output_image, [contour], -1, (0, 0, 0), 2)
            
            
            label = f"{shape_label} {color}"
            cv2.putText(output_image, label, (cx - 40, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            
            cv2.putText(output_image, str(i+1), (cx - 10, cy + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    
    cv2.imwrite(output_path, output_image)
    
    return results


def main():
    print("Print the image path (Enter for 'coins'):")
    image_path = input().strip() or None

    try:
        # Process image and extract contours
        image, contours = process_image(image_path)
        print(f"Contours found: {len(contours)}")

        # Classify contours
        circles, rects, others = classify_contours(contours, circularity_threshold=0.82)
        print(f"Circles: {len(circles)}, Rectangles: {len(rects)}, Others: {len(others)}")

        
        all_contours = circles + rects + others
        shape_labels = (['circle'] * len(circles) + 
                       ['rectangle'] * len(rects) + 
                       ['other'] * len(others))

        
        if image_path: 
            results = detect_shapes_and_colors(
                image_path=image_path,
                contours_list=all_contours,
                shape_labels=shape_labels
            )
            
            
            print("\nDetected objects:")
            for i, (contour, shape, color) in enumerate(results, 1):
                area = cv2.contourArea(contour)
                print(f"{i}: {shape} ({color}), area: {area:.1f}")

        return len(results) if image_path else 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    return 0

if __name__ == "__main__":
    main()
