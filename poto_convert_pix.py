# 图像预处理与特征提取
import cv2
import numpy as np
import turtle
from PIL import ImageGrab
import os
import dlib
from sklearn.mixture import GaussianMixture
import pandas as pd
import time

# 加载人脸检测器
detector = dlib.get_frontal_face_detector()

def analyze_image(image):
    """
    分析图像的平均亮度和对比度
    :param image: 输入的彩色图像
    :return: 平均亮度和对比度
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    contrast = np.std(gray)
    return avg_brightness, contrast

def preprocess_image(image_path, gaussian_kernel=(3, 3), low_threshold=15, high_threshold=45):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    
    # 分析图像特征
    avg_brightness, contrast = analyze_image(image)
    
    # 根据亮度和对比度动态调整 Canny 阈值
    if avg_brightness > 180:  # 非常亮的图像
        low_threshold = int(low_threshold * 1.2)
        high_threshold = int(high_threshold * 1.2)
    elif avg_brightness > 150:  # 较亮图像
        low_threshold = int(low_threshold * 1.1)
        high_threshold = int(high_threshold * 1.1)
    elif avg_brightness < 80:  # 非常暗的图像
        low_threshold = int(low_threshold * 0.6)
        high_threshold = int(high_threshold * 0.6)
    elif avg_brightness < 100:  # 较暗图像
        low_threshold = int(low_threshold * 0.8)
        high_threshold = int(high_threshold * 0.8)
    
    if contrast < 30:  # 极低对比度图像
        high_threshold = int(high_threshold * 0.7)
    elif contrast < 50:  # 低对比度图像
        high_threshold = int(high_threshold * 0.8)
    
    # 锐化处理增强图像细节
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪，减小高斯核大小
    blurred = cv2.GaussianBlur(gray, gaussian_kernel, 0)
    
    # 检测人脸
    faces = detector(gray)
    
    # 初始化边缘图像
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    face_rois = []
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # 根据人脸大小调整五官区域划分比例
        if w < 100:  # 小尺寸人脸
            padding_ratio = 0.3
        elif w > 300:  # 大尺寸人脸
            padding_ratio = 0.4
        else:
            padding_ratio = 0.35
        
        padding = int(min(w, h) * padding_ratio)  # 向内缩小
        x = max(0, x + padding)
        y = max(0, y + padding)
        w = max(1, w - 2 * padding)
        h = max(1, h - 2 * padding)
        face_rois.append((x, y, w, h))
        face_roi = blurred[y:y+h, x:x+w]
        
        # 双边滤波，在降噪的同时保留边缘
        face_roi = cv2.bilateralFilter(face_roi, 9, 75, 75)
        
        # 对比度限制直方图均衡化，增强对比度
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        face_roi = clahe.apply(face_roi)
        
        # 更精细的五官子区域划分
        eye_h = int(h * 0.25)
        eye_y = int(h * 0.15)
        brow_h = int(h * 0.12)
        brow_y = int(h * 0.05)
        nose_h = int(h * 0.32)
        nose_y = int(h * 0.32)
        mouth_h = int(h * 0.25)
        mouth_y = int(h * 0.65)

        def multi_scale_canny(roi, low, high):
            edges_1 = cv2.Canny(roi, low, high)
            edges_2 = cv2.Canny(roi, low * 2, high * 2)
            edges_3 = cv2.Canny(roi, low * 3, high * 3)
            return cv2.bitwise_or(edges_1, cv2.bitwise_or(edges_2, edges_3))

        # 初始化眉毛区域阈值
        left_brow_low_threshold = low_threshold
        left_brow_high_threshold = high_threshold
        right_brow_low_threshold = low_threshold
        right_brow_high_threshold = high_threshold

        # 眉毛区域，进一步降低阈值
        left_brow_roi = face_roi[brow_y:brow_y + brow_h, :int(w * 0.45)]
        right_brow_roi = face_roi[brow_y:brow_y + brow_h, int(w * 0.55):]
        
        left_brow_low_threshold = max(6, int(left_brow_low_threshold * 0.6))
        left_brow_high_threshold = max(18, int(left_brow_high_threshold * 0.6))
        right_brow_low_threshold = max(6, int(right_brow_low_threshold * 0.6))
        right_brow_high_threshold = max(18, int(right_brow_high_threshold * 0.6))
        
        left_brow_edges = multi_scale_canny(left_brow_roi, left_brow_low_threshold, left_brow_high_threshold)
        right_brow_edges = multi_scale_canny(right_brow_roi, right_brow_low_threshold, right_brow_high_threshold)
        face_roi[brow_y:brow_y + brow_h, :int(w * 0.45)] = left_brow_edges
        face_roi[brow_y:brow_y + brow_h, int(w * 0.55):] = right_brow_edges
        
        # 初始化眼睛区域阈值
        left_eye_low_threshold = low_threshold
        left_eye_high_threshold = high_threshold
        right_eye_low_threshold = low_threshold
        right_eye_high_threshold = high_threshold

        # 眼睛区域，进一步降低阈值
        left_eye_roi = face_roi[eye_y:eye_y + eye_h, :int(w * 0.45)]
        right_eye_roi = face_roi[eye_y:eye_y + eye_h, int(w * 0.55):]
        
        left_eye_low_threshold = max(8, int(left_eye_low_threshold * 0.6))
        left_eye_high_threshold = max(24, int(left_eye_high_threshold * 0.6))
        right_eye_low_threshold = max(8, int(right_eye_low_threshold * 0.6))
        right_eye_high_threshold = max(24, int(right_eye_high_threshold * 0.6))
        
        left_eye_edges = multi_scale_canny(left_eye_roi, left_eye_low_threshold, left_eye_high_threshold)
        right_eye_edges = multi_scale_canny(right_eye_roi, right_eye_low_threshold, right_eye_high_threshold)

        # 边缘筛选，过滤掉面积过小的边缘，进一步减小最小面积阈值
        def filter_small_edges(edges, min_area=30):
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_edges = np.zeros_like(edges)
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    cv2.drawContours(filtered_edges, [contour], -1, 255, 1)
            return filtered_edges

        left_eye_edges = filter_small_edges(left_eye_edges)
        right_eye_edges = filter_small_edges(right_eye_edges)

        # 使用形态学腐蚀操作细化边缘，减少迭代次数
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
        left_eye_edges = cv2.erode(left_eye_edges, kernel, iterations=1)
        right_eye_edges = cv2.erode(right_eye_edges, kernel, iterations=1)

        face_roi[eye_y:eye_y + eye_h, :int(w * 0.45)] = left_eye_edges
        face_roi[eye_y:eye_y + eye_h, int(w * 0.55):] = right_eye_edges
        
        # 初始化鼻子区域阈值
        nose_low_threshold = low_threshold
        nose_high_threshold = high_threshold

        # 鼻子区域，进一步降低阈值
        nose_roi = face_roi[nose_y:nose_y + nose_h, :]
        nose_low_threshold = max(6, int(nose_low_threshold * 0.6))
        nose_high_threshold = max(18, int(nose_high_threshold * 0.6))
        nose_canny_edges = multi_scale_canny(nose_roi, nose_low_threshold, nose_high_threshold)
        
        # 增加拉普拉斯边缘检测
        nose_laplacian_edges = cv2.Laplacian(nose_roi, cv2.CV_8U)
        nose_edges = cv2.bitwise_or(nose_canny_edges, nose_laplacian_edges)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        nose_edges = cv2.morphologyEx(nose_edges, cv2.MORPH_OPEN, kernel, iterations=1)
        nose_edges = cv2.morphologyEx(nose_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        face_roi[nose_y:nose_y + nose_h, :] = nose_edges
        
        # 初始化嘴巴区域阈值
        mouth_low_threshold = low_threshold
        mouth_high_threshold = high_threshold

        # 嘴巴区域，进一步降低阈值
        mouth_roi = face_roi[mouth_y:mouth_y + mouth_h, :]
        mouth_low_threshold = max(6, int(mouth_low_threshold * 0.6))
        mouth_high_threshold = max(18, int(mouth_high_threshold * 0.6))
        mouth_edges = multi_scale_canny(mouth_roi, mouth_low_threshold, mouth_high_threshold)
        
        # 增加 Sobel 边缘检测
        sobelx = cv2.Sobel(mouth_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mouth_roi, cv2.CV_64F, 0, 1, ksize=3)
        mouth_sobel_edges = cv2.bitwise_or(cv2.convertScaleAbs(sobelx), cv2.convertScaleAbs(sobely))
        mouth_edges = cv2.bitwise_or(mouth_edges, mouth_sobel_edges)
        
        face_roi[mouth_y:mouth_y + mouth_h, :] = mouth_edges
        
        # 初始化 Canny 阈值
        canny_low_threshold = low_threshold
        canny_high_threshold = high_threshold

        # 结合 Canny 和拉普拉斯算子的边缘检测，进一步降低阈值
        canny_edges = cv2.Canny(face_roi, max(12, int(canny_low_threshold * 0.6)), max(36, int(canny_high_threshold * 0.6)))
        laplacian_edges = cv2.Laplacian(face_roi, cv2.CV_8U)
        face_edges = cv2.bitwise_or(canny_edges, laplacian_edges)

        # 形态学操作
        kernel = np.ones((2, 2), np.uint8)
        face_edges = cv2.morphologyEx(face_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        face_edges = cv2.morphologyEx(face_edges, cv2.MORPH_OPEN, kernel, iterations=1)
        
        edges[y:y+h, x:x+w] = face_edges
    
    return edges, image.shape[:2], face_rois

def extract_contours(edges, face_rois):
    # 查找轮廓，使用 CHAIN_APPROX_SIMPLE 保留关键边界点
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 分级轮廓筛选，进一步减小筛选阈值
    primary_contours = []
    secondary_contours = []
    min_primary_contour_length = 20
    min_primary_contour_area = 60
    min_secondary_contour_length = 8
    min_secondary_contour_area = 20

    for contour in contours:
        if len(contour) > min_primary_contour_length and cv2.contourArea(contour) > min_primary_contour_area:
            primary_contours.append(contour)
        elif len(contour) > min_secondary_contour_length and cv2.contourArea(contour) > min_secondary_contour_area:
            secondary_contours.append(contour)

    # 合并主要和次要轮廓
    all_contours = primary_contours + secondary_contours

    filtered_contours = []
    for contour in all_contours:
        # 多边形逼近，调整脸部轮廓的平滑度，进一步减小 epsilon 值
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 修正脸部轮廓过圆的问题，重点处理下颌
        for x, y, w, h in face_rois:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if x <= cX <= x + w and y <= cY <= y + h and len(approx) > 60:
                    face_center_x = x + w / 2
                    face_center_y = y + h / 2
                    
                    # 提取下颌轮廓点
                    chin_points = []
                    for point in approx:
                        px, py = point[0]
                        if py > y + h * 0.8:
                            chin_points.append([px, py])
                    if len(chin_points) > 3:
                        chin_points = np.array(chin_points)
                        # 曲线拟合
                        z = np.polyfit(chin_points[:, 0], chin_points[:, 1], 3)
                        p = np.poly1d(z)
                        for i in range(len(approx)):
                            px, py = approx[i][0]
                            if py > y + h * 0.8:
                                fitted_y = p(px)
                                if py > fitted_y:
                                    approx[i][0][1] = int(py - (py - fitted_y) * 0.8)
                                else:
                                    approx[i][0][1] = int(py + (fitted_y - py) * 0.8)
        filtered_contours.append(approx)
    return filtered_contours

def draw_contours(contours, img_shape):
    """
    使用 turtle 绘制轮廓
    :param contours: 轮廓列表
    :param img_shape: 图像尺寸
    """
    turtle.setup(img_shape[1], img_shape[0])
    turtle.speed(0)
    turtle.penup()
    turtle.hideturtle()
    turtle.tracer(0)

    for contour in contours:
        for point in contour:
            x, y = point[0]
            turtle.goto(x - img_shape[1] / 2, img_shape[0] / 2 - y)
            turtle.pendown()
        turtle.penup()
    turtle.update()

def save_turtle_image(file_path):
    """
    保存 turtle 绘制的图像为 PNG 文件
    :param file_path: 保存路径
    """
    import PIL.Image as Image

    screen = turtle.getscreen()
    canvas = screen.getcanvas()
    canvas.postscript(file="temp.ps", colormode='color')
    
    # 使用 PIL 打开 PostScript 文件并转换为 PNG
    img = Image.open("temp.ps")
    img.save(file_path, "PNG")
    
    # 删除临时的 PostScript 文件，添加重试机制
    max_retries = 5
    retry_delay = 1  # 重试间隔时间，单位：秒
    for attempt in range(max_retries):
        try:
            os.remove("temp.ps")
            break
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"无法删除 temp.ps 文件，正在重试 ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print("多次尝试删除 temp.ps 文件失败，请手动删除该文件。")

def test_thresholds(image_path, low_thresholds, high_thresholds, min_contour_lengths, min_contour_areas):
    """
    测试不同的 Canny 阈值和轮廓筛选阈值对轮廓数量的影响
    :param image_path: 图像文件路径
    :param low_thresholds: Canny 低阈值列表
    :param high_thresholds: Canny 高阈值列表
    :param min_contour_lengths: 最小轮廓长度列表
    :param min_contour_areas: 最小轮廓面积列表
    :return: 包含不同参数组合下轮廓数量的字典
    """
    results = {}
    for low_threshold in low_thresholds:
        for high_threshold in high_thresholds:
            try:
                edges, img_shape, face_rois = preprocess_image(image_path, low_threshold=low_threshold, high_threshold=high_threshold)
                for min_contour_length in min_contour_lengths:
                    for min_contour_area in min_contour_areas:
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        filtered_contours = []
                        for contour in contours:
                            if len(contour) > min_contour_length and cv2.contourArea(contour) > min_contour_area:
                                filtered_contours.append(contour)
                        key = (low_threshold, high_threshold, min_contour_length, min_contour_area)
                        results[key] = len(filtered_contours)
            except FileNotFoundError:
                print(f"无法读取图像文件: {image_path}")
                break
    return results

# 主程序
if __name__ == "__main__":
    print(f"当前工作目录: {os.getcwd()}")
    # 获取用户输入的图像文件路径
    image_path = input("请输入图像文件的绝对路径: ")

    # 定义默认的参数范围
    low_thresholds = [10, 20, 30]
    high_thresholds = [40, 60, 80]
    min_contour_lengths = [10, 20, 30]
    min_contour_areas = [30, 60, 90]

    # 调用 test_thresholds 函数进行测试
    results = test_thresholds(image_path, low_thresholds, high_thresholds, min_contour_lengths, min_contour_areas)

    # 准备数据并保存为 Excel 文件
    data = {
        'Canny 低阈值': [],
        'Canny 高阈值': [],
        '最小轮廓长度': [],
        '最小轮廓面积': [],
        '轮廓数量': []
    }
    for key, value in results.items():
        low_threshold, high_threshold, min_contour_length, min_contour_area = key
        data['Canny 低阈值'].append(low_threshold)
        data['Canny 高阈值'].append(high_threshold)
        data['最小轮廓长度'].append(min_contour_length)
        data['最小轮廓面积'].append(min_contour_area)
        data['轮廓数量'].append(value)
    df = pd.DataFrame(data)
    excel_file_path = 'threshold_test_results.xlsx'
    try:
        df.to_excel(excel_file_path, index=False)
        print(f"测试结果已保存到 {excel_file_path}")
    except PermissionError:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_excel_file_path = f"threshold_test_results_{timestamp}.xlsx"
        try:
            df.to_excel(new_excel_file_path, index=False)
            print(f"由于权限问题，原文件无法写入，测试结果已保存到 {new_excel_file_path}")
        except Exception as e:
            print(f"保存文件时出现错误: {e}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")

    # 获取用户输入的参数组合
    user_input = input("请输入 Canny 低阈值, Canny 高阈值, 最小轮廓长度, 最小轮廓面积（用逗号分隔）: ")
    try:
        low_threshold, high_threshold, min_contour_length, min_contour_area = map(int, user_input.split(','))
    except ValueError:
        print("输入格式错误，请输入四个整数，用逗号分隔。")
        exit(1)

    # 检查输入是否在 Excel 文件中
    if 'new_excel_file_path' in locals():
        df = pd.read_excel(new_excel_file_path)
    mask = (df['Canny 低阈值'] == low_threshold) & (df['Canny 高阈值'] == high_threshold) & \
           (df['最小轮廓长度'] == min_contour_length) & (df['最小轮廓面积'] == min_contour_area)
    if not df[mask].empty:
        edges, img_shape, face_rois = preprocess_image(image_path, low_threshold=low_threshold, high_threshold=high_threshold)
        contours = extract_contours(edges, face_rois)
        draw_contours(contours, img_shape)

        # 保存图像
        output_file = f"sketch_{low_threshold}_{high_threshold}_{min_contour_length}_{min_contour_area}.png"
        save_turtle_image(output_file)
        print(f"素描图已保存为 {output_file}")
    else:
        print("输入的参数组合不在 threshold_test_results.xlsx 文件中，请重新输入。")