import cv2
import dlib
from datetime import datetime as dt
import numpy as np

cap = cv2.VideoCapture('./HW/rick.mp4')
output_file = 'rick_effects.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255) # 白色
thickness = 2
lt = 14



out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= 1200:
        break
    a = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    if frame_count <= 100:

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        i = (frame_count)
        s = (frame_count)
        M1 = cv2.getRotationMatrix2D((w/2, h/2), i*25, s/80) #表示旋轉的中心點,表示旋轉的角度,圖像縮放因子
        rotate = cv2.warpAffine(frame, M1, (w, h))
        cv2.putText(rotate, "Original+rotate+scale", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(rotate, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = rotate
        
        
    if 150>frame_count>100:
        for ss in range(frame_count-100):
            x_offset = ss-20# 平移量
            y_offset = 0
            rows, cols, channels = frame.shape
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            frame = cv2.warpAffine(frame, M, (cols, rows))
        
        cv2.putText(frame, "Run Run", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = frame
        
    elif 200>frame_count>=150:
        x_offset = ss+200
        y_offset = 0
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        frame = cv2.warpAffine(frame, M, (cols, rows))
        
        for ss2 in range(frame_count-150):
            x_offset = ss2*-1  # 平移量
            y_offset = 0
            rows, cols, channels = frame.shape
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            frame = cv2.warpAffine(frame, M, (cols, rows))
        
        cv2.putText(frame, "Left Run", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = frame
        
        
    elif 200 <= frame_count < 300:

        kernel = np.ones((5, 5), np.uint8)
        NN = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)  # 形態學梯度
        # NN = cv2.erode(frame, kernel, iterations=1)  # 腐蝕
        # NN = cv2.dilate(frame, kernel, iterations=1)  # 膨脹
        cv2.putText(NN, "MorphologyEx", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(NN, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = NN
        
    elif 300 <= frame_count < 400:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 0, 150)
        # 在200至400幀中套用第二個特效
        cv2.putText(edges, "Canny", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(edges, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    elif 400 <= frame_count < 500:
        color_matrix = np.array([[0.5, 0.5, 0],
                                 [0.5, 0.5, 0],
                                 [0.4, 0, 0.8]])
        frame = cv2.transform(frame, color_matrix)
        
        cv2.putText(frame, "Warm", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
        frame_with_text = frame
    
    elif 500 <= frame_count < 600:
        color_matrix = np.array([[1, 0.5, 0],
                                 [0.5, 0.5, 0],
                                 [0.5, 0, 1]])
        frame = cv2.transform(frame, color_matrix)
        
        cv2.putText(frame, "Cold", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
        frame_with_text = frame
    
    elif 600 <= frame_count < 700:
        color_matrix = np.array([[0.5, 0.5, 0],
                                 [1, 1, 0],
                                 [0.5, 0.5, 0]])
        
        frame = cv2.transform(frame, color_matrix)
        
        cv2.putText(frame, "Green?", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
        frame_with_text = frame
    
    elif 700 <= frame_count < 800:
        color_matrix = np.array([[0.5, 0.8, 0],
                                 [0.5, 0.8, 0],
                                 [0, 0, 1]])
        frame = cv2.transform(frame, color_matrix)
        
        cv2.putText(frame, "Color 1", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
        frame_with_text = frame
        
    elif 800 <= frame_count < 900:
        color_matrix = np.array([[0.5, .5, 0],
                                 [0.5, .5, 0],
                                 [0.3, 0, .5]])
        frame = cv2.transform(frame, color_matrix)
        
        cv2.putText(frame, "Color 2", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
        frame_with_text = frame
    
    
    
    elif 900 <= frame_count < 1000:
        # 在400幀以後套用第三個特效
        # hconcat_img = cv2.hconcat([a,a])
        # frame_with_text = hconcat_img  # a是彩色影格，直接使用
        frame_with_text = a
        
        cv2.putText(frame_with_text, "Color EX", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame_with_text, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
    elif 1000 <= frame_count < 1200:
        detector = dlib.get_frontal_face_detector()   
        face_rects, scores, idx = detector.run(frame, 0, -.5)  # 偵測人臉
        for i, d in enumerate(face_rects):               # 取出所有偵測的結果
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA) # 以方框標示偵測的人臉
            cv2.putText(frame, f'{scores[i]:.2f}, ({idx[i]:0.0f})', (x1, y1), font,          # 標示分數
                        0.7, (255, 255, 255), 1, lt)
        
        
        cv2.putText(frame, "Face Detection", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = frame    
        
        
        
        
        
        
    # 在此放置影格處理程式碼，包括將特效應用於每一幀

    out.write(frame_with_text)
    frame_count += 1
    cv2.imshow("RickRoll", frame_with_text)
    

cap.release()
out.release()
cv2.destroyAllWindows()
