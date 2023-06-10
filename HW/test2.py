import cv2
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
    if not ret or frame_count >= 600:
        break
    
    
    if frame_count <= 100:
        i = (frame_count)
        s = (frame_count)
        M1 = cv2.getRotationMatrix2D((width/2, height/2), i*50, s/100) #表示旋轉的中心點,表示旋轉的角度,圖像縮放因子
        rotate = cv2.warpAffine(frame, M1, (width, height))
        frame = rotate
        cv2.putText(frame, "Original", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
        
    if 150>frame_count>100:
        x_offset = 90 # 平移量
        y_offset = 0
        rows, cols, channels = frame.shape
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        frame = cv2.warpAffine(frame, M, (cols, rows))
        frame_with_text = frame
        
        cv2.putText(frame, "Original", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        
    elif 200>frame_count>=150:
        x_offset = -90  # 平移量
        y_offset = 0
        rows, cols, channels = frame.shape
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        frame = cv2.warpAffine(frame, M, (cols, rows))
        
        cv2.putText(frame, "Original", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = frame
        
        
        
        
        
        
        
        
        
        
    # 在此放置影格處理程式碼，包括將特效應用於每一幀

    out.write(frame)
    frame_count += 1
    cv2.imshow("RickRoll", frame)
    

cap.release()
out.release()
cv2.destroyAllWindows()
