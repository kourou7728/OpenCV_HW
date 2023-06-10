#用這個
import cv2
from datetime import datetime as dt
import numpy as np

cap = cv2.VideoCapture('./HW/rick.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255) # 白色
thickness = 2
lt = 14
frame_count = 0
frame_with_text = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('rick_effect.mp4', fourcc, frame_count, (w, h))


while True:
    ret, frame = cap.read()
    if not ret or cv2.waitKey(24) == 27: break 
            
    a = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # edges = cv2.Canny(blur, 50, 150)
#     
    if frame_count <= 100:
        # cv2.putText(frame, "Original", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        # cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
        #         (50, 100), font, .8, font_color , 2, lt)
        # kernel = np.ones((5, 5), np.uint8)
        # NN = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)  # 形態學梯度
        # # NN = cv2.erode(frame, kernel, iterations=1)  # 腐蝕
        # # NN = cv2.dilate(frame, kernel, iterations=1)  # 膨脹
        # frame_with_text = NN
        i = (frame_count)
        s = (frame_count)
        M1 = cv2.getRotationMatrix2D((w/2, h/2), i*50, s/100) #表示旋轉的中心點,表示旋轉的角度,圖像縮放因子
        rotate = cv2.warpAffine(frame, M1, (w, h))
        frame_with_text = rotate
        
        
        
        
        
#-------------------------------------------------------------
#         for i in range(frame_count):
        
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             matrix = cv2.getRotationMatrix2D((width/2, height/2), 10, 1)
#             frame = cv2.warpAffine(frame, matrix, (width, height))
#         frame_with_text = frame
        
        
#-------------------------------------------------------------
        # for i in range(frame_count):
        #     rows, cols, channels = frame.shape
        #     angle = 30
        #     center = (cols // 2, rows // 2)
        #     M = cv2.getRotationMatrix2D(center, angle, 1)
        #     frame = cv2.warpAffine(frame, M, (cols, rows))       
        # cv2.putText(frame, "Original", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        # cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
        #         (50, 100), font, .8, font_color , 2, lt)
#-------------------------------------------------------------


        #無限平移
        # for i in range(frame_count):
        #     x_offset = 90  # 平移量
        #     y_offset = 0
        #     rows, cols, channels = frame.shape
        #     M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        #     frame = cv2.warpAffine(frame, M, (cols, rows))
#             if i < 100:
#                 M = np.float32([[1, 0, 2], [0, 1, 0]])
#             else:
#                 M = np.float32([[1, -0.005*(i+99), 0], [0, 1, 0]])
                
#         frame = cv2.warpAffine(frame, M, (640, 360)
        # frame_with_text = frame
                                          # 在前200幀中套用第一個特效        
                                          
    # if 200>frame_count>=101:
    #     x_offset = 90  # 平移量
    #     y_offset = 0
    #     rows, cols, channels = frame.shape
    #     M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    #     frame = cv2.warpAffine(frame, M, (cols, rows))
        
    if 150>frame_count>100:
        for i in range (1,90):
            x_offset = i # 平移量
            y_offset = 0
            rows, cols, channels = frame.shape
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            frame = cv2.warpAffine(frame, M, (cols, rows))
            frame_with_text = frame
        
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
        
    elif 200 <= frame_count < 400:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 0, 150)
        # 在200至400幀中套用第二個特效
        cv2.putText(edges, "Effect 1", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(edges, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
        frame_with_text = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
    elif 400 <= frame_count < 600:
        # 在400幀以後套用第三個特效
        # hconcat_img = cv2.hconcat(frame,a)
        # frame_with_text = hconcat_img  # a是彩色影格，直接使用
        frame_with_text = a
        cv2.putText(frame_with_text, "Effect 2", (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame_with_text, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms', 
                (50, 100), font, .8, font_color , 2, lt)
    
    frame_count += 1
    cv2.imshow("RickRoll", frame_with_text)
    
    
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)   
