import mediapipe as mp
import cv2 



mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2)
    
    ##webcam
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        ###detection
        results=holistic.process(img)
        print(results.face_landmarks)
            
        ###landmarks
            
        ##tête
        mp_drawing.draw_landmarks(img, 
                                      results.face_landmarks,
                                      mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
            
        ##main droite
            
        mp_drawing.draw_landmarks(img, 
                                      results.right_hand_landmarks, 
                                      mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
            
        ## main gauce
        mp_drawing.draw_landmarks(img, 
                                      results.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                     )
            
        ##corp
        mp_drawing.draw_landmarks(img, 
                                      results.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
        
            
                    
            
            
            
            
            
        cv2.imshow('Raw Webcam Feed', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()