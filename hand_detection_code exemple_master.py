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
        ##print(results.face_landmarks)
        
        ###landmarks
        
        ##face
        mp_drawing.draw_landmarks(img, 
                                  results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
        
        ##righ hand
        
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        
        ##left hand
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        
        ##pose
        mp_drawing.draw_landmarks(img, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
    
          
                
        
        
        
        
        
        cv2.imshow('Raw Webcam Feed', img)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
