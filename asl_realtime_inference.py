import cv2
import numpy as np
import tensorflow as tf
import math

model = tf.keras.models.load_model('/home/aryan29/Downloads/asl_model(1).h5')

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']


cap = cv2.VideoCapture(2)

IMG_HEIGHT = 100
IMG_WIDTH = 100


sentence = ""
current_stable_prediction = ""
prediction_counter = 0
STABILITY_THRESHOLD = 15 
CONFIDENCE_THRESHOLD = 0.90 

while True:
    
    ret, frame = cap.read()
    if not ret:
        break


    frame_height, frame_width, _ = frame.shape

    frame = cv2.flip(frame, 1)
    
    # Define a Region of Interest (ROI) box
    x1, y1 = int(frame_width * 0.5) - 150, int(frame_height * 0.5) - 150
    x2, y2 = int(frame_width * 0.5) + 150, int(frame_height * 0.5) + 150
    
    # Ensure ROI is within frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)

    # Draw the ROI box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Extract the ROI
    roi = frame[y1:y2, x1:x2]
    

    if roi.shape[0] != IMG_HEIGHT or roi.shape[1] != IMG_WIDTH:
        if roi.size == 0:
            continue 
        roi_resized = cv2.resize(roi, (IMG_HEIGHT, IMG_WIDTH))
    else:
        roi_resized = roi
    
    img_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize dimensions
    img_array = np.expand_dims(img_rgb, axis=0) / 255.0  # (1, 96, 96, 3)


    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)
    predicted_letter = labels[predicted_class_index]

  
    live_text = f"{predicted_letter} ({confidence*100:.2f}%)"
    cv2.putText(frame, live_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if confidence >= CONFIDENCE_THRESHOLD:
        if predicted_letter == current_stable_prediction:
            prediction_counter += 1
        else:
            current_stable_prediction = predicted_letter
            prediction_counter = 1
            
        if prediction_counter == STABILITY_THRESHOLD:
            if predicted_letter == 'space':
                sentence += " "
            elif predicted_letter == 'del':
                sentence = sentence[:-1] 
            elif predicted_letter != 'nothing':
                sentence += predicted_letter
            

            prediction_counter = 0 
            
    else:
        current_stable_prediction = ""
        prediction_counter = 0

    
    paper_y = frame_height - 100
    cv2.rectangle(frame, (0, paper_y), (frame_width, frame_height), (255, 255, 255), -1)
    

    font_scale = 1.0
    thickness = 2
    
    max_line_width = frame_width - 20
    words = sentence.split(' ')
    display_line = ""
    y_text = paper_y + 30 

    for word in words:
        test_line = display_line + word + " "
        (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        if text_width > max_line_width:
            # Draw the current line and start a new one
            cv2.putText(frame, display_line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            y_text += 30 # Move to next line
            display_line = word + " "
        else:
            display_line = test_line
            
    cv2.putText(frame, display_line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


    cv2.imshow('ASL Typing Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
