import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path

cap = cv2.VideoCapture(0)
(ret, frame) = cap.read()
prediction = 'n.a.'

PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

while True:

    (ret, frame) = cap.read()

    cv2.putText(
        frame,
        'Prediction: ' + prediction,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        200,
        )
    color_histogram_feature_extraction.color_histogram_of_test_image(frame)
    prediction = knn_classifier.main('training.data', 'test.data')
    
    cv2.imshow('color classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()		
