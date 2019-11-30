import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
while(True):
    print("Enter the name of the image")
    name = input()
    if(name=="END"):
        break
    source_image = cv2.imread(name)
    width = source_image.shape[1]
    if(width<500):
        size = 1.5
    else:
        size = 3
    prediction = 'n.a.'

    PATH = './training.data'

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print ('training data is ready, classifier is loading...')
    else:
        print ('training data is being created...')
        open('training.data', 'w')
        color_histogram_feature_extraction.training()
        print ('training data is ready, classifier is loading...')

    color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
    prediction = knn_classifier.main('training.data', 'test.data')
    cv2.putText(
        source_image,
        'Prediction: ' + prediction,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        size,
        2000,
        )

    cv2.imshow('color classifier', source_image)
    cv2.waitKey(0)		
