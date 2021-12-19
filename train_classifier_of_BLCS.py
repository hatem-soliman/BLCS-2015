from rgbhistograms import RGBHistograms
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import classifer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import glob
import numpy as np
import warnings
import cPickle
warnings.filterwarnings("ignore")


//path = open('C://Users/Mine/code/dataset')
// path to the dataset: pd=path_to_dataset = '/home/shrobon/Assignment2/code/dbpedia/'

#This folder contains the extracted herb texts



data = [] #  I will keep the text info here
target = []# This will contain the labels


desc = RGBHistograms([8,8,8])



#iterating over each text
for i in textPaths:
	text = cv2.imread(i)
	features = desc.describe(text) # a flattened histogram will be returned

	#updating the feature vector
	data.append(features)

	#updating the corresponding labels
	label = i.split('/')

	#Since there are 2 classes beginning with L
	label = label[len(label)-1]
	if label[0] == 'L':
		label = label[:3]
	else:
		label = label[:2]

	target.append(label)
	#print label


#Getting all the unique class names i have in my dataset
targetNames = np.unique(target)
le = LabelEncoder() #This will required to encode the class names as 0's and 1's
target = le.fit_transform(target)
print target
(trainData,testData,trainTarget,testTarget) = train_test_split(data,target,test_size = 0.3, random_state=42)

model = RandomForestClassifier(n_estimators = 25, random_state=84)
model.fit(trainData,trainTarget)
print classification_report(testTarget,model.predict(testData),target_names=targetNames)



#Saving my classifier to disk, so that i can use it later just for prediction
f = open("model.cpickle","wb")
f.write(cPickle.dumps(model))
f.close()




#I will now test my classification to see how good my classifier works
TestPaths = '/home/shrobon/Assignment2/code/testClassificationtexts/'
Testtexts = sorted(glob.glob((TestPaths)+'*.jpg'))


print "Testing the performance of my classifier"
print "::::::::::::::::::::::::::::::::::::::::"
test_counter = 0
for i in range(0,len(Testtexts)):
	test_counter = test_counter+1

	text = cv2.imread(Testtexts[i])
	feature = desc.describe(text)
	flower = le.inverse_transform(model.predict(feature))[0]



	x = Testtexts[i].split('/')
	x = x[len(x)-1]
	if x[0] =='L':
		x = x[:3]
	else:
		x = x[:2]
	print "Test number    :->{}".format(test_counter)
	print "Displayed herb :-> {}".format(x.upper())
	print "Predicted herb :-> {}".format(flower.upper())
	if flower == x :
		print "VERDICT : Correct !!! "
	else:
		print "VERDICT : Oops !! i missed it :( sorry "
	print ":::::::::::::::::::::::::::::::::::::::::::"
	cv2.imshow("The herb",text)
	cv2.waitKey(0)
