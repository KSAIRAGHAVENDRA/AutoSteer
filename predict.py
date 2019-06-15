import numpy as np
from keras.models import load_model
import pandas as pd
import glob

data_path = "G:/Project/nvidia4/data/"
val_part = 3
model_path = "submissions/nvidia4_final_model.hdf5"
X_train_mean_path = "data/X_train_nvidia4_mean.npy"
row, col, ch =  192,256,3

test_path = "{}/X_test_preprocess.npy".format(data_path)
out_name = "submissions/final_predictions.csv"

print ("Loading model...")
model = load_model(model_path)

print ("Loading training data mean...")
X_train_mean = np.load(X_train_mean_path)

print ("Reading test data...")
X_test = np.load(test_path)
X_test = X_test.astype('float32', copy=False)
X_test -= X_train_mean
X_test /= 255.0

print ("Predicting...")
preds = model.predict(X_test)
preds = preds[:, 0]

# join predictions with frame_ids
filenames = glob.glob("{}/test/center/*.jpg".format(data_path))
filenames = sorted(filenames)
frame_ids = [f.replace(".jpg", "").replace("{}/test/center/".format(data_path), "") for f in filenames]

print ("Writing predictions...")
pd.DataFrame({"frame_id": frame_ids, "steering_angle": preds}).to_csv(out_name, index=False, header=True)

print ("Done!")

# Smoothening
#-------------------------------------------------------------------------------------
# delta=0.064500452503
# p = pd.read_csv("submissions/final_predictions.csv")["steering_angle"]
# for j in range(1,p.shape[0]):
# 	if(abs(p[j]-p[j-1]) > delta):
# 		if(p[j]>p[j-1]):
# 			p[j]=p[j-1]+delta
# 		else:
# 			p[j]=p[j-1]-delta
# pd.DataFrame({"frame_id": frame_ids, "steering_angle": p}).to_csv(out_name, index=False, header=True)


# Calculate RMSE
# ------------------------------------------------------------------------------------
print ("Calculating RMSE")
test = pd.read_csv("submissions/CH2_final_evaluation.csv")
predictions = pd.read_csv("submissions/final_predictions.csv")

t = test['angle']
p = predictions['steering_angle']

length = predictions.shape[0]
print ("Predicted angles: " + str(length))
sq = 0
mse = 0
for j in range(length):
    sqd = ((p[j] - t[j])**2)
    sq = sq + sqd
print(sq)
mse = sq/length
print(mse)
rmse = np.sqrt(mse)
print("model evaluated RMSE:", rmse)
