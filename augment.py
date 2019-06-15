from PIL import Image 
import glob
import pandas as pd

def main(): 
	filenames = glob.glob("{}/*.jpg".format(input_path))
	filenames = sorted(filenames)
	count = int(filenames[len(filenames)-1].split('\\')[1].split('.')[0]) +1	#name of last image in the directory
	total = len(filenames)														#number of images in the directory

	old_df=pd.read_csv(old_csv_path)
	new_df_train=pd.DataFrame(columns=["fullpath","angle"])
	new_df_test=pd.DataFrame(columns=["frame_id","angle"])


	i,j,k=0,0,0 #iterators for old_df , new_df_train , new_df_test
	for file in filenames:
		if(i%1000 == 0):
			print("processed "+str(i)+" images")
		try:
			img=Image.open(file)
			if(i/total<=0.8):
				img.save(output_train_path+file.split('\\')[1])
				new_df_train.loc[j] = [old_df['fullpath'].iloc[i], old_df['angle'].iloc[i]]
				j += 1
			else:
				img.save(output_test_path+file.split('\\')[1])
				new_df_test.loc[k] = [file.split('\\')[1].split('.')[0], old_df['angle'].iloc[i]]
				k += 1
			i += 1
		except IOError:
			print("IOError")
			pass

	i=0
	for file in filenames: 
		if(i%1000 == 0):
			print("mirrored "+str(i)+" images")
		try:
			img = Image.open(file)
			if(i/total<=0.8):
				transposed_img = img.transpose(Image.FLIP_LEFT_RIGHT)
				transposed_img.save(output_train_path+str(count)+".jpg")
				new_df_train.loc[j] = [fullpath+str(count)+".jpg", -old_df['angle'].iloc[i]]
				count += 1
				i += 1
				j += 1
			else:
				break
		except IOError:
			print("IOError")
			pass

	new_df_train.iloc[:int(0.8*len(new_df_train)),:].to_csv(new_train_csv_path)
	new_df_train.iloc[int(0.8*len(new_df_train)):,:].to_csv(new_valid_csv_path)
	new_df_test.to_csv(new_test_csv_path)
  
if __name__ == "__main__": 
	old_csv_path= "augment/train2.txt"
	new_train_csv_path = "data/train1.txt"
	new_valid_csv_path = "data/train3.txt"
	new_test_csv_path = "submissions/CH2_final_evaluation.csv"
	input_path = "augment/center"
	output_train_path = "data/train/center/"
	output_test_path ="data/test/center/"
	fullpath= "train/center/"
	main() 