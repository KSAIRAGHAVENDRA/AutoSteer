import pandas as pd
from matplotlib import pyplot

df1 = pd.read_csv('submissions/CH2_final_evaluation.csv')
df2 = pd.read_csv('submissions/final_predictions.csv')
df =pd.DataFrame(list(zip(df1['angle'],df2['steering_angle'])),columns=['expected','predicted'])
plt= df.plot(figsize=(20,10))
fig = plt.get_figure()
fig.savefig("plot.png",bbox_inches='tight',dpi=100)