import pickle
import pandas as pd

df = pd.read_csv('data\\raw\\creditcard.csv')
df = df.sample(frac=1).reset_index(drop=True)
y = df['Class']
y = y.values.reshape(-1, 1)
X = df.drop(columns=['Class'], axis=1)
X = X.values.reshape(-1, 30, 1)
print(X)

pickle_out = open('data\\processed\\X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('data\\processed\\y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
