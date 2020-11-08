
labels=['Topic '+str(i) for i in range(39)]

# set your data directory
data_dir='lib/models'

MODEL_URL='https://storage.googleapis.com/benayas_kaggle/topics_weights.pth' #example weights

# set some deployment settings
PORT=5000
print('settings loaded')
