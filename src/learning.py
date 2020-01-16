import pickle

def save_model(model, path):
  f = open(path,'wb')
  pickle.dump(model, f)

def load_model(path):
  f = open(path, 'wb')
  model = pickle.load(f)
  return model