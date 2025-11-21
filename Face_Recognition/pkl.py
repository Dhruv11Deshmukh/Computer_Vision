import pickle

# Open the .pkl file in read-binary mode
with open('known_faces.pkl', 'rb') as file:
    data = pickle.load(file)

# Now you can use 'data' (it could be a dict, list, model, etc.)
print(data)
