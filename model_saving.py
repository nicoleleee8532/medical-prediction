import pickle

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)