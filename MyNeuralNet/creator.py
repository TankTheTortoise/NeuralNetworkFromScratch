import pickle


# filename includes path
def create(filename):
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model
