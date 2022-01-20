from torch.utils.data import Dataset
class Dota2Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self,index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.X)