from torch.utils.data import Dataset


class FaultGraphDataset(Dataset):
    def __init__(self, fault_g_list):
        self.fault_g_list = fault_g_list

    def __len__(self):
        return len(self.fault_g_list)

    @staticmethod
    def collate_fn(data):
        return data

    def __getitem__(self, idx):
        fault_g = self.fault_g_list[idx]

        return fault_g
