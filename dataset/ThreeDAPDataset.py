import random
from torch.utils.data import Dataset
import pickle as pkl
from scipy.spatial.transform import Rotation as R


class ThreeDAPDataset(Dataset):
    """_summary_
    This class is for the data loading.
    """
    def __init__(self, data_path, mode):
        """_summary_

        Args:
            data_path (str): path to the dataset
        """
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        if self.mode in ["train", "val", "test"]:
            self._load_data()
        else:
            raise ValueError("Mode must be train, val, or test!")

    def _load_data(self):
        self.all_data = []
        
        with open(self.data_path, "rb") as f:
            data = pkl.load(f)
        random.shuffle(data)
        
        if self.mode == "train": data = data[:int(0.7 * len(data))]
        elif self.mode == "val": data = data[int(0.7 * len(data)):int(0.8 * len(data))]
        else: data = data[int(0.8 * len(data)):]
        
        for data_point in data:
            for affordance in data_point["affordance"]:
                for pose in data_point["pose"][affordance]:
                    new_data_dict = {
                        "shape_id": data_point["shape_id"],
                        "semantic class": data_point["semantic class"],
                        "point cloud": data_point["full_shape"]["coordinate"],
                        "affordance": affordance,
                        "affordance label": data_point["full_shape"]["label"][affordance],
                        "rotation": R.from_matrix(pose[:3, :3]).as_quat(),
                        "translation": pose[:3, 3]
                    }
                    self.all_data.append(new_data_dict)
            
    def __getitem__(self, index):
        """_summary_

        Args:
            index (int): the element index

        Returns:
            shape id, semantic class, coordinate, affordance text, affordance label, rotation and translation
        """
        data_dict = self.all_data[index]
        return data_dict['shape_id'], data_dict['semantic class'], data_dict['point cloud'], data_dict['affordance'], \
            data_dict['affordance label'], data_dict['rotation'], data_dict['translation']
        
    def __len__(self):
        return len(self.all_data)
    

if __name__ == "__main__":
    random.seed(1)
    dataset = ThreeDAPDataset(data_path="../full_shape_release.pkl", mode="train")
    print(len(dataset))