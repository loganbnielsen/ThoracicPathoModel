from torch import Tensor

from torch.utils.data import Dataset

from os import path
import io

class MedicalImage:
    def __init__(self, img, class_name, class_id, rad_id, x_min, y_min, x_max, y_max):
        self.img = img
        self.class_name = class_name
        self.class_id = class_id
        self.rad_id = rad_id
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def split(self, num_x_splits, num_y_splits):
        list_X, list_Y = self.get_split_dims(num_x_splits, num_y_splits)
        return self.exact_split(list_X, list_Y)

    def get_split_dims(self, num_x_splits, num_y_splits, ignore_split_incompatability = False):
        def split_idx_generator(gstep, gmax):
            g = 0
            while g <= gmax:
                yield g
                g += gstep
        x_dim, y_dim = img.shape[:2]
        x_spacing, y_spacing = x_dim/num_x_dim, y_dim/num_y_dim
        if not ignore_split_incompatability and (x_spacing != int(x_spacing) or y_spacing != int(y_spacing)):
            if x_spacing != int(x_spacing):
                raise ValueError(f"x_dim={x_dim} and num_x_splits={num_x_splits} does not divide evenly ({x_spacing:.3f}).")
            else:
                raise ValueError(f"y_dim={y_dim} and num_y_splits={y_spacing} does not divide evenly ({y_spacing:.3f}).")
        list_X, list_Y = split_idx_generator(x_spacing), split_idx_generator(y_spacing)
        return list_X, list_Y

    def exact_split(self, list_X, list_Y):
        """
                e.g. list_X=[0,10,15]; list_Y=[0,5,10] will produce images

                        [ [ (0,0),(10,5)], [ (0,6),(10,10)],
                          [(10,0),(15,5)], [(10,6),(15,10)]  ]
        """
        it_X = len(list_X) - 1
        it_Y = len(list_Y) - 1
        # images = [None]*(it_X*it_Y)
        # c = 0
        # for i in range(it_X):
        #     for j in range(it_Y):
        #         images[c] = self.img[list_X[i]:list_X[i+1]+1,
        #                              list_Y[j]:list_Y[j+1]+1]
        #         c +=1 
        # images = Tensor(images)
        return Tensor(
                  [self.img[list_X[i]:list_X[i+1]+1,
                            list_Y[j]:list_Y[j+1]+1]
                    for i in range(it_X) for y in range(it_Y)]
        )


class ThoracicDataset(Dataset):
    def __init__(self, summary_csv, root_dir, transform=None, pre_split=True):
        self.summary_df = pd.read_csv(path.join(root_dir, summary_csv))
        self.root_dir = root_dir
        self.transform = transform
        self.summary_csv = summary_csv
        self.pre_split = pre_split
        self.split_list_X = None
        self.split_list_Y = None
        self.num_patches = None
        self.shape = None

    def _register_X_split(self, list_X):
        self.split_list_X = list_X

    def _register_Y_split(self, list_Y):
        self.split_list_Y = list_Y

    def register_splits(self, list_X, list_Y):
        self._register_X_split(list_X)
        self._register_Y_split(list_Y)

    @property
    def number_of_patches(self):
        if not self.num_patches:
            if self.split_list_X and self.split_list_Y:
                self.num_patches = (len(self.split_list_X)-1)*(len(self.split_list_Y)-1)
            else:
                self.num_patches = 1
                # raise ValueError("Must register splits before number of splits can be determined.")
        return self.num_patches

    @property
    def shape(self):
        """
            will be tile shape if pre_split is True
        """
        if not self.ret_shape:
            self.shape = (*self[0][-2:])
        return self.shape
    
    def __len__(self):
        return len(self.summary_df)

    def __getitem__(self, idx):
        if torch.is_tesnor(idx):
            idx = idx.tolist()
        img_name = path.join(self.root_dir, self.summary_df.iloc[idx,0])
        img = io.imread(img_name)

        if self.transform: 
            img = self.transform(img)

        class_name = self.summary_df[idx,1]
        class_id   = self.summary_df[idx,2]
        rad_id     = self.summary_df[idx,3]
        x_min = self.summary_df[idx,4]
        y_min = self.summary_df[idx,5]
        x_max = self.summary_df[idx,6]
        y_max = self.summary_df[idx,7]

        med_img = MedicalImage(img, class_name, class_id, rad_id, x_min, y_min, x_max, y_max)
        res = med_img if not pre_split else med_img.exact_split(self.split_list_X, self.split_list_Y)

        return Tensor(res)




