import os
import glob
from .bases import BaseImageDataset

class OCC_ReID(BaseImageDataset):
    dataset_dir = 'Occluded_REID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        train_dir = os.path.join(self.dataset_dir, 'train')
        query_dir = os.path.join(self.dataset_dir, 'test/query')
        gallery_dir = os.path.join(self.dataset_dir, 'test/gallery')

        # mode 지정
        train = self._process(train_dir, mode='train')
        query = self._process(query_dir, mode='query')
        gallery = self._process(gallery_dir, mode='gallery')

        if verbose:
            print("=> Occluded-ReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # TransReID가 요구하는 데이터 속성
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process(self, folder, mode):
        """
        mode = 'train' | 'query' | 'gallery'
        query/galler는 camid를 다르게 배정해야 evaluation이 정상 작동한다.
        """
        pid_dirs = sorted(os.listdir(folder))
        pid2label = {pid: idx for idx, pid in enumerate(pid_dirs)}

        dataset = []
        for pid in pid_dirs:
            pid_path = os.path.join(folder, pid)
            if not os.path.isdir(pid_path):
                continue

            new_pid = pid2label[pid]
            img_paths = glob.glob(os.path.join(pid_path, '*.jpg'))

            for img in img_paths:
                if mode == 'train':
                    camid = 0
                elif mode == 'query':
                    camid = 0
                else:   # gallery
                    camid = 1

                dataset.append((img, new_pid, camid, 1))

        return dataset
