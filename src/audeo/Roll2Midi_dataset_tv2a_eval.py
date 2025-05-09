import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import glob
print(torch.cuda.current_device())
DEFAULT_DEVICE = 'cuda'

torch.cuda.set_device(0)

frames = 50 #2 seconds

min_key = 15
max_key = 65

class Roll2MidiDataset(Dataset):
    def __init__(self, path='/ailab-train/speech/shansizhe/audeo/data/tv2a_piano3_4000_pkl_npz/gt/npz/', est_roll_path='/ailab-train/speech/shansizhe/audeo/data/tv2a_piano3_4000_pkl_npz/v2a/npz/',
                    train=True,  device=DEFAULT_DEVICE):
        self.path = path
        self.est_roll_path = est_roll_path
        self.device = device
        self.train = train
        self.load_data()
    def __getitem__(self, index):
        if self.train:
            gt, roll = self.final_data['train'][index]
        else:
            gt, roll = self.final_data['test'][index]
        gt_ = gt.T.float().to(self.device)
        roll_ = roll.T.float().to(self.device)
        return torch.unsqueeze(gt_, dim=0), torch.unsqueeze(roll_, dim=0)

    def __len__(self):
        if self.train:
            return len(self.final_data['train'])
        else:
            return len(self.final_data['test'])

    def load_data(self):
        self.files = []
        self.labels = []

        # ground truth midi dir
        path = self.path
        #print(path)
        train_gt_folders = glob.glob(path + '/*')
        train_gt_folders.sort(key=lambda x: x.split('/')[-1].split('__')[-1])
        print(train_gt_folders)


        # Roll predictions dir
        train_roll_folder = glob.glob(self.est_roll_path + '/*')
        train_roll_folder.sort(key=lambda x: x.split('/')[-1].split('__')[-1])
        print(train_roll_folder)

        # self.folders: dictionary
        # key: train/test, values: list of tuples [(ground truth midi folder name, roll prediction folder name)]
        self.folders = {}
        self.folders['train'] = [(train_gt_folders[i], train_roll_folder[i]) for i in range(len(train_gt_folders))]
        print(self.folders['train'])

        # self.data: dictionary
        # key: train/test, value:list of tuples [(2 sec ground truth Midi, 2 sec Roll prediction logits)]
        self.data = {}
        self.data['train'] = []

        # self.final_data: similar to the data, but concat two continuous 2 sec Roll prediction (4 seconds, 100 frames)
        self.final_data = {}
        self.final_data['train'] = []

        # load training data
        for train_gt_folder, est_roll_folder in self.folders['train']:
            gt_files = glob.glob(train_gt_folder + '/*.npz')
            gt_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[0]))
            est_roll_files = glob.glob(est_roll_folder + '/*.npz')
            est_roll_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[0]))
            print("have the same files of training gt and est roll:", len(gt_files) == len(est_roll_files))
            for i in range(len(gt_files)):
                with np.load(gt_files[i]) as data:
                    gt = data['midi'][:, min_key:max_key + 1]
                    if gt.shape[0] != frames:
                        target = np.zeros((frames, max_key-min_key+1))
                        target[:gt.shape[0], :] = gt
                        gt = target
                    gt = np.where(gt > 0, 1, 0)
                with np.load(est_roll_files[i]) as data:
                    est_roll_logit = data['midi'][:, min_key:max_key + 1]
                    if est_roll_logit.shape[0] != frames:
                        target = np.zeros((frames, max_key-min_key+1))
                        target[:est_roll_logit.shape[0], :] = est_roll_logit
                        est_roll_logit = target
                    est_roll_logit = np.where(est_roll_logit > 0, 1, 0)
                self.data['train'].append((torch.from_numpy(gt), torch.from_numpy(est_roll_logit)))
        # make 4 sec data
        for i in range(len(self.data['train'])):
            if i + 1 < len(self.data['train']):
                one_gt, one_roll = self.data['train'][i]
                two_gt, two_roll = self.data['train'][i + 1]
                final_gt = torch.cat([one_gt, two_gt], dim=0)
                final_roll = torch.cat([one_roll, two_roll], dim=0)
                self.final_data['train'].append((final_gt, final_roll))

        print("total number of training data:", len(self.final_data['train']))

if __name__ == "__main__":
    dataset = Roll2MidiDataset()
    gt,midi = dataset.__getitem__(0)
    print(gt.shape)
    print(midi.shape)
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
    ax1.imshow(gt.cpu().numpy().squeeze(), plt.cm.gray)
    ax2.imshow(midi.cpu().numpy().squeeze(), plt.cm.gray)
    plt.show()
    data_loader = DataLoader(dataset, batch_size=64)
    for i,data in enumerate(data_loader):
        gts,midis = data
        break

