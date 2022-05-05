from model import CPM2DPose
from model_prob4 import DIFFER_STB_CPMAllPose_213output
import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

from glob import glob
import pandas

#추가
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

device = 'cuda:0'
num_joints = 21

class ObmanDataset(Dataset):
    def __init__(self, method=None):
        #self.root = '/data/TA_Test1000/' #Change this path
        #self.root = '/hdd_ext/hdd3000/AItoolkit/ass4/Obman_dataset/'
        #self.root='Obman_dataset/'
        #self.root ='/hdd_ext/hdd3000/OBMAN/obman/'
        self.root = 'Obman_dataset/'
        self.x_data = []
        self.y_data = []
        if method == 'train':
            self.root = self.root + 'train/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'),key=lambda x:int(x.split('.')[0].split('/')[-1]))

        elif method == 'test':
            self.root = self.root + 'test/'
            #self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'),key=lambda x:int(x.split('.')[0].split('/')[-1]))

        for i in tqdm.tqdm(range(len(self.img_path))):
            img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
            #print(self.img_path[i]) 0,1,2,3,4,5,6.....
            b, g, r = cv2.split(img) # 3x256x256 -> 채널별 분리
            img = cv2.merge([r, g, b])
            self.x_data.append(img)

            num = self.img_path[i].split('.')[0].split('/')[-1]
            img_pkl = self.root + 'meta/' + str(num) + '.pkl'
            pkl = pandas.read_pickle(img_pkl)
            coords_2d = pkl['coords_2d']
            self.y_data.append(coords_2d)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])

        return new_x_data, self.y_data[idx]


class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()

        dataset = ObmanDataset(method='train')
        self.test_dataset = ObmanDataset(method='test')
        
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Load of pretrained_weight file
        weight_root = self.root.split('/')
        del weight_root[-2]
        weight_root = "/".join(weight_root)
        #weight_PATH = weight_root + 'pretrained_model/pretrained_weight.pth'
        #weight_PATH = weight_root+'pretrained_model/_210625_1644_22_model.pth'
        #weight_PATH = '/hdd_ext/hdd3000/AItoolkit/ass4/Obman_dataset/pretrained_model/_210625_1644_22_model.pth'
        #weight_PATH = weight_root+'pretrained_model/_210625_1712_60_model.pth'
        #self.poseNet.load_state_dict(torch.load(weight_PATH))

        ## 추가
        self.optimizer = optim.Adam(params=self.poseNet.parameters(),lr=self.learning_rate)
        #self.loss = self.calcerror()
        self.loss = nn.MSELoss()
        print("Training...")

    def _build_model(self):
        # # 2d pose estimator
        # poseNet = CPM2DPose()
        
        poseNet = DIFFER_STB_CPMAllPose_213output()
        self.poseNet = poseNet.to(device)
        self.poseNet.train()

        print('Finish build model.')

    def skeleton2heatmap(self, _heatmap, keypoint_targets):
        heatmap_gt = torch.zeros_like(_heatmap, device=_heatmap.device)

        keypoint_targets = (((keypoint_targets)) // 8)
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                x = int(keypoint_targets[i, j, 0])
                y = int(keypoint_targets[i, j, 1])
                heatmap_gt[i, j, x, y] = 1

        heatmap_gt = heatmap_gt.detach().cpu().numpy()
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                heatmap_gt[i, j, :, :] = cv2.GaussianBlur(heatmap_gt[i, j, :, :], ksize=(3, 3), sigmaX=2, sigmaY=2) * 9 / 1.1772
        heatmap_gt = torch.FloatTensor(heatmap_gt).to(device)
        return heatmap_gt

    # def calc_error(self, gt,pred):
    #     ## heatmap끼리 MSEloss주는것 만들기
    #     pass
    
    def heatmap2skeleton(self, heatmapsPoseNet):
        skeletons = np.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2))
        for m in range(heatmapsPoseNet.shape[0]):
            for i in range(heatmapsPoseNet.shape[1]):
                u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i]), (32, 32))
                skeletons[m, i, 0] = u * 8
                skeletons[m, i, 1] = v * 8
        return skeletons

    def calc_error(self, gt,pred):
    # gt : 1x21x2    np.ndarray
    # pred : 1x21x2  torch.tensor
        pred = np.array(pred.squeeze(0),dtype=np.float64)
        gt = gt.squeeze(0)
        assert gt.shape[0]==21
        assert pred.shape[0]==21
        j_sum = 0
        for i in range(21):
            gt_x = gt[i][0]; gt_y = gt[i][1]
            pred_x = pred[i][0]; pred_y = pred[i][1]
            x_diff = (gt_x-pred_x)**2
            y_diff = (gt_y-pred_y)**2
            j_sum += np.sqrt(x_diff+y_diff)
        j_avg = j_sum/21
        return j_avg

    def train(self):
        date = '210625_1935'
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 2 == 0:
                torch.save(self.poseNet.state_dict(), "_".join([self.root, date, str(epoch), 'model.pth']))

            self.poseNet.train()
            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = samples
                heatmapsPoseNet = self.poseNet(x_train.cuda())
                gt_heatmap = self.skeleton2heatmap(heatmapsPoseNet[0], y_train)
                
                # loss
                loss = self.loss(heatmapsPoseNet[0],gt_heatmap)
                # optim
                self.optimizer.zero_grad() # 기울기 0으로 초기화
                loss.backward() # 기울기계산
                self.optimizer.step() #backward
                
                ## Write train result
                if batch_idx % 20 == 0:
                    with open('train_result_' + date + '.txt', 'a') as f:
                        f.write('Epoch {:4d}/{} Batch {}/{} Loss {}\n'.format(
                            epoch, self.epochs, batch_idx, len(self.dataloader),loss
                        ))
                    print('Epoch {:4d}/{} Batch {}/{} Loss {}'.format(
                        epoch, self.epochs, batch_idx, len(self.dataloader),loss
                    ))
                #break
            
            ### TEST
            self.poseNet.eval()
            # test data 106번에 대해서 error확인
            #test_dataset = ObmanDataset(method='test')
            #test_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            x_train,y_train = self.test_dataset[106] # self.test_dataset[106]
            x_train = torch.Tensor(x_train).unsqueeze(0); y_train = np.expand_dims(y_train,0)
            heatmapsPoseNet = self.poseNet(x_train.cuda())[0].cpu().detach().numpy()
            pred = self.heatmap2skeleton(heatmapsPoseNet)
            error = self.calc_error(y_train,pred) # 원래, loss:11.913603782653809
            print('error:{}'.format(error))
        print('Finish training.')
    
    


class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._build_model()

        dataset = ObmanDataset(method='test')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        # Load of pretrained_weight file
        weight_root = self.root.split('/')
        del weight_root[-2]
        weight_root = "/".join(weight_root)
        # weight_PATH = weight_root+'pretrained_model/pretrained_weight.pth'
        #weight_PATH = weight_root+'train/_210623_26_model.pth'
        #weight_PATH = weight_root+'pretrained_model/_210625_1712_60_model.pth'
        #weight_PATH = '/hdd_ext/hdd3000/AItoolkit/ass4/Obman_dataset/pretrained_model/_210625_1712_60_model.pth'
        #weight_PATH = weight_root+'pretrained_model/2021_4_6_31_pose_model.pth'
        weight_PATH = weight_root+'pretrained_model/weight_prob4.pth'
        self.poseNet.load_state_dict(torch.load(weight_PATH)) # strict=False:일치하지않는키를무시

        print("Testing...")

    def _build_model(self):
        # 2d pose estimator
        poseNet = DIFFER_STB_CPMAllPose_213output()
        self.poseNet = poseNet.to(device)
        self.poseNet.eval()
        print('Finish build model.')

    def heatmap2skeleton(self, heatmapsPoseNet):
        ### 1-1)
        ## heatmap출력: heatmapPoseNet: [1,21,32,32] 
        # heatmaps = np.squeeze(heatmapsPoseNet,0)
        # fig = plt.figure()
        # for i in range(21):
        #     heatmap = heatmaps[i]
        #     heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min()) # min-max norm
        #     heatmap = heatmap*255
        #     ax = fig.add_subplot(3,7,i+1) # 3x7 grid로 plot
        #     ax.set_title(i)
        #     ax.axis('off')
        #     ax.imshow(heatmap)
        # plt.show()
        # plt.savefig('1-1_.png') # plt.show()지워야 제대로 뜸
        # # exit()
        skeletons = np.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2))
        for m in range(heatmapsPoseNet.shape[0]):
            for i in range(heatmapsPoseNet.shape[1]):
                u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i]), (32, 32))
                skeletons[m, i, 0] = u * 8
                skeletons[m, i, 1] = v * 8
        return skeletons
    
    def calc_error(self, gt,pred):
    # gt : 1x21x2    np.ndarray
    # pred : 1x21x2  torch.tensor
        pred = np.array(pred.squeeze(0),dtype=np.float64)
        gt = gt.squeeze(0)
        assert gt.shape[0]==21
        assert pred.shape[0]==21
        j_sum = 0
        for i in range(21):
            gt_x = gt[i][0]; gt_y = gt[i][1]
            pred_x = pred[i][0]; pred_y = pred[i][1]
            x_diff = (gt_x-pred_x)**2
            y_diff = (gt_y-pred_y)**2
            j_sum += np.sqrt(x_diff+y_diff)
        j_avg = j_sum/21
        return j_avg
            
    def test(self):
        total_loss = 0
        for batch_idx, samples in enumerate(self.dataloader):
            ## 106번에 대해서만 수행
            # if batch_idx==106:
                x_test, y_test = samples
                heatmapsPoseNet = self.poseNet(x_test.cuda())[0].cpu().detach().numpy()
                skeletons_in = self.heatmap2skeleton(heatmapsPoseNet)
                ### 1-2)
                # plot_hand(x_test.squeeze(0).permute(1,2,0),skeletons_in.squeeze(0))
                # plot_hand(x_test.squeeze(0).permute(1,2,0),y_test.squeeze(0))
                # plt.show()
                #exit()
                loss = self.calc_error(y_test,skeletons_in)
                print('loss:{}'.format(loss))
                #exit()
                total_loss += loss
        total_loss_avg = total_loss/self.datalen
        print('total_loss_avg:{}'.format(total_loss_avg))
        #exit()


# def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
#     """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
#         <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""
#     links=[(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
#                         (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
#     print("Implement this.")

def plot_hand(img,joints):
    """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
        <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""
    """
    input : [21,2]
    """
    links=[(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                        (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    x = joints[:, 0]
    y = joints[:, 1]
    scatter =False
    if scatter:
        ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    joint_idxs = True
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=1, linewidth=2)
    ax.axis('equal')

def _draw2djoints(ax, annots, links, alpha=1, linewidth=1):
    #colors = ['r', 'm', 'b', 'c', 'g']
    colors = [(0,0,1),(1,0,1),(1,0,0),(0,1,0),(1,0.5,0)]
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth)

def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1, linewidth=1):
    ax.plot([annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
            c=c,
            alpha=alpha,
            linewidth=linewidth)


    

def main():

    epochs = 60
    #batchSize = 16
    batchSize = 1
    #batchSize = 32
    learningRate = 1e-4
    

    # trainer = Trainer(epochs, batchSize, learningRate)
    # trainer.train()

    tester = Tester(batchSize)
    tester.test()


if __name__ == '__main__':
    #x = torch.load('/hdd_ext/hdd3000/AItoolkit/ass4/Obman_dataset/pretrained_model/2021_4_6_31_pose_model.pth')
    main()
