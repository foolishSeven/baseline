import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score
import torch.optim as optim
import os

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--k', '--factors', default=64, type=int,
                    metavar='k', help='dim of items', dest='k')
parser.add_argument('--epoch', '--epoch number', default=100, type=int,
                    metavar='epoch', help='epoch', dest='epoch')
parser.add_argument('--lr1', '--learning-rate1', default=0.001, type=float,
                    metavar='LR1', help='initial learning rate', dest='lr1')
parser.add_argument('--reg1', '--regularization1', default=0, type=float,
                    metavar='reg1', help='regularization1', dest='reg1')
parser.add_argument('--lr2', '--learning-rate2', default=0.1, type=float,
                    metavar='LR2', help='initial learning rate', dest='lr2')
parser.add_argument('--reg2', '--regularization2', default=0.1, type=float,
                    metavar='reg2', help='regularization2', dest='reg2')
parser.add_argument('--lr3', '--learning-rate3', default=0.001, type=float,
                    metavar='LR3', help='initial learning rate', dest='lr3')
parser.add_argument('--reg3', '--regularization3', default=0.01, type=float,
                    metavar='reg3', help='regularization1', dest='reg3')


class Model(nn.Module):

    def compute_results(self, u, test_samples):
        u = u.cuda()
        if type(test_samples) != torch.Tensor:
            test_samples = torch.from_numpy(test_samples)
        test_samples = test_samples.cuda()
        rs = []
        for i in test_samples.T:
            # lt = torch.LongTensor(i)
            lt = i
            res_temp = self.predict(u, lt).detach()
            res_temp = res_temp.cpu()
            res_temp = res_temp.numpy()
            rs.append(res_temp)
        results = np.vstack(rs).T
        if np.isnan(results).any():
            raise Exception('nan')
        return results

    @staticmethod
    def auc_calculate(labels, pre_ds):
        print('.auc')
        labels = torch.from_numpy(labels)
        labels = labels.cuda()
        result_tensor = torch.Tensor(len(labels))
        pre_ds = torch.from_numpy(pre_ds)
        pre_ds = pre_ds.cuda()
        # 计算tp，阈值设置为0.0，即预测集中，值大于0的样例设置为1，小于0的样例 设置为0
        one = torch.ones_like(pre_ds)
        zero = torch.zeros_like(pre_ds)
        boundary = torch.mean(pre_ds)
        pre_ds = torch.where(pre_ds > boundary, one, zero)
        positive_lens = torch.sum(labels, dim=1)
        negative_lens = torch.abs(torch.sum(labels - 1, dim=1))

        positive_samples = torch.einsum('ij, ij->i', [labels.double(), pre_ds.double()])
        negative_samples = torch.einsum('ij, ij->i', [torch.abs(labels.double() - 1), pre_ds.double()])
        positive_results = positive_samples / positive_lens
        negative_results = negative_samples / negative_lens
        results = torch.mean((negative_results + positive_results) / 2.0)
        results = results.cpu()
        return np.array(results)

    def compute_scores(self, gt, preds):
        ret = {
            'auc':  self.auc_calculate(gt, preds),
            'ndcg': Metric.ndcg(gt, preds)

        }
        return ret

    def __logscore(self, scores):
        metrics = list(scores.keys())
        metrics.sort()
        print(' '.join(['%s: %s' % (m, str(scores[m])) for m in metrics]), file=self.file)
        # self.logging.info(' '.join(['%s: %s' % (m,str(scores[m])) for m in metrics]))

    def test(self):
        print('----- test -----', file=self.file)
        u = torch.LongTensor(range(self.usz))
        u = u.cuda()
        test_arr = self.test_samples
        test_tensor = torch.from_numpy(test_arr)
        test_tensor = test_tensor.cuda()
        results = self.compute_results(u, test_tensor)
        scores = self.compute_scores(self.test_gt, results)
        self.__logscore(scores)
        print('----- test -----end-----')

    def val(self):
        print('----- val -----', file=self.file)
        u = torch.LongTensor(range(self.usz))
        results = self.compute_results(u, self.val_samples)
        scores = self.compute_scores(self.val_gt, results)
        self.__logscore(scores)
        print('----- val -----end-----')

    def test_warm(self):
        print('----- test_warm -----', file=self.file)
        u = self.test_warm_u
        u = u.cuda()
        results = self.compute_results(u, self.test_warm_samples)
        scores = self.compute_scores(self.test_warm_gt, results)
        self.__logscore(scores)
        print('----- test_warm -----end-----')

    def test_cold(self):
        print('----- test_cold -----', file=self.file)
        u = self.test_cold_u
        u = u.cuda()
        results = self.compute_results(u, self.test_cold_samples)
        scores = self.compute_scores(self.test_cold_gt, results)
        self.__logscore(scores)
        print('----- test_cold -----end-----')

    def train(self):
        raise Exception('no implementation')

    def regs(self):
        raise Exception('no implementation')

    def predict(self):
        raise Exception('no implementation')

    def save(self):
        raise Exception('no implementation')


class Metric:
    @staticmethod
    def get_annos(gt, preds):
        p_num = np.sum(gt > 0, axis=1, keepdims=True).flatten()
        # print(p_num)
        pos = np.argsort(-preds)[range(len(p_num)), p_num]
        # print(pos)
        ref_score = preds[range(len(pos)), pos]
        # print(preds.T, ref_score)
        annos = (preds.T - ref_score).T > 0
        return annos

    @staticmethod
    def ndcg(gt, preds):
        print('.ndcg')
        gt = torch.from_numpy(gt)
        preds = torch.from_numpy(preds)
        K = [5, 10, 20]
        return [ndcg_score(gt, preds, k=k) for k in K]  # 看这个地方具体怎么写的 修改这个地方的具体实现

    def auc(gt, preds):
        print('.auc')
        return roc_auc_score(gt, preds, average='samples')


class MTPR(Model):

    def __init__(self, args, data, filename):
        super(MTPR, self).__init__()
        self.args = args
        self.dim = 32
        self.epochs = args.epoch
        self.filename = filename
        self.file = open(filename, 'a')
        self.usz = len(data.user_set)
        self.isz = len(data.item_set)
        self.fsz = 64
        self.train_pt = data.train_pt
        self.train_list = data.train_list
        self.sz = self.train_list.shape[0]
        self.batch_size = 512  # 这个参数应该从args里面获取啊
        self.test_cold_gt = data.test_cold_gt
        self.test_cold_samples = data.test_cold_samples
        self.test_cold_u = data.test_cold_u
        self.test_samples = data.test_samples
        self.test_warm_gt = data.test_warm_gt
        self.test_warm_samples = data.test_warm_samples
        self.test_warm_u = data.test_warm_u
        self.val_cold_gt = data.val_cold_gt
        self.val_cold_samples = data.val_cold_samples
        self.val_cold_u = data.val_cold_u
        self.val_warm_gt = data.val_warm_gt
        self.val_warm_samples = data.val_warm_samples
        self.val_warm_u = data.val_warm_u
        self.val_samples = data.val_samples
        self.val_gt = data.val_gt
        self.test_gt = data.test_gt
        self.new_feature = torch.Tensor(data.item_set)
        self.new_feature = self.new_feature.cuda()

        p1_weight = np.random.randn(self.usz, self.dim) * 0.01
        q_weight = np.random.randn(self.isz, self.dim) * 0.01

        p2_weight = np.random.randn(self.usz, self.dim) * 0.01
        p_weight = np.concatenate([p1_weight, p2_weight], axis=1)

        self.P = torch.nn.Embedding(self.usz, self.dim * 2)
        self.P.weight.data.copy_(torch.tensor(p_weight))
        self.P = self.P.cuda()

        self.Q = torch.nn.Embedding(self.isz, self.dim)
        self.Q.weight.data.copy_(torch.tensor(q_weight))
        self.Q = self.Q.cuda()

        self.W = torch.randn(self.fsz, self.dim, dtype=torch.float32) * 0.01
        self.W = self.W.cuda()
        self.weu = torch.randn(self.dim * 2, self.dim, dtype=torch.float32) * 0.01
        self.weu = self.weu.cuda()
        self.wei = torch.randn(self.dim * 2, self.dim, dtype=torch.float32) * 0.01
        self.wei = self.wei.cuda()

    def fimg(self, iid):  # normal representation
        return torch.cat((self.Q(iid), torch.mm(self.new_feature[iid], self.W)), dim=1)

    def zimg(self, iid):  # counterfactual representation
        fzero = torch.zeros_like(self.Q(iid))
        return torch.cat((fzero, torch.mm(self.new_feature[iid], self.W)), dim=1)

    def trf(self, emb, theta):
        return torch.mm(emb, theta)

    def predict(self, uid, iid):
        return torch.sum(self.trf(self.P(uid), self.weu) * self.trf(self.fimg(iid), self.wei), dim=1)

    def predict_z(self, uid, iid):
        return torch.sum(self.trf(self.P(uid), self.weu) * self.trf(self.zimg(iid), self.wei), dim=1)

    def bpr_loss_i(self, uid, iid, niid):

        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def bpr_loss_f(self, uid, iid, niid):
        pred_p = self.predict_z(uid, iid)
        pred_n = self.predict_z(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def bpr_loss_if(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict_z(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def bpr_loss_fi(self, uid, iid, niid):
        pred_p = self.predict_z(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    # multi-task learning
    def test_loss(self, uid, iid, niid):
        aloss = 0
        aloss += self.bpr_loss_i(uid, iid, niid) + self.bpr_loss_f(uid, iid, niid)  # two crucial task
        aloss += self.bpr_loss_if(uid, iid, niid) + self.bpr_loss_fi(uid, iid, niid)  # two constraints
        return aloss

    def regs(self, uid, iid, niid):
        wd1 = self.args.reg1
        wd2 = self.args.reg2
        wd3 = self.args.reg3

        p = self.P(uid)
        q = self.Q(iid)
        qn = self.Q(niid)
        w = self.W
        weu = self.weu
        wei = self.wei
        emb_regs = torch.sum(p * p) + torch.sum(q * q) + torch.sum(qn * qn)
        ctx_regs = torch.sum(w * w) + torch.sum(weu * weu)
        proj_regs = torch.sum(wei * wei)

        return wd1 * emb_regs + wd2 * ctx_regs + wd3 * proj_regs

    def train(self):
        lr1 = self.args.lr1
        lr2 = self.args.lr2
        lr3 = self.args.lr3
        optimizer = optim.Adagrad([self.P.weight, self.Q.weight], lr=lr1, weight_decay=0)
        optimizer2 = optim.Adam([self.W, self.weu], lr=lr2, weight_decay=0)
        optimizer3 = optim.Adam([self.wei], lr=lr3, weight_decay=0)
        epochs = 100
        for epoch in tqdm(range(epochs)):
            generator = self.sample()
            while True:
                s = next(generator)
                if s is None:
                    break
                uid, iid, niid = s[:, 0], s[:, 1], s[:, 2]

                optimizer.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                uid = uid.cuda()
                iid = iid.cuda()
                niid = niid.cuda()
                aloss = self.test_loss(uid, iid, niid) + self.regs(uid, iid, niid)
                aloss.backward()

                optimizer.step()
                optimizer2.step()
                optimizer3.step()

            if epoch > 0 and epoch % 5 == 0:
                print(f'epoch is {epoch}', file=self.file)
                self.val(), self.test(), self.test_warm(), self.test_cold()

        print('final')
        self.val(), self.test(), self.test_warm(), self.test_cold()

    def save(self, filename):
        np.save(filename, [self.P.weight.data.numpy(),
                           self.Q.weight.data.numpy(),
                           self.W.data.numpy(),
                           self.weu.data.numpy(),
                           self.wei.data.numpy()])
        self.logging.info('weights are saved to ' + filename)

    # , self.train_list, self.sz, self.batch_size, self.train, self.item_size
    def sample(self):
        np.random.shuffle(self.train_list)
        for i in range(self.sz // self.batch_size):
            pairs = []
            sub_train_list = self.train_list[i * self.batch_size:(i + 1) * self.batch_size, :]
            for m, j in sub_train_list:
                m_neg = j
                while m_neg in self.train_pt[m]:
                    m_neg = np.random.randint(self.isz)
                pairs.append((m, j, m_neg))

            yield torch.LongTensor(pairs)  # this position added cuda for test
        yield None


class Dataset:
    def __init__(self, user_set,
                 item_set,
                 train_pt,
                 train_list,
                 test_cold_gt,
                 test_cold_samples,
                 test_cold_u,
                 test_samples,
                 test_warm_gt,
                 test_warm_samples,
                 test_warm_u,
                 val_cold_gt,
                 val_cold_samples,
                 val_cold_u,
                 val_warm_gt,
                 val_warm_samples,
                 val_warm_u,
                 val_samples,
                 val_gt,
                 test_gt):
        super(Dataset, self).__init__()
        self.user_set = user_set
        self.item_set = item_set
        self.train_pt = train_pt
        self.train_list = train_list
        self.test_cold_gt = test_cold_gt
        self.test_cold_samples = test_cold_samples
        self.test_cold_u = test_cold_u
        self.test_samples = test_samples
        self.test_warm_gt = test_warm_gt
        self.test_warm_samples = test_warm_samples
        self.test_warm_u = test_warm_u
        self.val_cold_gt = val_cold_gt
        self.val_cold_samples = val_cold_samples
        self.val_cold_u = val_cold_u
        self.val_warm_gt = val_warm_gt
        self.val_warm_samples = val_warm_samples
        self.val_warm_u = val_warm_u
        self.val_samples = val_samples
        self.val_gt = val_gt
        self.test_gt = test_gt


def main():
    # 处理参数
    args = parser.parse_args()
    lr_params = f'lr1-{args.lr1}lr2-{args.lr2}lr3-{args.lr3}'
    reg_params = f'reg1-{args.reg1}reg2-{args.reg2}reg3-{args.reg3}'
    filename = f'mtpr-{lr_params}-{reg_params}.txt'

    if not os.path.exists(filename):
        open(filename, 'x')
    # 读取预处理的数据
    user_set = torch.load('..\\bpr\\pt\\user_set.pt')
    item_set = torch.load('..\\bpr\\pt\\item_set.pt')
    train_pt = torch.load('..\\bpr\\pt\\train.pt')
    train_list = torch.load('..\\bpr\\pt\\train_list_as_array.pt')

    test_cold_gt = torch.load('..\\bpr\\pt\\test_cold_gt.pt')
    test_cold_samples = torch.load('..\\bpr\\pt\\test_cold_samples.pt')
    test_cold_u = torch.load('..\\bpr\\pt\\test_cold_u.pt')
    test_samples = torch.load('..\\bpr\\pt\\test_samples.pt')
    test_set = torch.load('..\\bpr\\pt\\test_set.pt')
    test_warm_gt = torch.load('..\\bpr\\pt\\test_warm_gt.pt')
    test_warm_samples = torch.load('..\\bpr\\pt\\test_warm_samples.pt')
    test_warm_u = torch.load('..\\bpr\\pt\\test_warm_u.pt')
    val_cold_gt = torch.load('..\\bpr\\pt\\val_cold_gt.pt')
    val_cold_samples = torch.load('..\\bpr\\pt\\val_cold_samples.pt')
    val_cold_u = torch.load('..\\bpr\\pt\\val_cold_u.pt')

    val_set = torch.load('..\\bpr\\pt\\val_set.pt')
    val_warm_gt = torch.load('..\\bpr\\pt\\val_warm_gt.pt')
    val_warm_samples = torch.load('..\\bpr\\pt\\val_warm_samples.pt')
    val_warm_u = torch.load('..\\bpr\\pt\\val_warm_u.pt')

    val_samples = torch.load('..\\bpr\\pt\\val_samples.pt')
    val_gt = torch.load('..\\bpr\\pt\\val_gt.pt')
    test_gt = torch.load('..\\bpr\\pt\\test_gt.pt')

    data = Dataset(user_set,
                   item_set,
                   train_pt,
                   train_list,
                   test_cold_gt,
                   test_cold_samples,
                   test_cold_u,
                   test_samples,
                   test_warm_gt,
                   test_warm_samples,
                   test_warm_u,
                   val_cold_gt,
                   val_cold_samples,
                   val_cold_u,
                   val_warm_gt,
                   val_warm_samples,
                   val_warm_u,
                   val_samples,
                   val_gt,
                   test_gt)
    device = torch.device("cuda:0")
    torch.cuda.device(device)
    mtpr = MTPR(args, data, filename)
    print('mtpr is ready')
    mtpr = mtpr.cuda()
    mtpr.train()


if __name__ == '__main__':
    main()
