import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score, \
    recall_score, precision_score, average_precision_score

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--k', '--factors', default=32, type=int,
                    metavar='k', help='dim of items', dest='k')
parser.add_argument('--epoch', '--epoch number', default=100, type=int,
                    metavar='epoch', help='epoch', dest='epoch')
parser.add_argument('--lr1', nargs='?', default=0.0001,
                    help='lr for id embeddings')
parser.add_argument('--wd1', nargs='?', default=0.0001,
                    help='reg for id embeddings')
parser.add_argument('--lr2', nargs='?', default=0.0001,
                    help='lr2 for id embeddings')
parser.add_argument('--wd2', nargs='?', default=0.0001,
                    help='reg2 for id embeddings')


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
    def auc_calculate(labels, pre_ds, n_bins=100):
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
        positive_samples = torch.einsum('ij, ij->i', [labels, pre_ds])
        negative_samples = torch.einsum('ij, ij->i', [torch.abs[labels - 1], pre_ds])
        positive_results = positive_samples / positive_lens
        negative_results = negative_samples / negative_lens
        results = torch.mean((negative_results + positive_results) / 2.0)
        results = results.cpu()
        return np.array(results)

    def compute_scores(self, gt, preds):
        ret = {
            'auc': self.auc_calculate(gt, preds),
            'ndcg': Metric.ndcg(gt, preds)

        }
        return ret

    def __logscore(self, scores):
        metrics = list(scores.keys())
        metrics.sort()
        print(' '.join(['%s: %s' % (m, str(scores[m])) for m in metrics]))
        # self.logging.info(' '.join(['%s: %s' % (m,str(scores[m])) for m in metrics]))

    def test(self):
        print('----- test -----begin-----')
        u = torch.LongTensor(range(self.user_size))
        u = u.cuda()
        test_arr = self.test_samples
        test_tensor = torch.from_numpy(test_arr)
        test_tensor = test_tensor.cuda()
        results = self.compute_results(u, test_tensor)
        scores = self.compute_scores(self.test_gt, results)
        self.__logscore(scores)
        print('----- test -----end-----')

    def val(self):
        print('----- val -----begin-----')
        u = torch.LongTensor(range(self.user_size))
        results = self.compute_results(u, self.val_samples)
        scores = self.compute_scores(self.val_gt, results)
        self.__logscore(scores)
        print('----- val -----end-----')

    def test_warm(self):
        print('----- test_warm -----begin-----')
        u = self.test_warm_u
        u = u.cuda()
        results = self.compute_results(u, self.test_warm_samples)
        scores = self.compute_scores(self.test_warm_gt, results)
        self.__logscore(scores)
        print('----- test_warm -----end-----')

    def test_cold(self):
        print('----- test_cold -----begin-----')
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
        K = [5, 10, 20, 50, 100, 150, 200]
        return [ndcg_score(gt, preds, k=k) for k in K]  # 看这个地方具体怎么写的 修改这个地方的具体实现

    @staticmethod
    def auc(gt, preds):
        print('.auc')
        gt = torch.from_numpy(gt)
        # gt = gt.cuda()
        preds = torch.from_numpy(preds)
        # preds = preds.cuda()
        return roc_auc_score(gt, preds, average='samples')  # 两个nd-array 进行auc计算


class CBPR(Model):

    def __init__(self,
                 args,
                 data):
        super(CBPR, self).__init__()

        self.k = args.k
        self.lr1 = args.lr1
        self.wd1 = args.wd1
        self.lr2 = args.lr2
        self.wd2 = args.wd2
        self.epochs = args.epoch
        self.user_size = len(data.user_set)
        self.item_size = len(data.item_set)
        self.feature_size = 64
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

        self.user_matrix = nn.Embedding(self.user_size, self.k)  # k default value is 32
        self.user_matrix = self.user_matrix.cuda()
        # user_matrix 可以视为用户关于某个latent factor的权重
        nn.init.normal_(self.user_matrix.weight, std=0.01)  # 进行正则化操作

        self.feature_matrix = torch.randn(self.feature_size, self.k, dtype=torch.float32) * 0.01  # k default 64
        self.feature_matrix = self.feature_matrix.cuda()
        # item_matrix 可以视为item在某个latent factor的值的大小

    def predict(self, uid, iid):
        """
        uid of user_matrix
        iid of item_matrix
        :return:
        """
        p1 = self.user_matrix(uid)
        p2 = torch.mm(self.new_feature[iid], self.feature_matrix)
        return torch.sum(p1 * p2, dim=1)

    def bpr_loss(self, uid, iid, jid):
        """
        bpr的算法是，对一个用户u求i和j两个item的分数，然后比较更喜欢哪个，
        所以这里需要进行两次预测，分别是第i个item的和第j个item的
        """
        pre_i = self.predict(uid, iid)
        pre_j = self.predict(uid, jid)
        dev = pre_i - pre_j
        return torch.sum(fun.softplus(-dev))

    def train(self):
        """
        lr: learning rate default value is 0.01
        :return:
        """
        optimizer = torch.optim.Adagrad([self.user_matrix.weight], lr=self.lr1, weight_decay=self.wd1)
        optimizer2 = torch.optim.Adam([self.feature_matrix], lr=self.lr2, weight_decay=self.wd2)
        epochs = self.epochs
        for epoch in tqdm(range(epochs)):
            generator = self.sample()
            while True:
                optimizer.zero_grad()
                optimizer2.zero_grad()
                s = next(generator)
                if s is None:
                    break

                uid, iid, jid = s[:, 0], s[:, 1], s[:, 2]
                uid = uid.cuda()
                iid = iid.cuda()
                jid = jid.cuda()
                loss = self.bpr_loss(uid, iid, jid)

                loss.backward()
                optimizer.step()
                optimizer2.step()
            if epoch % 5 == 0 and epoch > 1:
                print(f'epoch is {epoch}')
                self.val(), self.test(), self.test_warm(), self.test_cold()

        self.test(), self.test_warm(), self.test_cold()

    def sample(self):
        np.random.shuffle(self.train_list)
        for i in range(self.sz // self.batch_size):
            pairs = []
            sub_train_list = self.train_list[i * self.batch_size:(i + 1) * self.batch_size, :]
            for m, j in sub_train_list:
                m_neg = j
                while m_neg in self.train_pt[m]:
                    m_neg = np.random.randint(self.item_size)
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
    cb_pr = CBPR(args,
                 data
                 )
    print('cb_pr is ready')
    cb_pr = cb_pr.cuda()
    cb_pr.train()


if __name__ == '__main__':
    main()
