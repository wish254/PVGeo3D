import os
import random

from datasets.io import IOStream

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from datasets.ModelNet40_c import ModelNetC
import numpy as np
import os
import sklearn.metrics as metrics
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from models.APF import create_point_former
from datasets.scanobjectnn_c import ScanObjectNNC
from datasets.eval import eval_corrupt_wrapper

def set_fixed_seed(args, fixed=-1):
    if fixed != -1:
        args.seed = fixed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True

def test(args, io):
    import time
    # work on single sample per batch in order to adaptively select threshold per sample
    device = torch.device("cuda" if args.cuda else "cpu")

    args.test_batch_size = 1
    model = create_point_former(num_classes=40)
    model = model.to(device)
    print("---------------------------------------------------------------------------------------")
    for k, v in model.named_parameters():
        print(k, v.shape)
    print("---------------------------------------------------------------------------------------")
    model_path_critical = '/root/wish/Point_PN/checkpoint/mn40-prompt-vit-93.4/best_checkpoint.pth'
    checkpoint = torch.load(model_path_critical)
    state_dict = checkpoint['net']
    for k, v in state_dict.items():
        print(k, v.shape)
    # del state_dict['encoder_large.EncP.raw_point_embed.net.0.weight']
    model.load_state_dict(state_dict)

    io.cprint(f"load {model_path_critical}")

    model.eval()

    def test_corrupt(args, split, model):
        with torch.no_grad():
            test_loader = DataLoader(ModelNetC(args=args, split=split),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            set_fixed_seed(args, 1)
            test_true = []
            test_pred = []
            start = time.time()
            for m_iter, (data, xyz, label) in enumerate(test_loader):
                # torch.cuda.empty_cache()

                data, xyz, label = data.to(device), xyz.to(device), label.to(device)
                # print(data.shape, label.shape)
                data = data.permute(0, 2, 1)
                # rand the input, to ensure handling permutation of the points

                logits = model(data, xyz)
                preds = logits.max(dim=1)[1]

                # test_true.append(label.squeeze(-1).cpu().numpy())
                test_true.append(label.cpu().numpy())
                # test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

            end = time.time()
            import scipy.io as sio
            if split == "clean":
                io.cprint(f"Time(Total): {end - start}")
                io.cprint(f"Time(per sample): {(end - start) / m_iter}")

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

            return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}
    # eval_corrupt_wrapper_scanobjectnnc(model, test_corrupt, {'args': args}, '', epoch=0)
    eval_corrupt_wrapper(model, test_corrupt, {'args': args}, io)



if __name__ == '__main__':
        # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--output_channels', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--t_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')

    # pointwolf params
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point')
    parser.add_argument('--w_sample_type', type=str, default='fps',
                        help='Sampling method for anchor point, option : (fps, random)')
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')
    parser.add_argument('--w_R_range', type=float, default=10,
                        help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25,
                        help='Maximum translation range of local transformation')

    # rsmix params
    parser.add_argument('--rdscale', action='store_true', help='random scaling data augmentation')
    parser.add_argument('--shift', action='store_true', help='random shift data augmentation')
    parser.add_argument('--shuffle', action='store_true', help='random shuffle data augmentation')
    parser.add_argument('--rot', action='store_true', help='random rotation augmentation')
    parser.add_argument('--jitter', action='store_true', help='jitter augmentation')
    parser.add_argument('--rddrop', action='store_true', help='random point drop data augmentation')
    parser.add_argument('--rsmix_prob', type=float, default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=1.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512,
                        help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--normal', action='store_true', help='use normal')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    # useful
    parser.add_argument('--type', type=str, default='B', help='evaluate the model')
    parser.add_argument('--freeze', default=True, action='store_true', help='evaluate the model')
    parser.add_argument('--robustify', default=False, action='store_true', help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.4, help='evaluate the model')
    parser.add_argument('--use_wolfmix', default=False, action='store_true', help='if to use wm models')

    parser.add_argument('--eval', default=True, action='store_true', help='evaluate the model')
    parser.add_argument('--is_retrain', default=False, type=bool)
    parser.add_argument('--dataset', type=str, default="scan")
    parser.add_argument('--model', type=str, default='ulip', metavar='N',
                        choices=['curvenet', 'dgcnn', 'dgcnn_v1', 'dgcnn_v2', 'dgcnn_v3', 'dgcnn_v5', 'gdanet',
                                 'rpc',
                                 'pct', 'custom_model', 'pointnet', 'paconv', 'pointbert', 'pointmae', 'recon',
                                 'pointnext', 'pointmlp', 'pointpn', 'ulip'],
                        help='Model Architecture to use as a basic model for random, patches and curves')
    parser.add_argument('--exp_name', type=str, default='baseline', metavar='N', help='Name of the experiment')

    args = parser.parse_args()

    io = IOStream(
        f'{"test" if args.eval else "train"}_{"robustify" if args.robustify else ""}_{args.lr}'
        f'_{"wm" if args.use_wolfmix else ""}_{args.model}_{"yes" if args.is_retrain else "no"}_{args.dataset}'
        f'_DGCNN{args.type}.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(
                torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    test(args, io)
    # if not args.eval:
    #     # train
    #     if args.robustify:
    #         train_robustify(args, io)
    #     else:
    #         train(args, io)
    # else:
    #     if args.robustify:
    #         test_ulip_tmodel(args, io)
    #     else:
    #         test(args, io)

