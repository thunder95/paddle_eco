import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from eco_model import ECOFull
from hmdb_reader import KineticsReader
from cfg_config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='eco_full',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/aistudio/work/hmdb_config.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        val_config =  merge_configs(config, 'valid', vars(args))
        # print_configs(train_config, 'Train')

        train_model = ECOFull(train_config['MODEL']['name'], 
            train_config['MODEL']['seg_num'], 
            train_config['MODEL']['num_classes'])

        #opt = fluid.optimizer.Momentum(0.005, 0.9, parameter_list=train_model.parameters()， regularization=)
        # opt = fluid.optimizer.SGD(0.001, parameter_list=train_model.parameters())
        # opt = fluid.optimizer.SGD(learning_rate=fluid.dygraph.ExponentialDecay(
        #       learning_rate=0.001,
        #       decay_steps=5000,
        #       decay_rate=0.1,
        #       staircase=True), 
        #       regularization=fluid.regularizer.L2Decay(regularization_coeff=1e-4),
        #       parameter_list=train_model.parameters())
        
        opt = fluid.optimizer.SGD(learning_rate=fluid.dygraph.CosineDecay(0.0008, 550, 1), 
              regularization=fluid.regularizer.L2Decay(regularization_coeff=4e-4),
              parameter_list=train_model.parameters())

        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            print("loaded pretrain_model: ", args.save_dir + '/best_model.pdparams')
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/best_model.pdparams')
            train_model.load_dict(model)

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()
        val_reader = KineticsReader(args.model_name.upper(),'valid', val_config).create_reader()

        epochs = args.epoch or train_model.epoch_num()
        max_acc = 0
        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                # print("xshape:", dy_x_data.shape)
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                # print("yshape:", y_data.shape)
                # print(y_data)

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                
                out, acc = train_model(img, label)
                

                # print(out.numpy(), label.numpy())
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()
                
                
                if batch_id % 50 == 0:
                    logger.info("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    
            #验证集
            train_model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(val_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                out, acc = train_model(img, label)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                accuracies.append(acc.numpy())
                losses.append(avg_loss.numpy())

            mean_acc = np.mean(accuracies)
            if mean_acc > max_acc:
                max_acc = mean_acc
                fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/best_model')

            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            train_model.train()

        fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/final_model')
        logger.info("Final loss: {}".format(avg_loss.numpy()))
        print("Final loss: {}".format(avg_loss.numpy()))
                

if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
