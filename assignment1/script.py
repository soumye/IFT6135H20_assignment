from solution import *
import argparse

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('-activation',default='relu')
   parser.add_argument('-lr',type=float,default=7e-4)
   parser.add_argument('-init_method',default='glorot')
   parser.add_argument('-epochs',type=int,default=10)
   parser.add_argument('-h1',default=784,type=int)
   parser.add_argument('-h2',default=256,type=int)
   return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    nn=NN(activation=args.activation,
        lr=args.lr,
        init_method=args.init_method,
        data=load_mnist(),
        hidden_dims=(args.h1, args.h2)
        )
    logs = nn.train_loop(args.epochs)
    import ipdb; ipdb.set_trace()

