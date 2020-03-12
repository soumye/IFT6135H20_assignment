from solution import *
import argparse
import matplotlib.pyplot as plt
def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('-activation',default='relu')
   parser.add_argument('-lr',type=float,default=7e-4)
   parser.add_argument('-init_method',default='glorot')
   parser.add_argument('-epochs',type=int,default=1)
   parser.add_argument('-h1',default=784,type=int)
   parser.add_argument('-h2',default=256,type=int)
   parser.add_argument('-n',default=10, type=int)
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
    N = [10**i for i in range(5)] + [3*(10**i) for i in range(5)]+ [5*(10**i) for i in range(5)]
    N = sorted(N)
    divs = []
    for n in N:
        mx = 0.0
        for i in range(10):
            new = nn.num_grad(i,n)
            mx = max(mx, new)
            print('#',new)
        #print('max diff for n = {}'.format(n), mx)
        divs.append(mx)
    plt.plot(N, divs, linestyle='--', marker='o', color='b')
    plt.xlabel('N(Logscale)')
    plt.xscale('log')
    plt.ylabel('Max Abs Diff of Gradients')
    plt.tight_layout
    plt.savefig('grad.png')
    import ipdb; ipdb.set_trace()

