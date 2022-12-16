import torch
from matplotlib import pyplot as plt




def softmax_binning_concept(alpha, N, w, b, x, tau = 0.01):
    # N = number of classes
    # alpha = N 

    for i in range(N):
        for j in range(len(x)):
            alpha[i,j] = (w[i]*x[j] + b[i])/tau
        
    softmax = torch.nn.Softmax(dim=0)
    alpha = softmax(alpha)

        
    return alpha


def softmax_binning(N, output, tau = 0.01):
   
    """
    :param N: number of classes
    :param output: [batch size x 1 x H x W] tensor
    :param tau: temperature parameter

    :return: alpha: [ batch size x N x n_steps H x W] tensor
    
    """
    B = output.shape[0]
    H = output.shape[-2]    # i use - because idk yet if the output is [batch size x 1 x H x W] or [batch size x H x W ]
    W = output.shape[-1]
    n_steps = 100


    weight = torch.arange(1,N + 1)
    b_tensor = torch.rand((B,N,H,W), requires_grad=True)
    b_tensor, _ = torch.sort(b_tensor, dim = 1)  # make sure cut_points is monotonically increasing
    b_tensor = torch.cat([torch.zeros([N,H,W]), -b_tensor], dim = 0)
    b_tensor = torch.cumsum(b_tensor,dim = 1)
    
    x_tensor = torch.linspace(0,1, steps = n_steps)

    alpha = torch.empty(B, N, n_steps, H, W)


    # now we calculate all the functions for all the pixels
    for b in range(B):
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    for x in range(n_steps):
                        
                        alpha[b, n, x, h, w] = (weight[n]*x_tensor[x] + b_tensor[b,n,h,w])/tau

    '''                   
    for n in range(N):
        for h in range(H):
            for w in range(W):
                for x in range(n_steps):
                    
                    alpha[n, x, h, w] = (weight[n]*x_tensor[x] + b_tensor[n,h,w])/tau
    '''
    softmax = torch.nn.Softmax(dim=2) # softmax over the n_steps dimension
    alpha = softmax(alpha)

    
    return alpha


def get_masks(alpha):

    """
    :param alpha: [ batch size x N x n_steps H x W] tensor

    :return: masks: [ batch size x N x H x W] tensor
    
    """
    B = alpha.shape[0]
    N = alpha.shape[1]
    
    H = alpha.shape[3]
    W = alpha.shape[4]

    masks = torch.empty(B, N, H, W)

    for b in range(B):
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    if alpha[b,n,-1,h,w] > 0.9:
                        masks[b,n,h,w] = 1
                    else:
                        masks[b,n,h,w] = 0
        
    """
    for n in range(N):
            for h in range(H):
                for w in range(W):
                    if alpha[n,-1,h,w] > 0.9:
                        masks[n,h,w] = 0
                    else:
                        masks[n,h,w] = 1
    """
    return masks


def mask_events_with_alpha_maps(alpha_mask, event_list):
    """
    :param alpha_mask: [ batch size x N_classes x H x W] tensor
    :param event_list: [ batch size x N_events x 4] input events (ts, y, x, p) ASSUMPTION: events coordinates are integers
    
    :return masked_event_list: [batch size x N_classes x N x 4] input masked events (ts, y, x, p) 
    """
    B = alpha_mask.shape[0]
    N_classes = alpha_mask.shape[1]
    H = alpha_mask.shape[2]
    W = alpha_mask.shape[3]
    N_events = event_list.shape[1]
    

    masked_event_list = None

    for b in range(B):
        for n in range(N_classes):
            for h in range(H):
                for w in range(W):
                    if alpha_mask[b,n,h,w] == 1:
                        for e in range(N_events):
                            if event_list[b,e,1] == h and event_list[b,e,2] == w:
                                if masked_event_list is None:
                                    masked_event_list = event_list[b,e,:]
                                else:
                                    masked_event_list = torch.cat([masked_event_list, event_list[b,e,:]], dim = 0)
                            else:
                                continue
                    else:
                        continue

def associate_events_with_motion_models(alpha_mask, event_list, motion_models):
    """
    :param alpha_mask: [ batch size x N_classes x H x W] tensor
    :param event_list: [ batch size x N_events x 4] input events (ts, y, x, p) ASSUMPTION: events coordinates are integers
    :param motion_models_list: [ batch size x N_classes x 2 x H x W] tensor of flows

    :return flow_list: [batch size x 2 x H x W] tensor with the flow associated to the event
    """
    B = alpha_mask.shape[0]
    N_classes = alpha_mask.shape[1]
    H = alpha_mask.shape[2]
    W = alpha_mask.shape[3]
    N_events = event_list.shape[1]
    

    flow_list = torch.empty((B, 2, H, W))

    for b in range(B):
        for n in range(N_classes):
            for h in range(H):
                for w in range(W):
                    if alpha_mask[b,n,h,w] == 1:
                        for e in range(N_events):
                            if event_list[b,e,1] == h and event_list[b,e,2] == w:
                                
                                flow_list[b,:] = motion_models[b,n,:]

    return flow_list                        

    
    """                   
    for n in range(N_events):
        if alpha_mask[event_list[n,1], event_list[n,2]] == 1:
            if masked_event_list is None:
                masked_event_list = event_list[n,:]
            else:
                masked_event_list = torch.cat([masked_event_list, event_list[n,:]], dim = 0)
        else:
            continue
    """
    return masked_event_list
                
    


    
    
    


if __name__ == '__main__':
    '''
    N = 4
    tau = 0.01
    x = torch.linspace(0,1,steps = 100)
    alpha = torch.zeros(N, x.shape[0])
    cut_off_points = N - 1
    w = torch.arange(1,N + 1)
    b = torch.linspace(0,-1,steps = cut_off_points +2) 
    b = b[b != -1] 

    r = 0
    beta = torch.zeros_like(b)

    for i in range(N):
        beta[i] = b[i] + r
        r += b[i]
    alpha = softmax_binning(alpha, N, w, beta, x)
    print(w)
    print(b)
    plt.close('all')
    plt.figure()
    plt.plot(x, alpha.T)
    plt.show()'''

    #N = 2
    #output = torch.rand((10,10))
    #alpha = softmax_binning_2(N, output)
    a = torch.rand((2,2,2,2))

    

    