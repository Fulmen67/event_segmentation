import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

def Maxout(alpha):
    """
    :param alpha: [batch size x N_classes x H x W] tensor
    return: [batch size x N_classes x H x W] tensor, where only the maximum value is 1 and the others are 0
    """
    # Compare the N_classes channels element-wise
    channel_max = torch.max(alpha, dim=1, keepdim=True)[0]

    # Set all other channel to zero
    output = torch.where(alpha == channel_max, alpha, torch.zeros_like(alpha))
    
    return output
    
def softmax_binning_concept(alpha, N, w, b, x, tau = 0.01):
    # N = number of classes
    # alpha = N 

    for i in range(N):
        for j in range(len(x)):
            alpha[i,j] = (w[i]*x[j] + b[i])/tau
        
    softmax = torch.nn.Softmax(dim=0)
    alpha = softmax(alpha)

        
    return alpha


def softmax_binning(N_classes, output, tau = 0.01):
   
    """
    :param N_classes: number of classes
    :param output: [batch size x 1 x H x W] tensor
    :param tau: temperature parameter

    :return: alpha: [ batch size x N_classes x n_steps x H x W] tensor
    
    """
    B = output.shape[0]
    H = output.shape[-2]    # i use - because idk yet if the output is [batch size x 1 x H x W] or [batch size x H x W ]
    W = output.shape[-1]
    n_steps = 100


    weight = torch.arange(1,N_classes + 1, dtype = torch.float32)
    b_tensor = torch.rand((B,N_classes,H,W), requires_grad=True)
    b_tensor, _ = torch.sort(b_tensor, dim = 1)  # make sure cut_points is monotonically increasing
    b_tensor = torch.cat([torch.zeros([B,1,H,W]), -b_tensor], dim = 1)
    b_tensor = torch.cumsum(b_tensor,dim = 1)
    
    x_tensor = torch.linspace(0,1, steps = n_steps, dtype = torch.float32)
    
    
    alpha = torch.empty(B, N_classes, n_steps, H, W)

    # now we calculate all the functions for all the pixels
    
    weight_x = torch.mm(weight.unsqueeze(1),x_tensor.unsqueeze(0))
    """
    for b in tqdm(range(B)):
        for n in tqdm(range(N_classes)):
            for h in range(H):
                for w in range(W):
                    for x in range(n_steps):
                        
                        #alpha[b, n, x, h, w] = (weight[n]*x_tensor[x] + b_tensor[b,n,h,w])/tau
                        alpha[b, n, x, h, w] = (weight_x[n,x] + b_tensor[b,n,h,w])/tau"""
                        
    """
                        : alpha = [batch size x N_classes x n_steps x H x W] = 8 x 2 x 100 x 640 x 480 matrix
                        : weight = [N_classes] = 2 x 1 vector
                        : x_tensor = [n_steps] = 100 x 1 vector
                        : b_tensor = [batch size x N_classes x H x W] = 8 x 2 x 640 x 480 matrix
                        : tau = scalar
                        
                        : weight_x = weight * x^T (2 x 100)
                        
                        """
                     
    for b in range(B):
        for n in range(N_classes):
            b_temp = b_tensor[b,n,:,:].unsqueeze(0).expand(n_steps, H, W)
            w_x_n = weight_x[n,:]
            b_temp = b_temp + w_x_n.expand(H, W, n_steps).permute(2, 0, 1)
            alpha[b,n,:,:,:] = b_temp/tau
        
    softmax = torch.nn.Softmax(dim=2) # softmax over the n_steps dimension
    alpha = softmax(alpha)

    if torch.isnan(alpha).any():
                 print("Nan values in alpha!")
    return alpha


def get_masks(alpha):

    """
    :param alpha: [ batch size x N_classes x n_steps x H x W] tensor

    :return: masks: [ batch size x N_classes x H x W] tensor
    
    """
    B, N, _, H, W = alpha.shape

    masks = torch.zeros(B, N, H, W)
    masks = (alpha[:,:, -1,:,:] > 0.7).type(torch.int)
    
    """
    for b in range(B):
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    if alpha[b,n,-1,h,w] > 0.9:
                        masks[b,n,h,w] = 1
                    else:
                        masks[b,n,h,w] = 0"""
    return masks


def mask_events_with_alpha_maps(alpha_mask, event_list):
    """
    :param alpha_mask: [ batch size x N_classes x H x W] tensor
    :param event_list: [ batch size x N_events x 4] input events (ts, y, x, p) ASSUMPTION: events coordinates are integers
    
    :return masked_event_list: [batch size x N_classes x N x 4] input masked events (ts, y, x, p) 
    """
    B, N_classes, H, W = alpha_mask.shape
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

def associate_events_with_motion_models(event_list, alpha_mask, motion_models):
    """
    :param alpha_mask: [ batch size x N_classes x H x W] tensor
    :param event_list: [ batch size x N_events x 4] input events (ts, y, x, p) ASSUMPTION: events coordinates are integers
    :param motion_models: [ batch size x N_classes x 2 x H x W] tensor of flows

    :return flow_list: [batch size x 2 x H x W] tensor with the flow associated to the event
    """
    B, N_classes, H, W = alpha_mask.shape
    _, N_events, _ = event_list.shape
    
    flow_list = torch.empty((B, 2, H, W))
    
    """
    for b in tqdm(range(B)):
        for n in range(N_classes):
            for h in range(H):
                for w in range(W):
                    if alpha_mask[b,n,h,w] == 1  and torch.isnan(flow_list[b,:,h,w]):
                        for e in range(N_events):
                            if event_list[b,e,1] == h and event_list[b,e,2] == w:
                                
                                flow_list[b,:,h,w] = motion_models[b,n,:,h,w]
    """
    
    # Create a mask indicating which pixels have not yet been associated with a flow
    flow_mask = torch.isnan(torch.zeros((B, 2, H, W)))
    
    for b in range(B):
        for e in range(N_events):
            h, w = event_list[b, e, 1].long().item(), event_list[b, e, 2].long().item()
            if torch.all(flow_mask[b, :, h, w]):
                continue
            for n in range(N_classes):
                if alpha_mask[b, n, h, w] == 1: # ASSUMPTION: the mask with lower index is the one making occlusion to the other
                    flow_list[b, :, h, w] = motion_models[b, n, :, h, w]
                    flow_mask[b, :, h, w] = True
                    break
              
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
    
    return masked_event_list
                
    


    
    
    


if __name__ == '__main__':
    
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
    plt.show()

    #N = 2
    #output = torch.rand((10,10))
    #alpha = softmax_binning_2(N, output)
    a = torch.rand((2,2,2,2))"""

    

    