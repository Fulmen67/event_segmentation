import torch


def get_optical_flow(A, flow_list, coords):
    """
    Calculates estimated optical flow for a given motion model A
    Input: A: [batch_size x N_classes x 2 x 3]   2x3   (1,x,y)x1
           flow_list: [batch_size x N_classes x 2 x H x W ] list of optical flow (x, y) maps
    Output: flow: batch_size x N_classes x 2 x H x W ]tensor of estimated optical flow
    """
    
    for b in range(A.shape[0]): # B
        for n in range(A.shape[1]): # N_classes  
            flow_list[b,n, :, :, :] = torch.matmul(A[b,n,:], coords.view(3, -1)).view(2, flow_list.shape[3], flow_list.shape[4])
    return flow_list

def get_optical_flow_old(A, flow_list):
    """
    Calculates estimated optical flow for a given motion model A
    Input: A: [batch_size x N_classes x 2 x 3]   2x3   (1,x,y)x1
           flow_list: [batch_size x N_classes x 2 x H x W ] list of optical flow (x, y) maps
    Output: flow: batch_size x N_classes x 2 x H x W ]tensor of estimated optical flow
    """
    B, N_classes, H, W = flow_list.shape
    
    for b in range(A.shape[0]): # B
        for n in range(A.shape[1]): # N_classes
            for h in range(H): 
                for w in range(W):
                    u_x, u_y = torch.matmul(A[b,n,:],torch.tensor([1,h,w], device = A.device, dtype = A.dtype))
                    flow_list[b,n,h,w,0] = u_x
                    flow_list[b,n,h,w,1] = u_y     
            
    return flow_list