import torch

def get_optical_flow(A, H, W):

    """
    Calculates estimated optical flow for a given motion model A
    Input: A: 2x3 motion model
              H: height of image
                W: width of image
    Output: flow: HxWx2 tensor of estimated optical flow

    """
   
    flow = torch.zeros((H, W, 2)) 

    for h in range(H):
        for w in range(W):
            u_x, u_y = torch.matmul(A,torch.tensor([1,h,w]))
            flow[h,w,0] = u_x
            flow[h,w,1] = u_y
    
        
    
    
    return flow


if __name__ == '__main__':
    
    # param flow_list: [batch_size x N_classes x 2 x H x W] list of optical flow (x, y) maps

    batch_size = 1
    N_classes = 3
    H = 4
    W = 5
    flow_list = torch.zeros((batch_size, N_classes, 2, H, W))

    for n in range(N_classes):
        flow_list_loop = flow_list[:,n,:,:,:]
        print(flow_list_loop.shape)
        for i, flow in enumerate(flow_list_loop):
            pass
            print(flow.shape)

    
            