import torch
import torch.nn as nn

class Shallow(nn.Module):
    
    def __init__(self, in_dim , h_dim , out_dim):
        super().__init__()
        self.ln1 = nn.Linear( in_dim , h_dim )
        self.act1 =nn.Sigmoid()
        self.ln2 = nn.Linear( h_dim , out_dim , bias=False )
        
    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        return out

class Shallow_ext(nn.Module):
    
    def __init__(self, in_dim , h_dim , out_dim, addiction_features=None):
        super().__init__()
        self.in_dim=in_dim

        if addiction_features is None:
            self.with_af=False
            self.ln1 = nn.Linear( in_dim , h_dim )
        else:
            self.with_af=True
            self.phi=lambda x: torch.stack([af(x) for af in addiction_features],dim=x.ndim-1)
            self.ln1 = nn.Linear( in_dim+ len(addiction_features) , h_dim )
        self.act1 =nn.Sigmoid()
        self.ln2 = nn.Linear( h_dim , out_dim , bias=False )

    def forward(self, x):
        if x.shape[-1]==self.in_dim and self.with_af:
            x=torch.hstack([x,self.phi(x)])
        # otherwise x directly contains addition_features or none
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        return out
    
class Deep_ext(nn.Module):
    def __init__(self, in_dim , h_dim , out_dim, layers=4,addiction_features=None):
        super().__init__()
        self.in_dim=in_dim

        if addiction_features is None:
            self.with_af=False
            # self.ln1 = nn.Linear( in_dim , h_dim )
            layer_list=[self.in_dim]+[h_dim]*layers+[out_dim] 
        else:
            self.with_af=True
            self.phi=lambda x: torch.stack([af(x) for af in addiction_features],dim=x.ndim-1)
            # self.ln1 = nn.Linear( in_dim+ len(addiction_features) , h_dim )
            layer_list=[self.in_dim+len(addiction_features)]+[h_dim]*layers+[out_dim] 

        self.act =nn.Sigmoid()
        self.list=nn.ModuleList()
        for i in range(1,len(layer_list)):
            if i==len(layer_list)-1:
                self.list.append(nn.Linear( h_dim , out_dim , bias=False ))
            else:
                self.list.append(nn.Linear(layer_list[i-1],layer_list[i]))

    def forward(self, x):
        if x.shape[-1]==self.in_dim and self.with_af:
            x=torch.hstack([x,self.phi(x)])
        # otherwise x directly contains addition_features or none
        for i,layer in enumerate(self.list):
            x=layer(x)
            if i!=len(self.list)-1:
                x=self.act(x)
        return x

    
class Shallow_Deep_ext(nn.Module):
    def __init__(self, in_dim , h_dim , out_dim, layers=4,addiction_features=None):
        super().__init__()
        self.in_dim=in_dim

        if addiction_features is None:
            self.with_af=False
            # self.ln1 = nn.Linear( in_dim , h_dim )
            layer_list=[self.in_dim]+[h_dim]*layers+[out_dim] 
        else:
            self.with_af=True
            self.phi=lambda x: torch.stack([af(x) for af in addiction_features],dim=x.ndim-1)
            # self.ln1 = nn.Linear( in_dim+ len(addiction_features) , h_dim )
            layer_list=[self.in_dim+len(addiction_features)]+[h_dim]*layers+[out_dim] 

        self.act =nn.Sigmoid()
        self.list=nn.ModuleList()
        for i in range(1,len(layer_list)):
            if i==len(layer_list)-1:
                self.list.append(nn.Linear( h_dim , out_dim , bias=False ))
            else:
                self.list.append(nn.Linear(layer_list[i-1],layer_list[i]))

    def forward(self, x):
        if x.shape[-1]==self.in_dim and self.with_af:
            x=torch.hstack([x,self.phi(x)])
        # otherwise x directly contains addition_features or none
        x = x0 = self.act(self.list[0](x))
        for i,layer in enumerate(self.list[1:-1]):
            x = self.act(layer(x))
        x = self.list[-1](x+x0)
        return x