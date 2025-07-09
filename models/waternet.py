
import torch
import torch.nn as nn
import torch.nn.functional as F

class FTU(nn.Module):
    def __init__(self):
        super(FTU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3), nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 1)  
        )

    def forward(self, x):
        return self.model(x)


class ConfidenceMap(nn.Module):
    def __init__(self):
        super(ConfidenceMap, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(12, 128, 7, padding=3), nn.ReLU(),
            nn.Conv2d(128, 128, 5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 7, padding=3), nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

class WaterNet(nn.Module):
    def __init__(self):
        super(WaterNet, self).__init__()
        self.ftu_wb = FTU()
        self.ftu_he = FTU()
        self.ftu_gc = FTU()
        self.conf_map = ConfidenceMap()

    def forward(self, raw, wb, he, gc):
        r_wb = self.ftu_wb(wb)
        r_he = self.ftu_he(he)
        r_gc = self.ftu_gc(gc)

        fusion_input = torch.cat([raw, wb, he, gc], dim=1)
        conf = self.conf_map(fusion_input)

        output = conf[:, 0:1] * r_gc + conf[:, 1:2] * r_he + conf[:, 2:3] * r_wb
        return output


