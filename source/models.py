import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, encoder, out_channels=3, features=[64, 128, 256, 512]):
        super(Decoder, self).__init__()
        self.ups = nn.ModuleList()
        self.encoder = encoder
        self.out_channels = out_channels

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = self.encoder.skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels

        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        self.skip_connections = []
        for down in self.downs:
            x = down(x)
            self.skip_connections.append(x)
            x = self.pool(x)

        self.skip_connections = self.skip_connections[::-1]

        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, last_layer=False):
        super(DownConv, self).__init__()

        if not last_layer:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, 0),
            )

    def forward(self, x):
        return self.conv(x)


class BinaryKernel3Classificator(nn.Module):
    def __init__(self):
        super(BinaryKernel3Classificator, self).__init__()

        self.a = DownConv(1, 128, kernel_size=3)
        self.b = DownConv(128, 64, kernel_size=3)
        self.c = DownConv(64, 1, kernel_size=3, last_layer=True)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x.squeeze()


class Kernel3Classificator(nn.Module):
    def __init__(self):
        super(Kernel3Classificator, self).__init__()

        self.a = DownConv(1, 128, kernel_size=3)
        self.b = DownConv(128, 64, kernel_size=3)
        self.c = DownConv(64, 3, kernel_size=3, last_layer=True)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x.squeeze()


class Kernel5Classificator(nn.Module):
    def __init__(self):
        super(Kernel5Classificator, self).__init__()

        self.a = DownConv(1, 64, kernel_size=5)
        self.b = DownConv(64, 3, kernel_size=3, last_layer=True)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)

        return x.squeeze()
        # Don't forget to apply torch.softmax(x, dim=1) when we classify


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        self.a = nn.Sequential(
            nn.Linear(7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.a(x.squeeze().reshape(x.shape[0], -1))

        return x.squeeze()
        # Don't forget to apply torch.softmax(x, dim=1) when we classify


class UNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, state_dict=None, features=[64, 128, 256, 512]
    ):
        super(UNet, self).__init__()
        self.bottleneck_size = features[-1] * 2

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(self.encoder, out_channels, features)
        self.bottleneck = DoubleConv(features[-1], self.bottleneck_size)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if state_dict:
            self.load_from_dict(state_dict)

    def get_statedict(self):
        return {
            "Encoder": self.encoder.state_dict(),
            "Bottleneck": self.bottleneck.state_dict(),
            "Decoder": self.decoder.state_dict(),
            "LastConv": self.final_conv.state_dict(),
        }

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])
        self.bottleneck.load_state_dict(dict["Bottleneck"])
        self.decoder.load_state_dict(dict["Decoder"])
        self.final_conv.load_state_dict(dict["LastConv"])

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return self.final_conv(x)


if __name__ == "__main__":
    x = torch.randn((4, 3, 512, 256))
    y = torch.randn((4, 2, 100))
    model = Kernel3Classificator()
    a = 1
    # model = UNet(in_channels=3, out_channels=3)
    # seg = model(x)
