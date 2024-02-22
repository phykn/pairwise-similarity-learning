import torch
from torch.nn.utils import clip_grad_norm_
from .ema import EMA
from .misc import fix_dtype


class Trainer:
    def __init__(self, args, encoder, simple, dl):
        # parameters
        self.device = args['train']['device']
        self.amp = args['train']['amp']
        self.max_grad_norm = args['train']['max_grad_norm']

        # model
        self.encoder = encoder.to(self.device)
        self.encoder_ema = EMA(self.encoder, decay = args['train']['ema_decay']).to(self.device)
        self.simple = simple.to(self.device)

        # utils
        self.dl = dl

        self.optimizer = torch.optim.SGD([
            {
                'params': self.encoder.parameters(), 
                'lr': args['optimizer']['encoder']['lr'],
                'momentum': args['optimizer']['momentum'],
                'weight_decay': args['optimizer']['encoder']['weight_decay']
            },
            {
                'params': self.simple.parameters(), 
                'lr': args['optimizer']['simple']['lr'],
                'momentum': args['optimizer']['momentum'],
                'weight_decay': args['optimizer']['simple']['weight_decay']
            },
        ])

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer = self.optimizer,
            **args['scheduler']
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled = self.amp)

        # bank
        self.num_bank = args['train']['num_bank']
        self.ptr = 0
        self.x_bank = 0.1 * torch.randn(
            size = (self.num_bank, self.encoder.embed_dim),
            dtype = torch.float32, 
            device = self.device
        )
        self.y_bank = torch.randint(
            low = 0, 
            high = int(1e8), 
            size = (self.num_bank, ),
            dtype = torch.long, 
            device = self.device,
        )

    @torch.no_grad()
    def update_bank(self, i, y):
        batch_size = len(i)
        assert self.num_bank % batch_size == 0

        self.x_bank[self.ptr:(self.ptr + batch_size)] = self.encoder_ema(i)
        self.y_bank[self.ptr:(self.ptr + batch_size)] = y

        self.ptr = (self.ptr + batch_size) % self.num_bank

    def step(self):
        # train mode
        self.encoder.train()
        self.encoder_ema.train()
        self.simple.train()

        # load data
        data = next(self.dl)
        data = {k: v.to(self.device) for k, v in data.items()}
        data = fix_dtype(data)

        i = data['image']
        y = data['label']

        # update bank
        with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.amp):
            self.encoder_ema.update()
            self.update_bank(i, y)

        # forward
        with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.amp):
            x = self.encoder(i)
            loss = self.simple(x, y, self.x_bank, self.y_bank)

        # backward
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        e_params = list(self.encoder.parameters())
        s_params = list(self.simple.parameters())        
        clip_grad_norm_(
            e_params + s_params, 
            max_norm = self.max_grad_norm, 
            norm_type = 2
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()